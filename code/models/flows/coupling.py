# -*- coding: utf-8 -*-
import torch
from torch import nn
# Internal imports
from .flow import Flow
from .layers import Conv2d, Conv2dZeros, sum_dims


class AffineCouplingFlow(Flow):
    """
    Affine coupling flow layer as defined in
    Density estimation using Real NVP - Dinh et al. (2016)
    (https://arxiv.org/pdf/1605.08803).

    This is mostly a scale and shift transform as defined in eqs. (4) and (5)
    of the original paper
    """

    def __init__(self, dim, n_hidden=64, n_layers=2, activation=nn.ReLU, amortized='none', input_dims=1, **kwargs):
        super(AffineCouplingFlow, self).__init__()
        self.k = dim // 2
        if (input_dims == 1):
            self.g_mu = self.transform_net(self.k, self.k, n_hidden, n_layers, activation)
            self.g_sig = self.transform_net(self.k, self.k, n_hidden, n_layers, activation)
            self.init_parameters()
        else:
            self.g_mu = self.transform_conv(self.k, self.k, n_hidden)
            self.g_sig = self.transform_conv(self.k, self.k, n_hidden)
        self.bijective = True

    def n_parameters(self):
        return sum(param.numel() for param in self.parameters())

    def set_parameters(self, params, batch_dim):
        i = 0
        for param in self.parameters():
            j = i + param.numel()
            param.data = params[:, i:j].reshape(param.shape)
            i = j

    def transform_net(self, nin, nout, nhidden, nlayer, activation):
        net = nn.ModuleList()
        for l in range(nlayer):
            net.append(nn.Linear(l==0 and nin or nhidden, (l==nlayer-1) and nout or nhidden))
            if (l < nlayer - 1):
                net.append(activation())
        return nn.Sequential(*net)

    def transform_conv(self, nin, nout,  nhidden):
        return nn.Sequential(
                Conv2d(nin, nhidden), nn.ReLU(inplace=False),
                Conv2d(nhidden, nhidden, kernel_size=[1, 1]), nn.ReLU(inplace=False),
                Conv2dZeros(nhidden, nout))

    def _call(self, z):
        """ Forward a batch to apply flow """
        z_k, z_D = z[:, :self.k], z[:, self.k:]
        zp_D = z_D * torch.exp(self.g_sig(z_k)) + self.g_mu(z_k)
        return torch.cat((z_k, zp_D), dim = 1)

    def _inverse(self, z):
        """ Apply inverse flow """
        zp_k, zp_D = z[:, :self.k], z[:, self.k:]
        z_D = (zp_D - self.g_mu(zp_k)) / torch.exp(self.g_sig(zp_k))
        return torch.cat((zp_k, z_D), dim = 1)

    def log_abs_det_jacobian(self, z):
        """ Compute log det Jacobian of flow """
        z_k = z[:, :self.k]
        sig = torch.exp(self.g_sig(z_k))
        return sum_dims(torch.log(torch.abs(sig)))


class MaskedCouplingFlow(Flow):
    """
    Masked affine coupling flow layer as defined in
    Density estimation using Real NVP - Dinh et al. (2016)
    (https://arxiv.org/pdf/1605.08803).

    This is mostly a scale and shift transform but it relies on masking rather
    than having to split the dimensions (as in eq. (8) of the original paper)
    """

    def __init__(self, dim, mask=None, n_hidden=64, n_layers=2, activation=nn.ReLU, amortized='none', **kwargs):
        super(MaskedCouplingFlow, self).__init__()
        self.k = dim // 2
        self.g_mu = self.transform_net(dim, dim, n_hidden, n_layers, activation)
        self.g_sig = self.transform_net(dim, dim, n_hidden, n_layers, activation)
        self.mask = mask or torch.cat((torch.ones(self.k), torch.zeros(self.k))).detach()
        self.init_parameters()
        self.bijective = True
        # Register buffers for CUDA call
        #self.register_buffer('mask_p', self.mask)

    def transform_net(self, nin, nout, nhidden, nlayer, activation):
        net = nn.ModuleList()
        for l in range(nlayer):
            module = nn.Linear(l==0 and nin or nhidden, l==nlayer-1 and nout or nhidden)
            nn.init.xavier_normal_(module.weight.data)
            module.bias.data.fill_(0)
            net.append(module)
            if (l < nlayer-1):
                net.append(activation())
        return nn.Sequential(*net)

    def _call(self, z):
        """ Forward a batch to apply flow """
        self.mask = self.mask.to(z.device)
        z_k = (self.mask * z)
        scale = torch.exp((1 - self.mask) * self.g_sig(z_k))
        bias = (1 - self.mask) * self.g_mu(z_k)
        return z * scale + bias

    def _inverse(self, z):
        """ Apply inverse flow """
        zp_k = (self.mask * z)
        scale = torch.exp(-((1 - self.mask) * self.g_sig(zp_k)))
        bias = (1 - self.mask) * self.g_mu(zp_k)
        return (z - bias) + scale

    def log_abs_det_jacobian(self, z):
        """ Compute log det Jacobian of flow """
        self.mask = self.mask.to(z.device)
        sig = torch.exp((1 - self.mask) * self.g_sig(z * self.mask))
        return sum_dims(torch.log(torch.abs(sig)))


class ConvolutionalCouplingFlow(Flow):
    """
    Convolutional version of the affine coupling flow layer
    """

    def __init__(self, dim, n_hidden=64, n_layers=3, activation=nn.ReLU, amortized='none', input_dims=3, **kwargs):
        super(ConvolutionalCouplingFlow, self).__init__()
        self.k = dim // 2
        self.g_net = self.transform_net(self.k, self.k * 2, n_hidden)
        self.bijective = True

    def transform_net(self, nin, nout,  nhidden):
        return nn.Sequential(
                Conv2d(nin, nhidden), nn.ReLU(inplace=False),
                Conv2d(nhidden, nhidden, kernel_size=[1, 1]), nn.ReLU(inplace=False),
                Conv2dZeros(nhidden, nout))

    def _call(self, z):
        """ Forward a batch to apply flow """
        z_k, z_D = z[:, :self.k], z[:, self.k:]
        mu, sig = self.g_net(z_k).chunk(2, dim = 1)
        sig = torch.exp(sig)
        zp_D = (z_D * sig) + mu
        return torch.cat((z_k, zp_D), dim = 1)

    def _inverse(self, z):
        """ Apply inverse flow """
        zp_k, zp_D = z[:, :self.k], z[:, self.k:]
        mu, sig = self.g_net(zp_k).chunk(2, dim = 1)
        sig = torch.exp(sig)
        z_D = (zp_D - mu) / sig
        return torch.cat((zp_k, z_D), dim = 1)

    def log_abs_det_jacobian(self, z):
        """ Compute log det Jacobian of flow """
        z_k = z[:, :self.k]
        mu, sig = self.g_net(z_k).chunk(2, dim = 1)
        sig = torch.exp(sig)
        return sum_dims(torch.log(torch.abs(sig)))
