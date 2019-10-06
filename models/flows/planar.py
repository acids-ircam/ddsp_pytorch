# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
# Internal imports
from .flow import Flow
from .layers import amortized_init, sum_dims


class PlanarFlow(Flow):

    """
    Planar normalizing flow, as defined in
    Variational Inference with Normalizing Flows - Rezende et al. (2015)
    http://proceedings.mlr.press/v37/rezende15.pdf
    """

    def __init__(self, dim, amortized='none'):
        """
        Initialize normalizing flow
        """
        super(PlanarFlow, self).__init__(amortized)
        self.weight = amortized_init(amortized, (1, 1, dim))
        self.scale = amortized_init(amortized, (1, dim, 1))
        self.bias = amortized_init(amortized, (1, 1, 1))
        self.init_parameters()
        self.dim = dim

    def _call(self, z):
        if self.amortized == 'none':
            bias = self.bias.repeat(z.shape[0], 1, 1)
            scale = self.scale.repeat(z.shape[0], 1, 1)
            weight = self.weight.repeat(z.shape[0], 1, 1)
        else:
            bias, scale, weight = self.bias, self.scale, self.weight
        z = z.unsqueeze(2)
        f_z = torch.bmm(weight, z) + bias
        return (z + scale * torch.tanh(f_z)).squeeze(2)

    def log_abs_det_jacobian(self, z):
        if self.amortized == 'none':
            bias = self.bias.repeat(z.shape[0], 1, 1)
            scale = self.scale.repeat(z.shape[0], 1, 1)
            weight = self.weight.repeat(z.shape[0], 1, 1)
        else:
            bias, scale, weight = self.bias, self.scale, self.weight
        z = z.unsqueeze(2)
        f_z = torch.bmm(weight, z) + bias
        psi = weight * (1 - torch.tanh(f_z) ** 2)
        det_grad = 1 + torch.bmm(psi, scale)
        return sum_dims(torch.log(det_grad.abs() + 1e-9))

    def set_parameters(self, p_list, batch_dim=64):
        if self.amortized in ('input', 'self', 'ext'):
            self.weight = p_list[:, :self.dim].unsqueeze(1)
            self.scale = p_list[:, self.dim:self.dim*2].unsqueeze(2)
            self.bias = p_list[:, self.dim*2].unsqueeze(1).unsqueeze(2)
        # Handle self or no amortization
        if self.amortized == 'self':
            self.weight = self.weight.repeat(batch_dim, 1, 1)
            self.scale = self.scale.repeat(batch_dim, 1, 1)
            self.bias = self.bias.repeat(batch_dim, 1, 1)
        # Reparametrize scale so that the flow becomes invertible
        #uw = torch.bmm(self.weight, self.scale)
        #m_uw = -1. + F.softplus(uw)
        #w_norm_sq = torch.sum(self.weight**2, dim=2, keepdim=True)
        #self.scale = self.scale + ((m_uw - uw) * self.weight.transpose(2, 1) / w_norm_sq)

    def n_parameters(self):
        return 2 * self.dim + 1

