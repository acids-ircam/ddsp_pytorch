# -*- coding: utf-8 -*-
import torch
# Internal imports
from .flow import Flow
from .layers import sum_dims, amortized_ones, amortized_zeros


class BatchNormFlow(Flow):
    """
    Batch norm flow layer as defined in
    Density estimation using Real NVP - Dinh et al. (2016)
    (https://arxiv.org/pdf/1605.08803).
    """

    def __init__(self, dim, momentum=0.95, eps=1e-5, amortized='none'):
        super(BatchNormFlow, self).__init__()
        self.dim = dim
        self.amortized = amortized
        # Running batch statistics
        self.r_mean = torch.zeros(dim)
        self.r_var = torch.ones(dim)
        # Momentum
        self.momentum = momentum
        self.eps = eps
        # Trainable scale and shift (cf. original paper)
        self.log_gamma = amortized_ones(amortized, (dim,))
        self.beta = amortized_zeros(amortized, (dim,))
        # Register buffers for CUDA call
        self.register_buffer('r_m', self.r_mean)
        self.register_buffer('r_v', self.r_var)

    def _call(self, z):
        if self.training:
            self.r_mean = self.r_mean.to(z.device)
            self.r_var = self.r_var.to(z.device)
            # Current batch stats
            self.b_mean = z.mean(0)
            self.b_var = (z - self.b_mean).pow(2).mean(0) + self.eps
            # Running mean and var
            self.r_mean = self.momentum * self.r_mean + ((1 - self.momentum) * self.b_mean)
            self.r_var = self.momentum * self.r_var + ((1 - self.momentum) * self.b_var)
            mean = self.b_mean
            var = self.b_var
        else:
            mean = self.r_mean
            var = self.r_var
        x_hat = (z - mean) / var.sqrt()
        y = torch.exp(self.log_gamma) * x_hat + self.beta
        return y

    def _inverse(self, z):
        if self.training:
            mean = self.b_mean
            var = self.b_var
        else:
            mean = self.r_mean
            var = self.r_var
        x_hat = (z - self.beta) / torch.exp(self.log_gamma)
        y = x_hat * var.sqrt() + mean
        return y

    def log_abs_det_jacobian(self, z):
        # Here we only need the variance
        mean = z.mean(0)
        var = (z - mean).pow(2).mean(0) + self.eps
        log_det = self.log_gamma - 0.5 * torch.log(var + self.eps)
        return sum_dims(log_det)

    def set_parameters(self, params, batch_dim = 64):
        """ Set parameters values (sub-modules) """
        if (self.amortized in ['input', 'self', 'ext']):
            self.log_gamma = params[:, :self.dim]
            self.beta = params[:, self.dim:(self.dim * 2)]
        if (self.amortized == 'self'):
            self.log_gamma = self.log_gamma.repeat(batch_dim, 1)
            self.beta = self.beta.repeat(batch_dim, 1)
            
    def n_parameters(self):
        """ Return number of parameters in flow """
        return self.dim * 2;


class ActNormFlow(Flow):
    """
    An implementation of the activation normalization layer defined in
    Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, dim, amortized='none'):
        super(ActNormFlow, self).__init__()
        self.weight = []
        self.bias = []
        self.amortized = amortized
        self.weight = amortized_ones(amortized, (1, dim, 1, 1))
        self.bias = amortized_zeros(amortized, (1, dim, 1, 1))
        self.initialized = False
        self.dim = dim

    def _call(self, z):
        return z*torch.exp(self.weight) + self.bias

    def _inverse(self, z):
        return (z - self.bias)*torch.exp(-self.weight)

    def log_abs_det_jacobian(self, z):
        # Data dependent init
        if self.initialized == False:
            self.bias.data.copy_(z.mean((0, 2, 3), keepdim=True) * -1)
            self.weight.data.copy_(torch.log(1.0 / (torch.sqrt(((z + self.bias.data) ** 2).mean((0, 2, 3), keepdim=True)) + 1e-6)))
            self.initialized = True
        return torch.sum(self.weight).repeat(z.shape[0], 1) * z.shape[2] * z.shape[3]

    def set_parameters(self, params, batch_dim):
        """ Set parameters values (sub-modules) """
        if self.amortized != 'none':
            self.weight = params[:, :(self.dim**2)]
            self.bias = params[:, (self.dim**2):(self.dim**2)*2]

    def n_parameters(self):
        """ Return number of parameters in flow """
        return self.dim * 2

