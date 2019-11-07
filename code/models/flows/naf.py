# -*- coding: utf-8 -*-
"""
Neural autoregressive flows (NAF).

Major contributions of the NAF paper is to define a family of sigmoidal flows 
that can be used as transformers in the other autoregressive flows (typically
in the scale-and-shift transform computed in IAF and MAF).

Hence, we can either use directly a Deep Sigmoid Flow (DSF) or the dense version
called Deep Dense Sigmoid Flow (DDSF)

Neural Autoregressive Flow - Huang et al. (2018)
(https://arxiv.org/pdf/1804.00779.pdf).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .flow import Flow, FlowList
from .layers import amortized_init, MaskedLinearAR, sum_dims


__all__ = [
    'DeepDenseSigmoidFlow',
    'DeepSigmoidFlow'
]


def log_mm(log_a, log_b):
    """Compute log(a @ b) in a numerically stable way."""
    log_a = log_a.unsqueeze(-1)
    log_b = log_b.unsqueeze(-3)
    return (log_a + log_b).logsumexp(-2)


class DDSFLayer(Flow):

    """Layer of a DDSF MLP.

    Shape:
        - Input : (batch_dim, dim, in_dim)
        - Output : (batch_dim, dim, out_dim)
    """

    def __init__(self, dim, in_dim, out_dim, amortized='none'):
        super().__init__(amortized)
        # Store dimensions
        self.dim = dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self._n_parameters = dim*(3*out_dim + in_dim)
        # Register statistical parameters
        self.v_u = nn.Parameter(torch.Tensor(1, dim, out_dim, in_dim))
        self.v_w = nn.Parameter(torch.Tensor(1, dim, out_dim, out_dim))
        # Register amortized parameters
        self.a = amortized_init(amortized, (1, dim, out_dim, 1))
        self.b = amortized_init(amortized, (1, dim, out_dim, 1))
        self.eta_u = amortized_init(amortized, (1, dim, 1, in_dim))
        self.eta_w = amortized_init(amortized, (1, dim, 1, out_dim))
        # Init parameters and cache
        self._cache = {}
        self.init_parameters()

    def _call(self, z, cache_log_abs_grad=True):
        # Compute parameters
        u = (self.v_u + self.eta_u).softmax(-1)
        w = (self.v_w + self.eta_w).softmax(-1)
        a, b = F.softplus(self.a), self.b
        # Compute some variables
        log_w = (self.v_w + self.eta_w).log_softmax(-1)
        C = (u @ z.unsqueeze(-1))*a + b
        log_sigm_C = F.logsigmoid(C)
        log_D = log_mm(log_w, log_sigm_C)
        log_1_minus_D = (1 - w @ C.sigmoid()).log()
        log_abs_1_minus_D = (1 - w @ C.sigmoid()).abs().log()
        output = (log_D - log_1_minus_D).squeeze(-1)
        if cache_log_abs_grad:
            # Compute extra variables
            log_u = (self.v_u + self.eta_u).log_softmax(-1)
            # Compute log-gradient
            log_abs_grad = a.squeeze(-1).unsqueeze(-2).log()
            log_abs_grad = log_abs_grad + log_w
            log_abs_grad = log_abs_grad + F.logsigmoid(-C)
            log_abs_grad = log_abs_grad + log_sigm_C.squeeze(-1).unsqueeze(-2)
            log_abs_grad = log_abs_grad - log_D - log_abs_1_minus_D
            log_abs_grad = log_mm(log_abs_grad, log_u)
            # Cache log-gradient
            self._cache = {z: log_abs_grad}
        return output

    def log_abs_grad(self, z):
        if self._cache is None or z not in self._cache:
            _ = self._call(z, cache_log_abs_grad=True)
        return self._cache[z]

    def n_parameters(self):
        return self._n_parameters

    def set_parameters(self, params, batch_dim=1):
        if self.amortized == 'none':
            return
        # Precompute params shapes
        dim, in_dim, out_dim = self.dim, self.in_dim, self.out_dim
        shapes = {
            'a': (-1, dim, out_dim, 1),
            'b': (-1, dim, out_dim, 1),
            'eta_u': (-1, dim, 1, in_dim),
            'eta_w': (-1, dim, 1, out_dim)
        }
        # Store params
        self.a = params[:, :dim*out_dim].reshape(shapes['a'])
        self.b = params[:, dim*out_dim:2*dim*out_dim].reshape(shapes['b'])
        self.eta_w = params[:, 2*dim*out_dim:3*dim*out_dim].reshape(shapes['eta_w'])
        self.eta_u = params[:, 3*dim*out_dim:].reshape(shapes['eta_u'])
        # Repeat along batch axis if self-amortized
        if self.amortized == 'self':
            for param_name in shapes:
                param = getattr(self, param_name).repeat(batch_dim, 1, 1, 1)
                setattr(self, param_name, param)


class DeepDenseSigmoidFlow(FlowList):

    def __init__(self, dim, hidden_dims=(16, 16), amortized='none'):
        # Store dimensions
        layer_dims = [1] + list(hidden_dims) + [1]
        self.dim = dim
        # Build layers
        layers = [
            DDSFLayer(dim, h0, h1, amortized)
            for h0, h1 in zip(layer_dims[:-1], layer_dims[1:])
        ]
        super().__init__(layers)

    def __call__(self, z):
        return super().__call__(z.unsqueeze(-1)).squeeze(-1)

    def log_abs_det_jacobian(self, z):
        z = z.unsqueeze(-1)
        log_abs_grad = torch.zeros(z.shape[0], z.shape[1], 1, 1, device=z.device)
        for layer in self:
            next_out = layer(z)
            log_abs_grad = log_mm(layer.log_abs_grad(z), log_abs_grad)
            z = next_out
        return sum_dims(log_abs_grad[..., 0, 0])


class DeepSigmoidFlow(DeepDenseSigmoidFlow):

    """DDSF with only one hidden layer."""

    def __init__(self, dim, hidden_dim=16, amortized='none'):
        super().__init__(dim, [hidden_dim], amortized)

