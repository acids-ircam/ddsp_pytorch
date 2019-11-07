# -*- coding: utf-8 -*-
import torch
from torch import nn
# Internal imports
from .flow import Flow
from .layers import MaskedLinear, sum_dims
from .naf import DeepSigmoidFlow, DeepDenseSigmoidFlow


__all__ = [
    'ContextIAFlow',
    'IAFlow',
    'DDSF_IAFlow'
]


class IAFlow(Flow):

    """
    Inverse autoregressive flow layer as defined in
    Improved Variational Inference with Inverse Autoregressive Flow - Kingma et al. (2016)
    (https://arxiv.org/pdf/1606.04934v2.pdf).

    This layer is implemented following the "numerically stable" version
    proposed in the original paper (rather than the vanilla flow), which is
    defined in equations (12), (13) and (14)
    """

    def __init__(
        self, dim, n_hidden=0, n_layers=2, activation=nn.ELU,
        amortized='none', forget_bias=2.
    ):
        super(IAFlow, self).__init__()
        if (n_hidden == 0):
            n_hidden = dim
        # Create auto-regressive nets
        self.ar_net = self.transform_net(dim, n_hidden, activation)
        # Add mean and std net
        self.g_mu = MaskedLinear(
            n_hidden, dim, self.create_mask(n_hidden, dim, dim, 'output'),
            zero_diag=True
        )
        self.g_sig = MaskedLinear(
            n_hidden, dim, self.create_mask(n_hidden, dim, dim, 'output'),
            zero_diag=True
        )
        self.forget_bias = forget_bias
        self.init_parameters()
        self.bijective = True
        self.sigma = []

    def create_mask(self, in_f, out_f, in_flow, m_type=None):
        in_degrees = torch.arange(in_f) % (in_flow - 1)
        out_degrees = torch.arange(out_f) % (in_flow - 1)
        if m_type == 'input':
            in_degrees = torch.arange(in_f) % in_flow
        if m_type == 'output':
            out_degrees = torch.arange(out_f) % in_flow - 1
        return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()

    def transform_net(self, dim, n_hidden, activation):
        input_mask = self.create_mask(dim, n_hidden, dim, 'input')
        output_mask = self.create_mask(n_hidden, n_hidden, dim)
        net = nn.Sequential(
            MaskedLinear(dim, n_hidden, input_mask),
            activation(),
            MaskedLinear(n_hidden, n_hidden, output_mask),
            activation()
        )
        return net

    def _call(self, z):
        """ Forward a batch to apply flow """
        out = self.ar_net(z)
        mu = self.g_mu(out)
        log_sig = torch.exp(self.g_sig(out))
        gate = torch.sigmoid(log_sig + self.forget_bias)
        z = gate * z + ((1 - gate) * mu)
        return z

    def _inverse(self, z):
        """ Apply inverse flow """
        z = (z - self.g_mu(z)) / self.g_sig(z)
        return z

    def log_abs_det_jacobian(self, z):
        """ Compute log det Jacobian of flow """
        out = self.ar_net(z)
        log_sig = torch.exp(self.g_sig(out))
        gate = torch.sigmoid(log_sig + self.forget_bias)
        return sum_dims(torch.log(gate + 1e-5))#.repeat(z.shape[0], 1)


class ContextIAFlow(IAFlow):

    """
    Context-based version of the Inverse Autoregressive Flow
    """

    def __init__(self, dim, n_hidden=32, n_layers=2, activation=nn.ELU, amortized='none', forget_bias=1.):
        super(ContextIAFlow, self).__init__(dim, n_hidden, n_layers, activation, amortized, forget_bias)
        self.context_net = self.transform_net(n_hidden, n_hidden, activation)

    def _call(self, z, h=None):
        """ Forward a batch to apply flow """
        # Transform input
        out = self.ar_net(z)
        # Add context
        out += h
        out = self.context_net(out)
        # Find mu and sig
        mu = self.g_mu(out)
        log_sig = self.g_sig(out)
        gate = torch.sigmoid(log_sig + self.forget_bias)
        z = gate * z + (1 - gate) * mu
        return z

    def _inverse(self, z, h=None):
        """ Apply inverse flow """
        z = (z - self.g_mu(z)) / self.g_sig(z)
        return z

    def log_abs_det_jacobian(self, z, h=None):
        """ Compute log det Jacobian of flow """
        out = self.ar_net(z)
        # Add context
        out += h
        out = self.context_net(out)
        log_sig = self.g_sig(out)
        gate = torch.sigmoid(log_sig + self.forget_bias)
        return torch.sum(torch.log(gate + 1e-5), -1, keepdim=True)


class DDSF_IAFlow(IAFlow):

    """
    Deep Sigmoid Flow version of IAF
    Neural Autoregressive Flow - Huang et al. (2018)
    (https://arxiv.org/pdf/1705.07057).
    """

    def __init__(
        self, dim, ddsf=None, n_layers=2, activation=nn.ELU,
        amortized='none', forget_bias=1.
    ):
        if ddsf is None:
            print(amortized)
            ddsf = DeepDenseSigmoidFlow(dim, amortized=amortized)
        super().__init__(
            dim, ddsf.n_parameters(), n_layers, activation, amortized, forget_bias
        )
        self.ddsf = ddsf
        self._cache = None  # Cache for log_abs_det_jacobian

    def _call(self, z):
        params = self.ar_net(z)
        self.ddsf.set_parameters(params)
        output = self.ddsf(z)
        self._cache = {z: self.ddsf.log_abs_det_jacobian(z)}
        return output 

    def log_abs_det_jacobian(self, z):
        if self._cache is None or z not in self._cache:
            _ = self(z)
        return self._cache[z]

