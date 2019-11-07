# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


"""
Definition of a set of useful lambdas
"""
# Sum a tensor through all dimensions except batch one
sum_dims = lambda x: torch.sum(x, dim=list(torch.arange(1, len(x.shape)))).unsqueeze(1)
# Number of dimensions (excepting batch)
dims = lambda x: torch.cumprod(torch.Tensor(list(x.shape[1:])), 0)[-1]
eps = 1e-6
# Simple functions
softplus_ = nn.Softplus()
softplus = lambda x: softplus_(x) + eps
sigmoid_ = nn.Sigmoid()
sigmoid = lambda x: sigmoid_(x) * (1-eps) + 0.5 * eps
logsigmoid = lambda x: -softplus(-x)
"""
Amortization lambdas
"""
amortized_init = lambda t, d: nn.Parameter(torch.Tensor(*d)) if (t == 'none') else []
amortized_ones = lambda t, d: nn.Parameter(torch.ones(*d)) if (t == 'none') else []
amortized_zeros = lambda t, d: nn.Parameter(torch.zeros(*d)) if (t == 'none') else []


class MaskedLinear(nn.Module):

    def __init__(self, in_dim, out_dim, mask, zero_diag=False):
        super(MaskedLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.uniform_(self.linear.weight.data, -0.0001, 0.0001)
        self.linear.bias.data.fill_(0)
        self.mask = mask.detach()
        if zero_diag:
            for i in range(self.mask.shape[0]):
                mask[i, i] = 0
        # Register buffers for CUDA call
        self.register_buffer('mask_p', self.mask)

    def forward(self, z, h=None):
        self.mask = self.mask.to(z.device)
        output = F.linear(z, self.linear.weight * self.mask, self.linear.bias)
        if h is not None:
            output += h
        return output


class MaskedLinearAR(nn.Linear):

    def init_parameters(self):
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.uniform_(-1, 1)

    def forward(self, input):
        return super()(input, self.weight.tril(diagonal=-1), self.bias)


class Conv2d(nn.Conv2d):

    pad_dict = {
        "same": lambda kernel, stride: [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)],
        "valid": lambda kernel, stride: [0 for _ in kernel]
    }

    @staticmethod
    def get_padding(padding, kernel_size, stride):
        # make paddding
        if isinstance(padding, str):
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            if isinstance(stride, int):
                stride = [stride, stride]
            padding = padding.lower()
            try:
                padding = Conv2d.pad_dict[padding](kernel_size, stride)
            except KeyError:
                raise ValueError("{} is not supported".format(padding))
        return padding

    def __init__(self, in_channels, out_channels, kernel_size=[3, 3], stride=[1, 1], padding="same", do_actnorm=False, weight_std=0.001):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, bias=(not do_actnorm))
        # init weight with std
        self.weight.data.normal_(mean=0.0, std=weight_std)
        if not do_actnorm:
            self.bias.data.zero_()
        else:
            self.actnorm = ActNormFlow(out_channels)
        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = super().forward(input)
        if self.do_actnorm:
            x = self.actnorm(x)
        return x


class Conv2dZeros(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size=[3, 3], stride=[1, 1], padding="same", logscale_factor=3):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        # logscale_factor
        self.logscale_factor = logscale_factor
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class GaussianDiag:

    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean, logs, x):
        """
        lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
              k = 1 (Independent)
              Var = logs ** 2
        """
        return -0.5 * (logs * 2. + ((x - mean) ** 2) / torch.exp(logs * 2.) + GaussianDiag.Log2PI)

    @staticmethod
    def logp(mean, logs, x):
        likelihood = GaussianDiag.likelihood(mean, logs, x)
        return torch.sum(likelihood, dim=[1, 2, 3])

    @staticmethod
    def sample(mean, logs, eps_std=None):
        eps_std = eps_std or 1
        eps = torch.normal(mean=torch.zeros_like(mean),
                           std=torch.ones_like(logs) * eps_std)
        return mean + torch.exp(logs) * eps

