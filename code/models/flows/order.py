# -*- coding: utf-8 -*-
import torch
# Internal imports
from .flow import Flow
from .layers import Conv2dZeros, GaussianDiag


class ReverseFlow(Flow):

    """
    Reverse flow layer as defined in
    Density estimation using Real NVP - Dinh et al. (2016)
    (https://arxiv.org/pdf/1605.08803).
    """

    def __init__(self, dim, amortized='none', **kwargs):
        super(ReverseFlow, self).__init__()
        self.permute = torch.arange(dim - 1, -1, -1)
        self.inverse = torch.argsort(self.permute)

    def _call(self, z):
        return z[:, self.permute]

    def _inverse(self, z):
        return z[:, self.inverse]

    def log_abs_det_jacobian(self, z):
        return torch.zeros(z.shape[0], 1).to(z.device)


class ShuffleFlow(ReverseFlow):

    """
    Shuffle flow layer
    """

    def __init__(self, dim, amortized='none', **kwargs):
        super(ShuffleFlow, self).__init__(dim, **kwargs)
        self.permute = torch.randperm(dim)
        self.inverse = torch.argsort(self.permute)


class SplitFlow(Flow):

    """
    An implementation of the split layer defined in
    Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).

    This flow simply splits the input across the channel dimension.
    """

    def __init__(self, num_channels, amortized='none'):
        super().__init__()
        self.conv = Conv2dZeros(num_channels // 2, num_channels)
        self.eps_std = None

    def split(self, tensor, type_s="split"):
        """ type = ["split", "cross"] """
        C = tensor.size(1)
        if type_s == "split":
            return tensor[:, :C//2, ...], tensor[:, C//2:, ...]
        elif type_s == "cross":
            return tensor[:, 0::2, ...], tensor[:, 1::2, ...]

    def prior(self, z):
        h = self.conv(z)
        return self.split(h, "cross")

    def _call(self, z):
        z1, z2 = self.split(z, "split")
        return z1

    def log_abs_det_jacobian(self, z):
        z1, z2 = self.split(z, "split")
        mean, logs = self.prior(z1)
        log_det = GaussianDiag.logp(mean, logs, z2)
        return log_det

    def _inverse(self, z):
        mean, logs = self.prior(z)
        z2 = GaussianDiag.sample(mean, logs, self.eps_std)
        z = torch.cat((z, z2), dim=1)
        return z


class SqueezeFlow(Flow):

    """
    An implementation of the squeeze layer defined in
    Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).

    This flow simply splits the input across the channel dimension.
    """
    def __init__(self, dim, factor=2, amortized='none'):
        super().__init__()
        self.factor = factor

    def squeeze(self, z, factor=2):
        if factor == 1:
            return z
        B, C, H, W = z.size()
        assert H % factor == 0 and W % factor == 0, "{}".format((H, W))
        x = z.view(B, C, H // factor, factor, W // factor, factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * factor * factor, H // factor, W // factor)
        return x

    def unsqueeze(self, z, factor=2):
        factor2 = factor ** 2
        if factor == 1:
            return z
        B, C, H, W = z.size()
        x = z.view(B, C // factor2, factor, factor, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C // (factor2), H * factor, W * factor)
        return x

    def _call(self, z):
        return self.squeeze(z, self.factor)

    def _inverse(self, z):
        return self.unsqueeze(z, self.factor)

    def log_abs_det_jacobian(self, z):
        return torch.zeros(z.shape[0])

