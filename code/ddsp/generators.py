# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from ddsp.synth import SynthModule
    
class Generator(SynthModule):
    """
    Generic class for trainable signal generators.
    """
    
    def __init__(self):
        super(Generator, self).__init__()
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        pass

    def forward(self, z):
        z, conditions = z
        return z

class FilteredNoise(Generator):
    """
    Filtered noise generator implemented through frequency sampling as described in
    Engel et al. "DDSP: Differentiable Digital Signal Processing"
    https://openreview.net/pdf?id=B1x1ma4tDr
    
    Arguments:
            filter_size (int)   : size of the filter (number of coefficients)
            block_size (int)    : size of the block
    """
    
    def __init__(self, filter_size, block_size):
        super(Generator, self).__init__()
        self.apply(self.init_parameters)
        self.block_size = block_size
        self.filter_size = filter_size
        self.noise_att = 1e-4
        self.filter_window = nn.Parameter(torch.hann_window(filter_size).roll(filter_size//2,-1),requires_grad=False)
        self.filter_coef = None
    
    def n_parameters(self):
        return self.filter_size // 2 + 1
    
    def init_parameters(self, m):
        pass
    
    def set_parameters(self, params, batch_dim=64):
        self.filter_coef = params

    def forward(self, z):
        sig, conditions = z
        # Create noise source
        noise = torch.randn([sig.shape[0], sig.shape[1], self.block_size]).detach().to(sig.device).reshape(-1, self.block_size) * self.noise_att
        S_noise = torch.rfft(noise, 1).reshape(sig.shape[0], -1, self.block_size // 2 + 1, 2)
        # Reshape filter coefficients to complex form
        filter_coef = self.filter_coef.reshape([-1, self.filter_size // 2 + 1, 1]).expand([-1, self.filter_size // 2 + 1, 2]).contiguous()
        filter_coef[:,:,1] = 0
        # Compute filter windowed impulse response
        h = torch.irfft(filter_coef, 1, signal_sizes=(self.filter_size,))
        h_w = self.filter_window.unsqueeze(0) * h
        h_w = nn.functional.pad(h_w, (0, self.block_size - self.filter_size), "constant", 0)
        # Compute the spectral mask
        H = torch.rfft(h_w, 1).reshape(sig.shape[0], -1, self.block_size // 2 + 1, 2)
        # Filter the original noise
        S_filtered          = torch.zeros_like(H)
        S_filtered[:,:,:,0] = H[:,:,:,0] * S_noise[:,:,:,0] - H[:,:,:,1] * S_noise[:,:,:,1]
        S_filtered[:,:,:,1] = H[:,:,:,0] * S_noise[:,:,:,1] + H[:,:,:,1] * S_noise[:,:,:,0]
        S_filtered          = S_filtered.reshape(-1, self.block_size // 2 + 1, 2)
        # Inverse the spectral noise back to signal
        filtered_noise = torch.irfft(S_filtered, 1)[:,:self.block_size].reshape(sig.shape[0], -1)
        return filtered_noise