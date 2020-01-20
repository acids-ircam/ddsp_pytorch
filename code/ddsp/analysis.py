# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from pyworld import dio
import numpy as np
import crepe

# Lambda for computing squared amplitude
amp = lambda x: x[...,0]**2 + x[...,1]**2
    
class Analysis(nn.Module):
    """
    Generic class for trainable analysis modules.
    """
    
    def __init__(self):
        super(Analysis, self).__init__()
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        pass

    def forward(self, x):
        pass
    
class FundamentalFrequency(Analysis):
    """
    Compute the fundamental frequency of a signal
    
    Arguments:
            sr (int)            : sample rate of the signal
            block_size (int)    : size of a block of conditionning
            sequence_size (int) : size of the conditioning sequence
    """
    
    def __init__(self, sr, block_size, sequence_size, method='dio'):
        super(FundamentalFrequency, self).__init__()
        self.method = method
        self.sequence_size = sequence_size
        self.block_size = block_size
        self.sr = sr

    def forward(self, x):
        # Compute the hop length
        hop = int(1000 * self.block_size / self.sr)
        if self.method == "dio":
            f0 = dio(x.astype(np.float64), self.sr,
                     frame_period=hop,
                     f0_floor=50,
                     f0_ceil=2000)[0]
        elif self.method == "crepe":
            f0 = crepe.predict(x, self.sr, step_size=hop, verbose=False)[1]
        return f0.astype(np.float)
    
class Loudness(Analysis):
    """
    Compute the loudness of a signal
    
    Arguments:
            block_size (int)    : size of a block of conditionning
            sequence_size (int) : size of the conditioning sequence
    """
    
    def __init__(self, block_size, kernel_size):
        super(Loudness, self).__init__()
        self.kernel_size = kernel_size
        self.block_size = block_size

    def forward(self, x):
        win = np.hamming(self.kernel_size)
        x = x.reshape(-1, self.block_size)
        lo = np.log(np.mean(x ** 2, -1) + 1e-15)
        lo = lo.reshape(-1)
        lo = np.convolve(lo, win, "same")
        return lo
    
    
class MultiscaleFFT(Analysis):
    """
    Compute the FFT of a signal at multiple scales
    
    Arguments:
            block_size (int)    : size of a block of conditionning
            sequence_size (int) : size of the conditioning sequence
    """
    
    def __init__(self, scales, overlap=0.75, reshape=True):
        super(MultiscaleFFT, self).__init__()
        self.apply(self.init_parameters)
        self.scales = scales
        self.overlap = overlap
        self.reshape = reshape
        self.windows = nn.ParameterList(
                nn.Parameter(torch.from_numpy(np.hanning(scale)).float(), requires_grad=False)\
            for scale in self.scales)
    
    def init_parameters(self, m):
        pass

    def forward(self, x):
        stfts = []
        for i, scale in enumerate(self.scales):
            cur_fft = torch.stft(x, n_fft=scale, window=self.windows[i], hop_length=int((1-self.overlap)*scale), center=False)
            stfts.append(amp(cur_fft))
        if (self.reshape):
            stft_tab = []
            for b in range(x.shape[0]):
                cur_fft = []
                for s, _ in enumerate(self.scales):
                    cur_fft.append(stfts[s][b])
                stft_tab.append(cur_fft)
            stfts = stft_tab
        return stfts
