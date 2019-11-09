# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from pyworld import dio
import numpy as np
import crepe
    
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
    
class FundamentalFrequency(nn.Module):
    """
    Compute the fundamental frequency of a signal
    
    Arguments:
            sr (int)            : sample rate of the signal
            block_size (int)    : size of a block of conditionning
            sequence_size (int) : size of the conditioning sequence
    """
    
    def __init__(self, sr, block_size, sequence_size):
        super(Analysis, self).__init__()
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
        return f0[:self.sequence_size].astype(np.float)
    
class Louodness(nn.Module):
    """
    Compute the loudness of a signal
    
    Arguments:
            block_size (int)    : size of a block of conditionning
            sequence_size (int) : size of the conditioning sequence
    """
    
    def __init__(self, block_size, kernel_size):
        super(Analysis, self).__init__()
        self.kernel_size = kernel_size
        self.block_size = block_size

    def forward(self, x):
        win = np.hamming(self.kernel_size)
        x = x.reshape(-1, self.block_size)
        lo = np.log(np.mean(x ** 2, -1) + 1e-15)
        lo = lo.reshape(-1)
        lo = np.convolve(lo, win, "same")
        return lo
