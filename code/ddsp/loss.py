# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.loss import _Loss
    
class Loss(_Loss):
    
    def __init__(self):
        super(Loss, self).__init__()
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        pass

    def forward(self, x):
        pass

# Lambda for computing squared amplitude
amp = lambda x: x[...,0]**2 + x[...,1]**2

class MSSTFTLoss(Loss):
    """
    Compute the FFT of a signal at multiple scales
    
    Arguments:
            block_size (int)    : size of a block of conditionning
            sequence_size (int) : size of the conditioning sequence
    """
    
    def __init__(self, scales, overlap=0.75):
        super(MSSTFTLoss, self).__init__()
        self.apply(self.init_parameters)
        self.scales = scales
        self.overlap = overlap
        self.windows = nn.ParameterList(nn.Parameter(torch.from_numpy(np.hanning(scale)).float(), requires_grad=False) for scale in self.scales)
    
    def init_parameters(self, m):
        pass

    def forward(self, x, stfts_orig):
        stfts = []
        # First compute multiple STFT for x
        for i, scale in enumerate(self.scales):
            cur_fft = torch.stft(x, n_fft=scale, window=self.windows[i], hop_length=int((1-self.overlap)*scale), center=False)
            stfts.append(amp(cur_fft))
        # Compute loss 
        lin_loss = sum([torch.mean(abs(stfts_orig[i][j] - stfts[i][j])) for j in range(len(stfts[i])) for i in range(len(stfts))])
        log_loss = sum([torch.mean(abs(torch.log(stfts_orig[i][j] + 1e-4) - torch.log(stfts[i][j] + 1e-4)))  for j in range(len(stfts[i])) for i in range(len(stfts))])
        return lin_loss + log_loss