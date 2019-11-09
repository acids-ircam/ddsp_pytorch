# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
    
class Loss(nn.Module):
    
    def __init__(self):
        super(Effects, self).__init__()
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        pass

    def forward(self, x):
        pass
    
class MultiscaleFFTLoss(Loss):
    
    def __init__(self):
        super(MultiscaleFFTLoss, self).__init__()
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        pass

    def forward(self, x):
        stfts = []
        for i,scale in enumerate(preprocess.fft_scales):
            stfts.append(amp(
                torch.stft(x,
                           n_fft=scale,
                           window=self.windows[i],
                           hop_length=int((1-overlap)*scale),
                           center=False)
            ))
        return stfts