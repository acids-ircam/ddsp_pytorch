# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from modules import ResConv1d
    
class Filter(nn.Module):
    """
    Generic filtering class, to be used in DDSP
    """
    
    def __init__(self):
        super(Filter, self).__init__()
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        pass

    def forward(self, x):
        pass

"""
#######################

STFT-Masking based trainable filters

#######################
"""
class FIRFilter(Filter):
    """
    FIR filter block implemented through frequency sampling as described in
    Engel et al. "DDSP: Differentiable Digital Signal Processing"
    https://openreview.net/pdf?id=B1x1ma4tDr
    
    Arguments:
            conv (nn.Conv1d)            : convolution module to wrap
            window_name (str or None)   : name of the window used to smooth the convolutions
            squared (bool)              : if `True`, square the smoothing window
    """
    
    def __init__(self, in_s, out_s, n_layers=10, n_dil=5, channels=64):
        super(NeuralSourceFilter, self).__init__()
        # Create modules
        modules = nn.Sequential()
        modules.add(nn.Linear(in_s, 128))
        for l in n_layers:
            dil = (2 ** (l % n_dil))
            in_s = (l==0) and 1 or channels
            out_s = (l == n_layers - 1) and 1 or channels
            modules.add_module('c%i'%l, ResConv1d(in_s, out_s, kernel, stride, pad, dilation = dil))
            if (l < n_layers - 1):
                modules.add_module('b2%i'%l, nn.BatchNorm2d(out_s))
                modules.add_module('a2%i'%l, nn.ReLU())
                modules.add_module('d2%i'%l, nn.Dropout2d(p=.25))
            size[0] = int((size[0]+2*pad-(dil*(kernel-1)+1))/stride+1)
            size[1] = int((size[1]+2*pad-(dil*(kernel-1)+1))/stride+1)
        modules.add(nn.Linear(128, out_s))
        self.net = modules
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        """ Initialize internal parameters (sub-modules) """
        m.data.uniform_(-0.01, 0.01)

    def forward(self, x):
        out = x
        for m in range(len(self.net)):
            out = self.net[m](out)
        return out + x

"""
#######################

Generic trained filters (based on traditional convolution approach)

#######################
"""
class NeuralSourceFilter(Filter):
    """
    Neural filter block as described in
    Wang, X. Takaki, S. and Yamagishi, J. "Neural source-filter waveform models 
    for statistical parametric speech synthesis"
    https://arxiv.org/pdf/1904.12088.pdf
    
    Arguments:
            conv (nn.Conv1d)            : convolution module to wrap
            window_name (str or None)   : name of the window used to smooth the convolutions
            squared (bool)              : if `True`, square the smoothing window
    """
    
    def __init__(self, in_s, out_s, n_layers=10, n_dil=5, channels=64):
        super(NeuralSourceFilter, self).__init__()
        # Create modules
        modules = nn.Sequential()
        modules.add(nn.Linear(in_s, 128))
        for l in n_layers:
            dil = (2 ** (l % n_dil))
            in_s = (l==0) and 1 or channels
            out_s = (l == n_layers - 1) and 1 or channels
            modules.add_module('c%i'%l, ResConv1d(in_s, out_s, kernel, stride, pad, dilation = dil))
            if (l < n_layers - 1):
                modules.add_module('b2%i'%l, nn.BatchNorm2d(out_s))
                modules.add_module('a2%i'%l, nn.ReLU())
                modules.add_module('d2%i'%l, nn.Dropout2d(p=.25))
            size[0] = int((size[0]+2*pad-(dil*(kernel-1)+1))/stride+1)
            size[1] = int((size[1]+2*pad-(dil*(kernel-1)+1))/stride+1)
        modules.add(nn.Linear(128, out_s))
        self.net = modules
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        """ Initialize internal parameters (sub-modules) """
        m.data.uniform_(-0.01, 0.01)

    def forward(self, x):
        out = x
        for m in range(len(self.net)):
            out = self.net[m](out)
        return out + x