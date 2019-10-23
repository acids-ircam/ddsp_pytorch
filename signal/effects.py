# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
    
class Effects(nn.Module):
    
    def __init__(self):
        super(Effects, self).__init__()
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        pass

    def forward(self, x):
        pass
    
class ReverbSimple(Effects):
    """
    Perform reverb as a simplified room impulse response (decay function). 
    Two trainable parameters are used to control the impulse reponse: 
        - wet amount 
        - exponential decay

    Arguments:
    ----------
        length (int)        : Number of samples of the impulse response
    """
    
    def __init__(self, length):
        super(ReverbSimple, self).__init__()
        self.apply(self.init_parameters)
        self.length = length

        self.impulse  = nn.Parameter(torch.rand(1, size) * 2 - 1, requires_grad=False)
        self.identity = nn.Parameter(torch.zeros(1,size), requires_grad=False)

        self.impulse[:,0]  = 0
        self.identity[:,0] = 1

        self.decay   = nn.Parameter(torch.Tensor([2]), requires_grad=True)
        self.wetdry  = nn.Parameter(torch.Tensor([4]), requires_grad=True)

    def forward(self):
        idx = torch.sigmoid(self.wetdry) * torch.linspace()
        imp = torch.sigmoid(1 - self.wetdry) * self.impulse
        dcy = torch.exp(-(torch.exp(self.decay)+2) * \
              torch.linspace(0,1, self.size).to(self.decay.device))

        return idx + imp * dcy