# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
    
class Effects(nn.Module):
    """
    Generic class for effects
    """
    
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
        # CONVOLUTION WITH AN IMPULSE RESPONSE #################################
        y = nn.functional.pad(y, (0, preprocess.block_size*preprocess.sequence_size),
                              "constant", 0)

        Y_S = torch.rfft(y,1)

        impulse = self.impulse(reverb, conv_pass)

        impulse = nn.functional.pad(impulse,
                                    (0, preprocess.block_size*preprocess.sequence_size),
                                    "constant", 0)

        if y.shape[-1] > preprocess.sequence_size * preprocess.block_size:
            impulse = nn.functional.pad(impulse,
                                        (0, y.shape[-1]-impulse.shape[-1]),
                                        "constant", 0)
        if conv_pass:
            IR_S = torch.rfft(impulse,1).expand_as(Y_S)
        else:
            IR_S = torch.rfft(impulse.detach(),1).expand_as(Y_S)

        Y_S_CONV = torch.zeros_like(IR_S)
        Y_S_CONV[:,:,0] = Y_S[:,:,0] * IR_S[:,:,0] - Y_S[:,:,1] * IR_S[:,:,1]
        Y_S_CONV[:,:,1] = Y_S[:,:,0] * IR_S[:,:,1] + Y_S[:,:,1] * IR_S[:,:,0]

        y = torch.irfft(Y_S_CONV, 1, signal_sizes=(y.shape[-1],))

        y = y[:,:-preprocess.block_size*preprocess.sequence_size]
        idx = torch.sigmoid(self.wetdry) * torch.linspace()
        imp = torch.sigmoid(1 - self.wetdry) * self.impulse
        dcy = torch.exp(-(torch.exp(self.decay)+2) * \
              torch.linspace(0,1, self.size).to(self.decay.device))

        return idx + imp * dcy