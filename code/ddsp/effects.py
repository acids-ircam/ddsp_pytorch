# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from ddsp.synth import SynthModule
    
class Effects(SynthModule):
    """
    Generic class for effects
    """
    
    def __init__(self):
        super(Effects, self).__init__()
        self.apply(self.init_parameters)
    
    def n_parameters(self):
        """ Return number of parameters in the module """
        return 0

    def forward(self, z):
        z, conditions = z
        return z
    
class ReverbSimple(Effects):
    """
    Model of a room impulse response. Two parameters controls the shape of the
    impulse reponse: wet/dry amount and the exponential decay.

    Parameters
    ----------

    size: int
        Size of the impulse response (samples)
    """
    def __init__(self, size):
        super(ReverbSimple, self).__init__()
        self.size    = size

        self.impulse  = nn.Parameter(torch.rand(1, size) * 2 - 1, requires_grad=False)
        self.identity = nn.Parameter(torch.zeros(1,size), requires_grad=False)

        self.impulse[:,0]  = 0
        self.identity[:,0] = 1
        # Amount of reverb and decay parameter
        self.wetdry = nn.Parameter(torch.Tensor([2]), requires_grad=False)
        self.decay  = nn.Parameter(torch.Tensor([4]), requires_grad=False)

    def forward(self, z):
        z, _ = z
        wetdry = self.wetdry
        decay  = self.decay
        idx = torch.sigmoid(wetdry) * self.identity
        imp = torch.sigmoid(1 - wetdry) * self.impulse
        dcy = torch.exp(-(torch.exp(decay)+2) * torch.linspace(0,1, self.size).to(z.device))
        return idx + imp * dcy

    def n_parameters(self):
        """ Return number of parameters in the module """
        return 0


    
class Reverb(Effects):
    """
    Perform reverb as a simplified room impulse response (decay function). 
    Two trainable parameters are used to control the impulse reponse: 
        - wet amount 
        - exponential decay

    Arguments:
    ----------
        length (int)        : Number of samples of the impulse response
    """
    
    def __init__(self, args):
        super(Reverb, self).__init__()
        self.block_size = args.block_size
        self.sequence_size = args.sequence_size
        self.size = self.block_size * self.sequence_size
        # Decay parameter
        # self.decay   = nn.Parameter(torch.Tensor([2]), requires_grad=True)
        # Amount of reverb
        # self.wetdry  = nn.Parameter(torch.Tensor([4]), requires_grad=True)
        self.apply(self.init_parameters)
        # Impulse response of the reverb
        self.impulse  = nn.Parameter(torch.rand(1, self.size) * 2 - 1, requires_grad=False)
        self.identity = nn.Parameter(torch.zeros(1, self.size), requires_grad=False)
        self.identity[:,0] = 1

    def n_parameters(self):
        """ Return number of parameters in the module """
        return 2
    
    def set_parameters(self, params, batch_dim=64):
        params = torch.mean(params, 1)
        self.decay = params[:, 0].unsqueeze(-1)
        self.wetdry = params[:, 1].unsqueeze(-1)

    def forward(self, z):
        z, conditions = z
        # Pad the input sequence
        y = nn.functional.pad(z, (0, self.size), "constant", 0)
        # Compute STFT
        Y_S = torch.rfft(y, 1)
        # Compute the current impulse response
        idx = torch.sigmoid(self.wetdry) * self.identity
        imp = torch.sigmoid(1 - self.wetdry) * self.impulse
        dcy = torch.exp(-(torch.exp(self.decay) + 2) * torch.linspace(0,1, self.size).to(z.device))
        final_impulse = idx + imp * dcy
        # Pad the impulse response
        impulse = nn.functional.pad(final_impulse, (0, self.size), "constant", 0)
        if y.shape[-1] > self.size:
            impulse = nn.functional.pad(impulse, (0, y.shape[-1] - impulse.shape[-1]), "constant", 0)
        IR_S = torch.rfft(impulse.detach(),1).expand_as(Y_S)
        # Apply the reverb
        Y_S_CONV = torch.zeros_like(IR_S)
        Y_S_CONV[:,:,0] = Y_S[:,:,0] * IR_S[:,:,0] - Y_S[:,:,1] * IR_S[:,:,1]
        Y_S_CONV[:,:,1] = Y_S[:,:,0] * IR_S[:,:,1] + Y_S[:,:,1] * IR_S[:,:,0]
        # Invert the reverberated signal
        y = torch.irfft(Y_S_CONV, 1, signal_sizes=(y.shape[-1],))
        return y