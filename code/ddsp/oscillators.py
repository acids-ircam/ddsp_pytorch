# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from ddsp.synth import SynthModule
    
class Oscillator(SynthModule):
    
    def __init__(self):
        super(Oscillator, self).__init__()
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        pass

    def forward(self, z):
        pass
    
class HarmonicOscillators(Oscillator):
    
    def __init__(self, n_partial, sample_rate, block_size):
        super(Oscillator, self).__init__()
        self.apply(self.init_parameters)
        self.n_partial = n_partial
        self.upsample = nn.Upsample(scale_factor = block_size, mode="linear", align_corners=False)
        self.k = nn.Parameter(torch.arange(1, n_partial + 1).reshape(1,1,-1).float(), requires_grad=False)
        self.sample_rate = sample_rate
    
    def forward(self, z):
        # Retrieve synth parameters
        z, conditions = z
        f0, loud = conditions
        # Upsample parameters
        f0          = self.upsample(f0.transpose(1,2)).squeeze(1) / self.sample_rate
        amp         = self.upsample(self.amp.transpose(1,2)).squeeze(1)
        alpha       = self.upsample(self.alpha.transpose(1,2)).transpose(1,2)
        # Generate phase
        phi = torch.zeros(f0.shape).to(f0.device)
        for i in np.arange(1,phi.shape[-1]):
            phi[:,i] = 2 * np.pi * f0[:,i] + phi[:,i-1]
        phi = phi.unsqueeze(-1).expand(alpha.shape)
        # Filtering above Nyquist
        anti_alias = (self.k * f0.unsqueeze(-1) < .5).float()
        # Generate the output signal
        y =  amp * torch.sum(anti_alias * alpha * torch.sin(self.k * phi), -1)
        return y
    
    def set_parameters(self, params, batch_dim=64):
        """ Set parameters values (sub-modules) """
        self.amp = params[:, :, 0].unsqueeze(2)
        self.alpha = params[:, :, 1:]

    def n_parameters(self):
        """ Return number of parameters in the module """
        return self.n_partial + 1