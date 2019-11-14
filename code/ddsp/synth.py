# -*- coding: utf-8 -*-

import torch.nn as nn
from ddsp.effects import Reverb
from ddsp.oscillators import HarmonicOscillators
from ddsp.generators import FilteredNoise

class SynthModule(nn.Module):
    """
    Generic class defining a synthesis module.
    """

    def __init__(self, amortized='none'):
        """ Initialize as module """
        nn.Module.__init__(self)
        # Handle amortization
        self.amortized = amortized

    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for param in self.parameters():
            param.data.uniform_(-0.001, 0.001)

    def __hash__(self):
        """ Dirty hack to ensure nn.Module compatibility """
        return nn.Module.__hash__(self)

    def set_parameters(self, params, batch_dim):
        """ Set parameters values (sub-modules) """
        pass

    def n_parameters(self):
        """ Return number of parameters in the module """
        return 0

class Synth(nn.ModuleList, SynthModule):
    """
    Generic class for defining a synthesizer (as a list of synthesis modules)
    """

    def __init__(self, modules=[]):
        SynthModule.__init__(self, modules[0].amortized if modules else 'none')
        nn.ModuleList.__init__(self, modules)

    def _call(self, z):
        for flow in self:
            z = flow(z)
        return z

    def append(self, synth):
        nn.ModuleList.append(self, synth)

    def n_parameters(self):
        return sum(flow.n_parameters() for flow in self)

    def set_parameters(self, params, batch_dim=1):
        i = 0
        for flow in self:
            j = i + flow.n_parameters()
            flow.set_parameters(params[:, i:j], batch_dim)
            i = j
    
class HarmonicSynth(Synth):
    """
    Harmonic synthesizer with filtered noise and reverb as described in the
    original DDSP paper.
    """
    
    def __init__(self, args):
        super(HarmonicSynth, self).__init__()
        self.apply(self.init_parameters)
        self.harmonic = HarmonicOscillators(args.n_partial, args.sr, args.block_size)
        self.noise = FilteredNoise(args.filter_size, args.block_size)
        self.reverb = Reverb(args)
    
    def init_parameters(self, m):
        pass

    def forward(self, x):
        x, (f0, loud) = x
        amp, alpha, filter_coeff, h, reverb = x
        x_harmonic = self.harmonic((amp, alpha, f0))
        print(filter_coeff.shape)
        x_noise = self.noise((x_harmonic, filter_coeff))
        x_full = x_harmonic + x_noise
        x = self.reverb(x_full)
        return x

def construct_synth(args):
    """ Construct normalizing flow """
    if (args.synth_type == 'basic'):
        synth = HarmonicSynth(args)
    else:
        synth = SynthList(args)
    return synth