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
        for modules in self:
            z = modules(z)
        return z

    def append(self, synth):
        """ Append a new module to the synth """
        nn.ModuleList.append(self, synth)
        self.n_params.append(synth.n_parameters())

    def amortization(self, z):
        """ Handles the amortization of the modules """
        if self.amortized in ['none', 'ext']:
            return
        elif self.amortized == 'self':
            if self.amortized_seed.device != z.device:
                self.amortized_seed = self.amortized_seed.to(z.device)
            params = self.amortized_params(self.amortized_seed)
        elif self.amortized == 'input':
            params = self.amortized_params(z)
        self.set_parameters(params, z.shape[0])
        
    def set_parameters(self, params, batch_dim=64):
        """ Set the flows parameters """
        param_list = params.split(self.n_params, dim=1)
        for params, bijector in zip(param_list, self.bijectors):
            bijector.set_parameters(params, batch_dim)

    def n_parameters(self):
        """ Total number of parameters for all flows """
        return sum(self.n_params)
    
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