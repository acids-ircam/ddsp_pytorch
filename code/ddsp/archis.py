# -*- coding: utf-8 -*-

from ddsp.effects import Reverb
from ddsp.oscillators import HarmonicOscillators
from ddsp.generators import FilteredNoise
from ddsp.synth import Synth, SynthModule, Add


"""
#######################

Definition of specific synths

#######################
"""

class HarmonicSynth(Synth):
    """
    Harmonic synthesizer with filtered noise and reverb as described in the
    original DDSP paper.
    """
    
    def __init__(self, args):
        super(HarmonicSynth, self).__init__()
        # Harmonic series source
        harmonic = HarmonicOscillators(args.n_partial, args.sr, args.block_size)
        # Filtered noise source
        noise = FilteredNoise(args.filter_size, args.block_size)
        # Go through reverb
        reverb = Reverb(args)
        # Add all modules in order
        add = Add([harmonic, noise])
        self.append(add)
        self.append(reverb)
        # Init parameters
        self.apply(self.init_parameters)

"""
#######################

Helper functions to construct synthesizers

#######################
"""
def construct_synth(args):
    """ Construct synthesizer """
    if (args.synth_type == 'basic'):
        synth = HarmonicSynth(args)
    return synth