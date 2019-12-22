# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

"""
#######################

High-level synthesizer modules classes

#######################
"""

class SynthModule(nn.Module):
    """
    Generic class defining a synthesis module.
    """
    
    def __init__(self, amortized='input'):
        """ Initialize as module """
        super(SynthModule, self).__init__()
        # Handle amortization
        self.amortized = amortized

    def init_parameters(self, m=None):
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
    Generic class for defining a synthesizer (seen as a list of synthesis modules).
    """

    def __init__(self, modules=[], param_network=None):
        SynthModule.__init__(self, modules[0].amortized if modules else 'input')
        nn.ModuleList.__init__(self, modules)
        self.n_params = []
        # Params are given right away
        for m in modules:
            self.n_params.append(m.n_parameters())
        #if (param_network is None):
        #    self.amortized_params = SynthIdentity()

    def forward(self, z):
        """ Call the synthesis flow """
        # Split parameters and conditions
        z, conditions = z
        # Apply the parameters
        self.amortization(z)
        # Go through synth
        for modules in self._modules.values():
            z = modules((z, conditions))
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
            params = z
        self.set_parameters(params, z.shape[0])
        
    def set_parameters(self, params, batch_dim=64):
        """ Set the flows parameters """
        param_list = params.split(self.n_params, dim=2)
        for params, module in zip(param_list, self._modules.values()):
            module.set_parameters(params, batch_dim)

    def n_parameters(self):
        """ Total number of parameters for all flows """
        return sum(self.n_params)


"""
#######################

Definition of simple synth operators

#######################
"""

class SynthIdentity(nn.Identity, SynthModule):
    
    def __init__(self):
        SynthModule.__init__(self)
        nn.Identity.__init__(self)
    
class Add(Synth):
    
    def __init__(self, modules=[]):
        super(Add, self).__init__(modules)

    def forward(self, z):
        z, conditions = z
        z_f = None
        for modules in self._modules.values():
            if (z_f is None):
                z_f = modules((z, conditions))
                continue
            z_f += modules((z, conditions))
        return z_f

    def n_parameters(self):
        """ Total number of parameters for all flows """
        return sum([mod.n_parameters() for mod in self._modules.values()])

class Mul(Synth):
    
    def __init__(self, modules=[]):
        super(Mul, self).__init__(modules)

    def forward(self, z):
        z, _ = z
        z_f = torch.ones_as(z)
        for modules in self._modules.values():
            z_f *= modules(z)
        return z_f

    def n_parameters(self):
        """ Total number of parameters for all flows """
        return sum([mod.n_parameters() for mod in self._modules.values()])

