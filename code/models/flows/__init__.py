# -*-coding:utf-8 -*-
 
"""
    The ``flow`` module
    ========================
 
    This package contains all normalizing and generative flow classes
 
    :Example:
 
    >>> 
 
    Subpackages available
    ---------------------

        * Generic
        * Audio
        * Midi
        * References
        * Time Series
        * Pytorch
        * Tensorflow
 
    Comments and issues
    ------------------------
        
        None for the moment
 
    Contributors
    ------------------------
        
    * Philippe Esling       (esling@ircam.fr)
 
"""

from .activation import PReLUFlow
from .coupling import AffineCouplingFlow, MaskedCouplingFlow, ConvolutionalCouplingFlow
from .flow import Flow, NormalizingFlow, NormalizingFlowContext, GenerativeFlow, FlowList
from .iaf import IAFlow, ContextIAFlow, DDSF_IAFlow
from .maf import MAFlow, ContextMAFlow
from .naf import DeepSigmoidFlow, DeepDenseSigmoidFlow
from .normalization import BatchNormFlow, ActNormFlow
from .order import ReverseFlow, ShuffleFlow, SplitFlow, SqueezeFlow
 
# info
__version__ = "1.0"
__author__  = "esling@ircam.fr, chemla@ircam.fr"
__date__    = ""
__all__     = [
    'PReLUFlow', 
    'AffineCouplingFlow', 'MaskedCouplingFlow', 'ConvolutionalCouplingFlow',
    'Flow', 'NormalizingFlow', 'NormalizingFlowContext', 'GenerativeFlow', 'FlowList',
    'IAFlow', 'ContextIAFlow', 'DDSF_IAFlow',
    'MAFlow', 'ContextMAFlow',
    'DeepSigmoidFlow', 'DeepDenseSigmoidFlow',
    'BatchNormFlow', 'ActNormFlow',
    'ReverseFlow', 'ShuffleFlow', 'SplitFlow', 'SqueezeFlow'
]

# import sub modules
#from . import ar
#from . import basic
#from . import flow
#from . import generative
#from . import layers
#from . import order
#from . import temporal
