# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
    
class Analysis(nn.Module):
    
    def __init__(self):
        super(Analysis, self).__init__()
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        pass

    def forward(self, x):
        pass