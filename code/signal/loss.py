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