# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class GatedDense(nn.Module):
    def __init__(self, input_size, output_size, activation=torch.relu):
        super(GatedDense, self).__init__()
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Linear(input_size, output_size)
        self.g = nn.Linear(input_size, output_size)

    def forward(self, x):
        h = self.h(x)
        if self.activation is not None:
            h = self.activation( self.h( x ) )
        g = self.sigmoid( self.g( x ) )
        return h * g

class GatedConv2d(nn.Module):
    
    def __init__(self, in_c, out_c, kernel, stride, pad, dilation=1, act=torch.relu):
        super(GatedConv2d, self).__init__()
        self.activation = act
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Conv2d(in_c, out_c, kernel, stride, pad, dilation)
        self.g = nn.Conv2d(in_c, out_c, kernel, stride, pad, dilation)

    def forward(self, x):
        h = self.activation(self.h(x))
        g = self.sigmoid(self.g(x))
        return h * g

class ResConv2d(nn.Module):
    
    def __init__(self, in_c, out_c, kernel, stride, pad, dilation=1, act=torch.relu):
        super(ResConv2d, self).__init__()
        self.activation = act
        self.h = nn.Conv2d(in_c, out_c, kernel, stride, pad, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_c)
        self.g = nn.Conv2d(in_c, out_c, kernel, stride, pad, dilation)
        for i in range(self.g.weight.shape[0]):
            for j in range(self.g.weight.shape[1]):
                nn.init.eye_(self.g.weight.data[i, j])
        nn.init.constant_(self.g.bias.data, 0)
        self.g.weight.requires_grad = False
        self.g.bias.requires_grad = False

    def forward(self, x):
        h = self.activation(self.bn(self.h(x)))
        g = self.g(x)
        return h + g

class ResConvTranspose2d(nn.Module):
    
    def __init__(self, in_c, out_c, kernel, stride, pad, output_padding=0, dilation=1, act=torch.relu):
        super(ResConvTranspose2d, self).__init__()
        self.activation = act
        self.h = nn.ConvTranspose2d(in_c, out_c, kernel, stride, pad, output_padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_c)
        self.g = nn.ConvTranspose2d(in_c, out_c, kernel, stride, pad, output_padding, dilation=dilation)
        for i in range(self.g.weight.shape[0]):
            for j in range(self.g.weight.shape[1]):
                nn.init.eye_(self.g.weight.data[i, j])
        nn.init.constant_(self.g.bias.data, 0)
        self.g.weight.requires_grad = False
        self.g.bias.requires_grad = False

    def forward(self, x):
        h = self.activation(self.bn(self.h(x)))
        g = self.g(x)
        return h + g
    
class GatedConvTranspose2d(nn.Module):
    
    def __init__(self, in_c, out_c, kernel, stride, pad, output_padding=0, dilation=1, act=torch.relu):
        super(GatedConvTranspose2d, self).__init__()
        self.activation = act
        self.sigmoid = nn.Sigmoid()
        self.h = nn.ConvTranspose2d(in_c, out_c, kernel, stride, pad, output_padding, dilation=dilation)
        self.g = nn.ConvTranspose2d(in_c, out_c, kernel, stride, pad, output_padding, dilation=dilation)

    def forward(self, x):
        h = self.activation(self.h(x))
        g = self.sigmoid(self.g(x))
        return h * g