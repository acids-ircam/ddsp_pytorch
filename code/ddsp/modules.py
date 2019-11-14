# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
#######################

Smoothed windowed convolution functions

#######################
"""

def get_window(name, window_length, squared=False):
    """
    Returns a windowing function.
    
    Arguments:
    ----------
        window (str)                : name of the window, currently only 'hann' is available
        window_length (int)         : length of the window
        squared (bool)              : if true, square the window
        
    Returns:
    ----------
        torch.FloatTensor           : window of size `window_length`
    """
    if name == "hann":
        window = torch.hann_window(window_length)
    elif name == "hamming":
        window = torch.hamming_window(window_length)
    elif name == "blackman":
        window = torch.blackman_window(window_length)
    else:
        raise ValueError("Invalid window name {}".format(name))
    if squared:
        window *= window
    return window

class WindowedConv1d(nn.Module):
    """
    Smooth a convolution using a window.
    
    Arguments:
        conv (nn.Conv1d)            : convolution module to wrap
        window_name (str or None)   : name of the window used to smooth the convolutions
        squared (bool)              : if `True`, square the smoothing window
    """

    def __init__(self, conv, window_name='hann', squared=True):
        super(WindowedConv1d, self).__init__()
        self.window_name = window_name
        if squared:
            self.window_name += "**2"
        self.register_buffer('window',get_window(window_name, conv.weight.size(-1), squared=squared))
        self.conv = conv

    def forward(self, input):
        weight = self.window * self.conv.weight
        return F.conv1d(
            input,
            weight,
            bias=self.conv.bias,
            stride=self.conv.stride,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
            padding=self.conv.padding)

    def __repr__(self):
        return "WindowedConv1d(window={},conv={})".format(
            self.window_name, self.conv)


class WindowedConvTranspose1d(nn.Module):
    """
    Smooth a transposed convolution using a window.
    
    Arguments:
            conv (nn.Conv1d)        : convolution module to wrap
            window_name (str)       : name of the window used to smooth the convolutions
            squared (bool)          : if `True`, square the smoothing window
    """

    def __init__(self, conv_tr, window_name='hann', squared=True):
        super(WindowedConvTranspose1d, self).__init__()
        self.window_name = window_name
        if squared:
            self.window_name += "**2"
        self.register_buffer('window', get_window(window_name, conv_tr.weight.size(-1), squared=squared))
        self.conv_tr = conv_tr

    def forward(self, input):
        weight = self.window * self.conv_tr.weight
        return F.conv_transpose1d(input,
            weight,
            bias=self.conv_tr.bias,
            stride=self.conv_tr.stride,
            padding=self.conv_tr.padding,
            output_padding=self.conv_tr.output_padding,
            groups=self.conv_tr.groups,
            dilation=self.conv_tr.dilation)

    def __repr__(self):
        return "WindowedConvTranpose1d(window={},conv_tr={})".format(
            self.window_name, self.conv_tr)


class ResConv1d(nn.Conv1d):
    
    def __init__(self, *args, **kwargs):
        super(nn.Conv1d, self).__init__(*args, **kwargs)
        # Add the initialization of parameters
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

    def forward(self, x, c=None):
        # Process convolution
        c_out = super.forward(x)
        out = x + c_out
        # Add condition
        if (c is not None):
            out = out + c
        return out

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

    def forward(self, x):
        h = self.activation(self.bn(self.h(x)))
        g = self.g(x)
        return h + g