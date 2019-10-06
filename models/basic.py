# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal
# Model layers imports
from models.layers import GatedDense, GatedConv2d, ResConv2d
from models.layers import GatedConvTranspose2d, ResConvTranspose2d 
# Flows library imports
from models.flows.coupling import MaskedCouplingFlow
from models.flows.flow import NormalizingFlow
from models.flows.iaf import IAFlow, ContextIAFlow, DDSF_IAFlow
from models.flows.maf import MAFlow, ContextMAFlow
from models.flows.naf import DeepSigmoidFlow, DeepDenseSigmoidFlow
from models.flows.normalization import BatchNormFlow
from models.flows.order import ReverseFlow, ShuffleFlow
from models.flows.planar import PlanarFlow
from models.flows.sylvester import TriangularSylvesterFlow

def construct_encoder_decoder(in_size, enc_size, latent_size, hidden_size = 512, channels = 32, n_layers = 6, n_mlp = 2, type_ae = 'ae', type_mod='gated_cnn', args=None):
    """ Construct encoder and decoder layers for AE models """
    # MLP layers
    if (type_mod in ['mlp', 'gated_mlp']):
        type_ed = (type_mod == 'mlp') and 'normal' or 'gated'
        encoder = GatedMLP(np.prod(in_size), enc_size, hidden_size, n_layers, type_ed)
        decoder = DecodeMLP(latent_size, in_size, hidden_size, n_layers, type_ed)
    elif (type_mod in ['cnn', 'gated_cnn', 'res_cnn']):
        type_ed = (type_mod == 'cnn') and 'normal' or ((type_mod == 'res_cnn') and 'residual' or 'gated')
        encoder = GatedCNN(in_size, enc_size, channels, n_layers, hidden_size, n_mlp, type_ed, args)
        decoder = DecodeCNN(latent_size, encoder.cnn_size, in_size, channels, n_layers, hidden_size, n_mlp, type_ed, args)
    return encoder, decoder

def construct_flow(flow_dim, flow_type='maf', flow_length=16, amortization='input'):
    """ Construct normalizing flow """
    if flow_type == 'planar':
        blocks = [ PlanarFlow ]
    elif flow_type == 'sylvester':
        blocks = [ TriangularSylvesterFlow, BatchNormFlow, ShuffleFlow ]
    elif flow_type == 'real_nvp':
        blocks = [ MaskedCouplingFlow, BatchNormFlow, ShuffleFlow ]
    elif flow_type == 'maf':
        blocks = [ MAFlow, BatchNormFlow, ReverseFlow ]
    elif flow_type == 'iaf':
        blocks = [ IAFlow, BatchNormFlow, ShuffleFlow ]
    elif flow_type == 'dsf':
        blocks = [ DeepSigmoidFlow, BatchNormFlow, ReverseFlow ]
    elif flow_type == 'ddsf':
        blocks = [ DeepDenseSigmoidFlow, BatchNormFlow, ReverseFlow ]
    elif flow_type == 'ddsf_iaf':
        blocks = [ DDSF_IAFlow, BatchNormFlow, ShuffleFlow ]
    elif flow_type == 'iaf_ctx':
        blocks = [ ContextIAFlow, BatchNormFlow, ShuffleFlow ]
    elif flow_type == 'maf_ctx':
        blocks = [ ContextMAFlow, BatchNormFlow, ReverseFlow ]
    else:
        raise ValueError('Invalid flow choice : ' + flow_type)
    flow = NormalizingFlow(
        dim=flow_dim, blocks=blocks, flow_length=flow_length,
        density=MultivariateNormal(torch.zeros(flow_dim),
        torch.eye(flow_dim)), amortized='self'
    )
    return flow, blocks

class RegressionModel(nn.Module):
    
    def __init__(self, **kwargs):
        super(RegressionModel, self).__init__(**kwargs)
    
    def train_epoch(self, loader, loss, optimizer, args):
        self.train()
        full_loss = 0
        for (x, y, _, _) in loader:
            # Send to device
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            out = self(x)
            b_loss = loss(out, y)
            b_loss.backward()
            optimizer.step()
            full_loss += b_loss
        full_loss /= len(loader)
        return full_loss
    
    def eval_epoch(self, loader, loss, args):
        self.eval()
        full_loss = 0
        with torch.no_grad():
            for (x, y, _, _) in loader:
                x, y = x.to(args.device), y.to(args.device)
                out = self(x).data
                full_loss += loss(out, y)
            full_loss /= len(loader)
        return full_loss

class GatedMLP(RegressionModel):
    
    def __init__(self, in_size, out_size, hidden_size = 512, n_layers = 6, type_mod='gated', **kwargs):
        super(GatedMLP, self).__init__(**kwargs)
        dense_module = (type_mod == 'gated') and GatedDense or nn.Linear
        # Create modules
        modules = nn.Sequential()
        for l in range(n_layers):
            in_s = (l==0) and in_size or hidden_size
            out_s = (l == n_layers - 1) and out_size or hidden_size
            modules.add_module('l%i'%l, dense_module(in_s, out_s))
            if (l < n_layers - 1):
                modules.add_module('b%i'%l, nn.BatchNorm1d(out_s))
                modules.add_module('a%i'%l, nn.ReLU())
                modules.add_module('a%i'%l, nn.Dropout(p=.3))
        self.net = modules
    
    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)
        
    def forward(self, inputs):
        # Flatten the input
        out = inputs.view(inputs.shape[0], -1)
        for m in range(len(self.net)):
            out = self.net[m](out)
        return out
    
class DecodeMLP(GatedMLP):
    
    def __init__(self, in_size, out_size, hidden_size = 512, n_layers = 6, type_mod='gated', **kwargs):
        super(DecodeMLP, self).__init__(in_size, np.prod(out_size), hidden_size, n_layers, type_mod, **kwargs)
        # Record final size
        self.out_size = out_size
        
    def forward(self, inputs):
        # Use super function
        out = GatedMLP.forward(self, inputs)
        # Reshape output
        out = out.view(inputs.shape[0], *self.out_size)
        return out
    
class GatedCNN(RegressionModel):
    
    def __init__(self, in_size, out_size, channels = 32, n_layers = 5, hidden_size = 512, n_mlp = 2, type_mod='gated', args=None):
        super(GatedCNN, self).__init__()
        conv_module = (type_mod == 'gated') and GatedConv2d or nn.Conv2d
        conv_module = (type_mod == 'residual') and ResConv2d or conv_module
        dense_module = (type_mod == 'gated') and GatedDense or nn.Linear
        # Create modules
        modules = nn.Sequential()
        size = [in_size[-2], in_size[-1]]
        in_channel = 1 if len(in_size)<3 else in_size[0] #in_size is (C,H,W) or (H,W)
        kernel = args.kernel
        stride = 2
        """ First do a CNN """
        for l in range(n_layers):
            dil = ((args.dilation == 3) and (2 ** l) or args.dilation)
            pad = 3 * (dil + 1)
            in_s = (l==0) and in_channel or channels
            out_s = (l == n_layers - 1) and 1 or channels
            modules.add_module('c2%i'%l, conv_module(in_s, out_s, kernel, stride, pad, dilation = dil))
            if (l < n_layers - 1):
                modules.add_module('b2%i'%l, nn.BatchNorm2d(out_s))
                modules.add_module('a2%i'%l, nn.ReLU())
                modules.add_module('d2%i'%l, nn.Dropout2d(p=.25))
            size[0] = int((size[0]+2*pad-(dil*(kernel-1)+1))/stride+1)
            size[1] = int((size[1]+2*pad-(dil*(kernel-1)+1))/stride+1)
        self.net = modules
        self.mlp = nn.Sequential()
        """ Then go through MLP """
        for l in range(n_mlp):
            in_s = (l==0) and (size[0] * size[1]) or hidden_size
            out_s = (l == n_mlp - 1) and out_size or hidden_size
            self.mlp.add_module('h%i'%l, dense_module(in_s, out_s))
            if (l < n_layers - 1):
                self.mlp.add_module('b%i'%l, nn.BatchNorm1d(out_s))
                self.mlp.add_module('a%i'%l, nn.ReLU())
                self.mlp.add_module('d%i'%l, nn.Dropout(p=.25))
        self.cnn_size = size
    
    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)
        
    def forward(self, inputs):
        out = inputs.unsqueeze(1) if len(inputs.shape) < 4 else inputs # force to (batch, C, H, W)
        for m in range(len(self.net)):
            out = self.net[m](out)
        out = out.view(inputs.shape[0], -1)
        for m in range(len(self.mlp)):
            out = self.mlp[m](out)
        return out
    
class DecodeCNN(RegressionModel):
    
    def __init__(self, in_size, cnn_size, out_size, channels = 32, n_layers = 5, hidden_size = 512, n_mlp = 2, type_mod='gated', args=None):
        super(DecodeCNN, self).__init__()
        conv_module = (type_mod == 'gated') and GatedConvTranspose2d or nn.ConvTranspose2d
        conv_module = (type_mod == 'residual') and ResConvTranspose2d or conv_module
        dense_module = (type_mod == 'gated') and GatedDense or nn.Linear
        # Create modules
        self.cnn_size = [cnn_size[0], cnn_size[1]]
        size = cnn_size
        kernel = args.kernel
        stride = 2
        self.mlp = nn.Sequential()
        """ First go through MLP """
        for l in range(n_mlp):
            in_s = (l==0) and (in_size) or hidden_size
            out_s = (l == n_mlp - 1) and np.prod(cnn_size) or hidden_size
            self.mlp.add_module('h%i'%l, dense_module(in_s, out_s))
            if (l < n_layers - 1):
                self.mlp.add_module('b%i'%l, nn.BatchNorm1d(out_s))
                self.mlp.add_module('a%i'%l, nn.ReLU())
                self.mlp.add_module('d%i'%l, nn.Dropout(p=.25))
        modules = nn.Sequential()
        """ Then do a CNN """
        for l in range(n_layers):
            dil = ((args.dilation == 3) and (2 ** ((n_layers - 1) - l)) or args.dilation)
            pad = 3 * (dil + 1)
            if (args.dilation == 1):
                pad = 2
            out_pad = (pad % 2)
            in_s = (l==0) and 1 or channels
            out_s = (l == n_layers - 1) and out_size[0] or channels
            modules.add_module('c2%i'%l, conv_module(in_s, out_s, kernel, stride, pad, output_padding=out_pad, dilation = dil))
            if (l < n_layers - 1):
                modules.add_module('b2%i'%l, nn.BatchNorm2d(out_s))
                modules.add_module('a2%i'%l, nn.ReLU())
                modules.add_module('a2%i'%l, nn.Dropout2d(p=.25))
            size[0] = int((size[0] - 1) * stride - (2 * pad) + dil * (kernel - 1) + out_pad + 1)
            size[1] = int((size[1] - 1) * stride - (2 * pad) + dil * (kernel - 1) + out_pad + 1)
        self.net = modules
        self.out_size = out_size #(H,W) or (C,H,W)
    
    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)
        
    def forward(self, inputs):
        out = inputs
        for m in range(len(self.mlp)):
            out = self.mlp[m](out)
        out = out.unsqueeze(1).view(-1, 1, self.cnn_size[0], self.cnn_size[1])
        for m in range(len(self.net)):
            out = self.net[m](out)
        if len(self.out_size) < 3:
            out = out[:, :, :self.out_size[0], :self.out_size[1]].squeeze(1)
        else:
            out = out[:, :, :self.out_size[1], :self.out_size[2]]
        return out