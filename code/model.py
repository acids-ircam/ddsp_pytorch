# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import models

"""
###################

High-level models classes

###################
"""

class DDSSynth(nn.Module, models.AE):
    """
    Definition of a generic Differentiable Digital Signal Synthesizer globally
    seen as an auto-encoding architecture. 
    
    The implementation proposed here is somewhat more abstract than 
    the architecture defined in the original paper as it
        - Allows to implement various types of AEs (VAE, WAE, AAE) through duck typing
            (cf. models/ and models/vae/ toolbox)
        - Provides generic handling of conditionning signals
        - Allows to impose orthogonality and diagonalization constraints on latents
        - Maps generically to any modular type of synthesizer
        
    At test time, this module can be seen as a parametric synthesizer whose 
    parameters are defined as
        - A set of conditioning signals
        - Parameters of the synth controlled by a given decoder.
    """
    def __init__(self, encoder, decoder, synth, args, upsampler=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.synth = synth
        if (upsampler is None):
            self.upsampler = nn.Upsample(scale_factor=args.block_size, mode="linear")
        self.init_parameters()
        
    def init_parameters(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def decode(self, z, c):
        """
        Decoding function of the DDSSynth.
        This is the only function that differs from the classic *AE architectures
        as we use a modular synthesizer after the decoding operation
        """
        # First run through decoder
        y = self.decoder(torch.cat(z, c))
        # Then go through the synthesizer modules
        x = self.synth(y)
        return x

class DDSSynthVAE(nn.Module, models.VAE):
    """
    Definition of the variational version of the DDSSynthesizer
    seen as a variational auto-encoding architecture. 
    """

    def __init__(self, encoder, decoder, synth, args, upsampler=None):
        super().__init__(encoder, decoder, synth, args, upsampler)

class DDSSynthVAEFlow(nn.Module, models.VAEFlow):
    """
    Definition of the variational version of the DDSSynthesizer
    seen as a variational auto-encoding architecture. 
    """

    def __init__(self, encoder, decoder, synth, args, upsampler=None):
        super().__init__(encoder, decoder, synth, args, upsampler)

class DDSSynthWAE(nn.Module, models.WAE):
    """
    Definition of the variational version of the DDSSynthesizer
    seen as a variational auto-encoding architecture. 
    """

    def __init__(self, encoder, decoder, synth, args, upsampler=None):
        super().__init__(encoder, decoder, synth, args, upsampler)
        
"""
###################

Original paper architecture classes (for reference)

###################
"""

def mod_sigmoid(x):
    """
    Implementation of the modified sigmoid described in the original article.

    Arguments :
        x (Tensor)      : input tensor, of any shape
    Returns:
        Tensor          : output tensor, same shape of x
    """
    return 2*torch.sigmoid(x)**np.log(10) + 1e-7

class MLP(nn.Module):
    """
    Implementation of a Multi Layer Perceptron, as described in the
    original article (see README)

    Parameters :
        in_size (int)   : input size of the MLP
        out_size (int)  : output size of the MLP
        loop (int)      : number of repetition of Linear-Norm-ReLU
    """
    def __init__(self, in_size=512, out_size=512, loop=3):
        super().__init__()
        self.linear = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.modules.normalization.LayerNorm(out_size),
                nn.ReLU()
            )] + [nn.Sequential(
                nn.Linear(out_size, out_size),
                nn.modules.normalization.LayerNorm(out_size),
                nn.ReLU()
            ) for i in range(loop - 1)])

    def forward(self, x):
        for lin in self.linear:
            x = lin(x)
        return x

class Encoder(nn.Module):
    """
    Raw waveform encoder, based on VQVAE
    """
    def __init__(self, args):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(1,
                        ddsp.conv_hidden_size,
                        ddsp.conv_kernel_size,
                        padding=ddsp.conv_kernel_size//2,
                        stride=ddsp.strides[0])]+\
            [
                nn.Conv1d(ddsp.conv_hidden_size,
                          ddsp.conv_hidden_size,
                          ddsp.conv_kernel_size,
                          padding=ddsp.conv_kernel_size//2,
                          stride=ddsp.strides[i]) for i in range(1, len(ddsp.strides)-1)
            ]+\
            [nn.Conv1d(ddsp.conv_hidden_size,
                       2 * ddsp.conv_out_size,
                       ddsp.conv_kernel_size,
                       padding=ddsp.conv_kernel_size//2,
                       stride=ddsp.strides[-1])]
        )


    def forward(self, x):
        for i,conv in enumerate(self.convs):
            x = conv(x)
            if i != len(self.convs)-1:
                x = torch.relu(x)
        z_mean, z_var = torch.split(x.transpose(1,2), ddsp.conv_out_size,-1)
        return z_mean.contiguous(), z_var.contiguous()


class Decoder(nn.Module):
    """
    Decoder with the architecture originally described in the DDSP paper

    Parameters:
        hidden_size (int)       : Size of vectors inside every MLP + GRU + Dense
        n_partial (int
        Number of partial involved in the harmonic generation. (>1)
    filter_size: int
        Size of the filter used to shape noise.
    """
    def __init__(self, hidden_size, n_partial, filter_size):
        super().__init__()
        # Map the different conditions
        self.f0_MLP = MLP(1,hidden_size)
        self.lo_MLP = MLP(1,hidden_size)
        # Map the latent vector
        self.z_MLP  = MLP(ddsp.conv_out_size, hidden_size)
        # Recurrent model to handle temporality
        self.gru    = nn.GRU(3 * hidden_size, hidden_size, batch_first=True)
        # Mixing MLP after the GRU
        self.fi_MLP = MLP(hidden_size, hidden_size)
        # Outputs to different parameters of the synth
        self.dense_amp    = nn.Linear(hidden_size, 1)
        self.dense_alpha  = nn.Linear(hidden_size, n_partial)
        self.dense_filter = nn.Linear(hidden_size, filter_size // 2 + 1)
        self.dense_reverb = nn.Linear(hidden_size, 2)
        self.n_partial = n_partial

    def forward(self, z, f0, lo, hx=None):
        # 
        f0 = self.f0_MLP(f0)
        lo = self.lo_MLP(lo)
        z  = self.z_MLP(z)
        # Recurrent model
        x,h = self.gru(torch.cat([z, f0, lo], -1), hx)
        # Mixing parameters
        x = self.fi_MLP(x)
        # Retrieve various parameters
        amp          = mod_sigmoid(self.dense_amp(x))
        alpha        = mod_sigmoid(self.dense_alpha(x))
        filter_coeff = mod_sigmoid(self.dense_filter(x))
        reverb       = self.dense_reverb(x)

        alpha        = alpha / torch.sum(alpha,-1).unsqueeze(-1)

        return amp, alpha, filter_coeff, h, reverb
