# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from models.vae.ae import AE
from models.vae.vae import VAE
from models.vae.vae_flow import VAEFlow
from models.vae.wae import WAE

"""
###################

High-level models classes

###################
"""

class DDSSynth(AE):
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
        super(DDSSynth, self).__init__(encoder, decoder, args.encoder_dims, args.encoder_dims)
        self.synth = synth
        if (upsampler is None):
            self.upsampler = nn.Upsample(scale_factor=args.block_size, mode="linear", align_corners=False)
    
    def decode(self, z):
        """
        Decoding function of the DDSSynth.
        This is the only function that differs from the classic *AE architectures
        as we use a modular synthesizer after the decoding operation
        """
        # Separate conditions
        _, cond = z
        # First run through decoder
        z = self.decoder(z)
        # Then go through the synthesizer modules
        x = self.synth((z, cond))
        return x

    def forward(self, x):
        if (type(x) == tuple):
            x, condition = x
        # Encode the inputs
        z = self.encode(x)
        # Potential regularization
        z_tilde, z_loss = self.regularize(z)
        # Decode the samples
        x_tilde = self.decode((z_tilde, condition))
        return x_tilde, z_tilde, z_loss
    
    def train_epoch(self, loader, loss, optimizer, args):
        self.train()
        full_loss = 0
        for (x, f0, loud, y) in loader:
            # Send to device
            x, f0, loud = [it.to(args.device, non_blocking=True) for it in [x, f0, loud]]
            f0, loud = f0.transpose(1, 2), loud.transpose(1, 2)
            # Auto-encode
            x_tilde, z_tilde, z_loss = self((x, (f0, loud)))
            # Reconstruction loss
            rec_loss = loss(x_tilde, y) / float(x.shape[1] * x.shape[2])
            # Final loss
            b_loss = (rec_loss + (args.beta * z_loss)).mean(dim=0)
            # Perform backward
            optimizer.zero_grad()
            b_loss.backward()
            optimizer.step()
            full_loss += b_loss
        full_loss /= len(loader)
        return full_loss
    
    def eval_epoch(self, loader, loss, args):
        self.eval()
        full_loss = 0
        with torch.no_grad():
            for (x, f0, loud, y) in loader:
                # Send to device
                x, f0, loud = [it.to(args.device, non_blocking=True) for it in [x, f0, loud]]
                f0, loud = f0.transpose(1, 2), loud.transpose(1, 2)
                # Auto-encode
                x_tilde, z_tilde, z_loss = self((x, (f0, loud)))
                # Final loss
                rec_loss = loss(x_tilde, y)
                full_loss += rec_loss
            full_loss /= len(loader)
        return full_loss

class DDSSynthVAE(DDSSynth, VAE):
    """
    Definition of the variational version of the DDSSynthesizer
    seen as a variational auto-encoding architecture. 
    """

    def __init__(self, encoder, decoder, synth, args, upsampler=None):
        DDSSynth.__init__(self, encoder, decoder, synth, args, upsampler)
        VAE.__init__(self)

class DDSSynthVAEFlow(DDSSynth, VAEFlow):
    """
    Definition of the variational + normalizing flow version of the DDSSynthesizer
    """

    def __init__(self, encoder, decoder, synth, args, upsampler=None):
        DDSSynth.__init__(self, encoder, decoder, synth, args, upsampler)
        VAEFlow.__init__(self)

class DDSSynthWAE(DDSSynth, WAE):
    """
    Definition of the Wasserstein version of the DDSSynthesizer
    """

    def __init__(self, encoder, decoder, synth, args, upsampler=None):
        DDSSynth.__init__(self, encoder, decoder, synth, args, upsampler)
        WAE.__init__(self)
        
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
    Implementation of the MLP, as described in the original paper

    Parameters :
        in_size (int)   : input size of the MLP
        out_size (int)  : output size of the MLP
        loop (int)      : number of repetition of Linear-Norm-ReLU
    """
    def __init__(self, in_size=512, out_size=512, loop=3):
        super().__init__()
        self.linear = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_size, out_size),
                nn.modules.normalization.LayerNorm(out_size),
                nn.ReLU()
            )] + [nn.Sequential(nn.Linear(out_size, out_size),
                nn.modules.normalization.LayerNorm(out_size),
                nn.ReLU()
            ) for i in range(loop - 1)])

    def forward(self, x):
        for lin in self.linear:
            x = lin(x)
        return x

class EncoderWave(nn.Module):
    """
    Raw waveform encoder, based on VQVAE
    """
    def __init__(self, args):
        super().__init__()
        self.out_size = args.encoder_dims
        self.convs = nn.ModuleList(
            [nn.Conv1d(1, args.channels, args.kernel_size,
                        padding=args.kernel_size // 2,
                        stride=args.strides[0])]
            + [nn.Conv1d(args.channels, args.channels, args.kernel_size,
                         padding=args.kernel_size // 2,
                         stride=args.strides[i]) for i in range(1, len(args.strides) - 1)]
            + [nn.Conv1d(args.channels, args.encoder_dims, args.kernel_size,
                         padding=args.kernel_size // 2,
                         stride=args.strides[-1])])

    def forward(self, x):
        for i,conv in enumerate(self.convs):
            x = conv(x)
            if i != len(self.convs)-1:
                x = torch.relu(x)
        return x


class Decoder(nn.Module):
    """
    Decoder with the architecture originally described in the DDSP paper

    Parameters:
        hidden_size (int)       : Size of vectors inside every MLP + GRU + Dense
        n_partial (int)         : Number of partial involved in the harmonic generation. (>1)
        filter_size (int)       : Size of the filter used to shape noise.
    """
    def __init__(self, args):
        super().__init__()
        # Map the different conditions
        self.f0_MLP = MLP(1,args.n_hidden)
        self.lo_MLP = MLP(1,args.n_hidden)
        # Map the latent vector
        self.z_MLP  = MLP(args.latent_dims, args.n_hidden)
        # Recurrent model to handle temporality
        self.gru    = nn.GRU(3 * args.n_hidden, args.n_hidden, batch_first=True)
        # Mixing MLP after the GRU
        self.fi_MLP = MLP(args.n_hidden, args.n_hidden)
        # Outputs to different parameters of the synth
        self.dense_amp    = nn.Linear(args.n_hidden, 1)
        self.dense_alpha  = nn.Linear(args.n_hidden, args.n_partial)
        self.dense_filter = nn.Linear(args.n_hidden, args.filter_size // 2 + 1)
        self.dense_reverb = nn.Linear(args.n_hidden, 2)
        self.n_partial = args.n_partial

    def forward(self, z, hx=None):
        if (type(z) == tuple):
            z, condition = z
            f0, lo = condition
        # Forward pass for the encoding
        z = z.transpose(1, 2)
        f0 = self.f0_MLP(f0)
        lo = self.lo_MLP(lo)
        z  = self.z_MLP(z)
        # Recurrent model
        x, h = self.gru(torch.cat([z, f0, lo], -1), hx)
        # Mixing parameters
        x = self.fi_MLP(x)
        # Retrieve various parameters
        amp          = mod_sigmoid(self.dense_amp(x))
        alpha        = mod_sigmoid(self.dense_alpha(x))
        filter_coeff = mod_sigmoid(self.dense_filter(x))
        reverb       = self.dense_reverb(x)
        # Compute the final alpha
        alpha        = alpha / torch.sum(alpha,-1).unsqueeze(-1)
        # Return the set of parameters
        return torch.cat([amp, alpha, filter_coeff, reverb], dim=2)

"""
###################

Helper functions to construct the encoders and decoders

###################
"""

def construct_architecture(args):
    """ Construct encoder and decoder layers for AE models """
    # For starters, construct basic encoder and decoder
    encoder = EncoderWave(args)
    decoder = Decoder(args)
    return encoder, decoder
