# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

def mod_sigmoid(x):
    """
    Implementation of the modified sigmoid described in the original article.

    Parameters
    ----------

    x: Tensor
        Input tensor, of any shape

    Returns
    -------

    Tensor:
        Output tensor, shape of x
    """
    return 2*torch.sigmoid(x)**np.log(10) + 1e-7

class MLP(nn.Module):
    """
    Implementation of a Multi Layer Perceptron, as described in the
    original article (see README)

    Parameters
    ----------

    in_size: int
        Input size of the MLP
    out_size: int
        Output size of the MLP
    loop: int
        Number of repetition of Linear-Norm-ReLU
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
            ) for i in range(loop - 1)]
        )

    def forward(self, x):
        for lin in self.linear:
            x = lin(x)
        return x

class Encoder(nn.Module):
    """
    Raw waveform encoder, based on VQVAE
    """
    def __init__(self):
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
    Decoder of the architecture described in the original architecture.

    Parameters
    ----------

    hidden_size: int
        Size of vectors inside every MLP + GRU + Dense
    n_partial: int
        Number of partial involved in the harmonic generation. (>1)
    filter_size: int
        Size of the filter used to shape noise.


    """
    def __init__(self, hidden_size, n_partial, filter_size):
        super().__init__()
        self.f0_MLP = MLP(1,hidden_size)
        self.lo_MLP = MLP(1,hidden_size)
        self.z_MLP  = MLP(ddsp.conv_out_size, hidden_size)

        self.gru    = nn.GRU(3 * hidden_size, hidden_size, batch_first=True)

        self.fi_MLP = MLP(hidden_size, hidden_size)

        self.dense_amp    = nn.Linear(hidden_size, 1)
        self.dense_alpha  = nn.Linear(hidden_size, n_partial)
        self.dense_filter = nn.Linear(hidden_size, filter_size // 2 + 1)
        self.dense_reverb = nn.Linear(hidden_size, 2)

        self.n_partial = n_partial

    def forward(self, z, f0, lo, hx=None):
        f0 = self.f0_MLP(f0)
        lo = self.lo_MLP(lo)
        z  = self.z_MLP(z)




        x,h = self.gru(torch.cat([z, f0, lo], -1), hx)

        x = self.fi_MLP(x)

        amp          = mod_sigmoid(self.dense_amp(x))
        alpha        = mod_sigmoid(self.dense_alpha(x))
        filter_coeff = mod_sigmoid(self.dense_filter(x))
        reverb       = self.dense_reverb(x)

        alpha        = alpha / torch.sum(alpha,-1).unsqueeze(-1)

        return amp, alpha, filter_coeff, h, reverb

class NeuralSynth(nn.Module):
    """
    Implementation of a parametric Harmonic + Noise + IR reverb synthesizer,
    whose parameters are controlled by the previously implemented decoder.
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

        self.decoder = Decoder(ddsp.hidden_size,
                               ddsp.n_partial,
                               ddsp.filter_size)

        self.condition_upsample = nn.Upsample(scale_factor=preprocess.block_size,
                                              mode="linear")

        for n,p in self.named_parameters():
            try:
                nn.init.xavier_normal_(p)
            except:
                # print(f"Skipped initialization of {n}")
                pass


        self.k = nn.Parameter(torch.arange(1, ddsp.n_partial + 1)\
                              .reshape(1,1,-1)\
                              .float(), requires_grad=False)

        self.windows = nn.ParameterList(
        nn.Parameter(torch.from_numpy(
                np.hanning(scale)
            ).float(), requires_grad=False)\
            for scale in preprocess.fft_scales)

        self.filter_window = nn.Parameter(torch.hann_window(ddsp.filter_size)\
                                               .roll(ddsp.filter_size//2,-1),
                                               requires_grad=False)

        self.impulse = Reverb(preprocess.sequence_size * preprocess.block_size)



    def forward(self, x, f0, lo,
                amp_pass=True,
                synth_pass=True,
                noise_pass=False,
                conv_pass=False,
                pre_encoded=False):
        bs = f0.shape[0]
        assert len(f0.shape)==3, f"f0 input must be 3-dimensional, but is {len(f0.shape)}-dimensional."
        assert len(lo.shape)==3, f"lo input must be 3-dimensional, but is {len(lo.shape)}-dimensional."

        # Z ENCODING OF AUDIO ##################################################
        if pre_encoded:
            z = x
        else:
            z_mean, z_var = self.encoder(x)
            z = torch.exp(z_var) * torch.randn_like(z_mean) + z_mean

        # ADDITIVE SYNTH #######################################################
        # GETTING SYNTH PARAMETERS
        amp, alpha, filter_coef, h, reverb = self.decoder(z, f0, lo)

        if not amp_pass:
            amp = torch.ones_like(amp, requires_grad=False)

        reverb = torch.mean(reverb, 1)


        # UPSAMPLING PARAMETERS TO SAMPLERATE
        f0          = self.condition_upsample(f0.transpose(1,2))\
                          .squeeze(1)/preprocess.samplerate
        amp         = self.condition_upsample(amp.transpose(1,2)).squeeze(1)
        alpha       = self.condition_upsample(alpha.transpose(1,2)).transpose(1,2)

        # GENERATING PHASE
        phi = torch.zeros(f0.shape).to(f0.device)
        for i in np.arange(1,phi.shape[-1]):
            phi[:,i] = 2 * np.pi * f0[:,i] + phi[:,i-1]
        phi = phi.unsqueeze(-1).expand(alpha.shape)

        # FILTERING OUT FREQUENCIES ABOVE NYQUIST
        antia_alias = (self.k * f0.unsqueeze(-1) < .5).float()

        y =  amp * torch.sum(antia_alias * alpha * torch.sin( self.k * phi),-1)

        if not synth_pass:
            y = y.detach()

        # NOISE SHAPING ########################################################

        # CREATING SOURCE NOISE
        noise = (torch.rand(y.shape)*2-1)\
                .to(y.device)\
                .reshape(-1, preprocess.block_size)/100
        S_noise = torch.rfft(noise,1).reshape(bs,-1,preprocess.block_size//2+1,2)

        # FORMATTING FILTER_COEF TO COMPLEX FORM
        filter_coef = filter_coef.reshape([-1, ddsp.filter_size//2+1, 1])\
                      .expand([-1, ddsp.filter_size//2+1, 2]).contiguous()
        filter_coef[:,:,1] = 0

        # COMPUTING FILTER WINDOWED IMPULSE RESPONSE
        h = torch.irfft(filter_coef, 1, signal_sizes=(ddsp.filter_size,))
        h_w = self.filter_window.unsqueeze(0) * h
        h_w = nn.functional.pad(h_w, (0,preprocess.block_size-ddsp.filter_size),
                                "constant", 0)

        H = torch.rfft(h_w, 1).reshape(bs, -1, preprocess.block_size//2 + 1, 2)


        S_filtered_noise          = torch.zeros_like(H)
        S_filtered_noise[:,:,:,0] = H[:,:,:,0] * S_noise[:,:,:,0] - H[:,:,:,1] * S_noise[:,:,:,1]
        S_filtered_noise[:,:,:,1] = H[:,:,:,0] * S_noise[:,:,:,1] + H[:,:,:,1] * S_noise[:,:,:,0]

        S_filtered_noise          = S_filtered_noise.reshape(-1, preprocess.block_size//2 + 1, 2)

        filtered_noise = torch.irfft(S_filtered_noise,1)[:,:preprocess.block_size].reshape(bs,-1)



        if noise_pass:
            y += filtered_noise


        # CONVOLUTION WITH AN IMPULSE RESPONSE #################################
        y = nn.functional.pad(y, (0, preprocess.block_size*preprocess.sequence_size),
                              "constant", 0)

        Y_S = torch.rfft(y,1)

        impulse = self.impulse(reverb, conv_pass)

        impulse = nn.functional.pad(impulse,
                                    (0, preprocess.block_size*preprocess.sequence_size),
                                    "constant", 0)

        if y.shape[-1] > preprocess.sequence_size * preprocess.block_size:
            impulse = nn.functional.pad(impulse,
                                        (0, y.shape[-1]-impulse.shape[-1]),
                                        "constant", 0)
        if conv_pass:
            IR_S = torch.rfft(impulse,1).expand_as(Y_S)
        else:
            IR_S = torch.rfft(impulse.detach(),1).expand_as(Y_S)

        Y_S_CONV = torch.zeros_like(IR_S)
        Y_S_CONV[:,:,0] = Y_S[:,:,0] * IR_S[:,:,0] - Y_S[:,:,1] * IR_S[:,:,1]
        Y_S_CONV[:,:,1] = Y_S[:,:,0] * IR_S[:,:,1] + Y_S[:,:,1] * IR_S[:,:,0]

        y = torch.irfft(Y_S_CONV, 1, signal_sizes=(y.shape[-1],))

        y = y[:,:-preprocess.block_size*preprocess.sequence_size]

        return z, y, amp, alpha, S_filtered_noise.reshape(bs,
                                                       -1,
                                                       preprocess.block_size//2+1, 2)


    def multiScaleFFT(self, x, overlap=75/100, amp = lambda x: x[:,:,:,0]**2 + x[:,:,:,1]**2):
        stfts = []
        for i,scale in enumerate(preprocess.fft_scales):
            stfts.append(amp(
                torch.stft(x,
                           n_fft=scale,
                           window=self.windows[i],
                           hop_length=int((1-overlap)*scale),
                           center=False)
            ))
        return stfts
