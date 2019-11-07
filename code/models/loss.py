# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import librosa

def multinomial_loss(x_logit, x):
    batch_size = x.shape[0]
    # Reshape input
    x_logit = x_logit.view(batch_size, -1, x.shape[1])
    # Take softmax
    x_logit = F.log_softmax(x_logit, 1)
    # make integer class labels
    target = (x * (x_logit.shape[1] - 1)).long()
    # computes cross entropy over all dimensions separately:
    ce = F.nll_loss(x_logit, target, weight=None, reduction='mean')
    # Return summed loss
    return ce.sum()

def multinomial_mse_loss(x_logit, x):
    b_size = x.shape[0]
    n_out = x.shape[1]
    # Reshape our logits
    x_rep = x_logit.view(b_size, -1, n_out)
    x_multi = x_rep[:, :-1, :].view(b_size, -1)
    # Take the multinomial loss
    multi_loss = multinomial_loss(x_multi, x)
    # Retrieve MSE part of loss
    x_mse = x_rep[:, -1, :]
    # Compute values for MSE
    mse_loss = F.mse_loss(x_mse, x)
    return mse_loss + multi_loss

def spectral_losses(x_tilde, x, test_loader, args=None, raw=False):
    if (raw): #x_tilde is audio, x is data from dataset (mel, mfcc, mel_mfcc) cannot be batch
        if (args.data in ['mel', "mel_mfcc"]):
            b_mel = librosa.feature.melspectrogram(x_tilde, sr=22050, n_fft=2048, n_mels=64, hop_length=1024, fmin=30, fmax=11000)
            b_mel = b_mel[:64,:80]
        if (args.data == 'mfcc'):
            b_mel = librosa.feature.mfcc(x_tilde, sr=22050, n_mfcc=13)
        b_mel = torch.from_numpy(b_mel).to(x.device).float()
        # Un-normalize the input
        in_mel = (x[0] * test_loader.dataset.vars["mel"]) + test_loader.dataset.means["mel"]
        if (args.data in ['mel', "mel_mfcc"]):
            in_mel = torch.exp(in_mel) - 1e-3
    else: #both x_tilde and x must be un-normalized must be (batch, channel, H, W)
        # Un-normalize the inputs
        b_mel = (x_tilde[:,0] * test_loader.dataset.vars["mel"]) + test_loader.dataset.means["mel"]
        if (args.data in ['mel', "mel_mfcc"]):
            b_mel = torch.exp(b_mel) - 1e-3
        in_mel = (x[:,0] * test_loader.dataset.vars["mel"]) + test_loader.dataset.means["mel"]
        if (args.data in ['mel', "mel_mfcc"]):
            in_mel = torch.exp(in_mel) - 1e-3
    # Compute MSE loss
    mse_loss = F.mse_loss(in_mel, b_mel).mean().unsqueeze(0)
    # Compute spectral convergence
    frobenius_diff = torch.sqrt(torch.sum(torch.pow(in_mel - b_mel, 2)))
    frobenius_input_amp = torch.sqrt(torch.sum(torch.pow(in_mel, 2)))
    spectral_convergence = torch.mean(frobenius_diff/frobenius_input_amp)
    sc_loss = spectral_convergence.unsqueeze(0)
    # Compute log magnitude differences
    if (args.data != 'mfcc'):
        log_stft_mag_diff = torch.sum(torch.abs(torch.log(in_mel + 1e-9) - torch.log(b_mel + 1e-9)))
        log_stft_ref = torch.sum(torch.abs(torch.log(in_mel + 1e-9)))
    else:
        log_stft_mag_diff = torch.sum(torch.abs(in_mel - b_mel))
        log_stft_ref = torch.sum(torch.abs(in_mel))
    log_stft_mag = torch.mean(log_stft_mag_diff/(log_stft_ref + 1e-9))
    lm_loss = log_stft_mag.unsqueeze(0)
    full_loss = mse_loss + sc_loss + lm_loss
    return full_loss, mse_loss, sc_loss, lm_loss, b_mel
        
