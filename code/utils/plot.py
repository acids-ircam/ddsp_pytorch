# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

def plot_batch(batch, name=None):
    # Create one big image for plot
    print(batch.shape)
    img = np.zeros((batch.shape[2] * 4 + 3, batch.shape[3] * 4 + 3))
    for b in range(batch.shape[0]):
        row = int(b / 4); col = int(b % 4)
        r_p = row * batch.shape[2] + row; c_p = col * batch.shape[3] + col
        img[r_p:(r_p+batch.shape[2]),c_p:(c_p+batch.shape[3])] = batch[b].squeeze()
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    if (name is not None):
        plt.savefig(name + '.pdf')
        plt.close()

def plot_batch_detailed(batch, n_examples = 4, name=None):
    audio, f0, loudness, fft = batch
    # Create one big image for plot
    fig, axes = plt.subplots(n_examples, 9, figsize=(20, 10))
    for b in range(min(n_examples, audio.shape[0])):
        axes[b, 0].plot(audio[b].squeeze(0)[:2000])
        axes[b, 1].plot(f0[b].squeeze(0), 'r')
        axes[b, 2].plot(loudness[b].squeeze(0), 'g')
        for w in range(6):
            axes[b, w+3].plot(np.log(fft[w][b, :, 0] + 1e-3))#, aspect='auto')
    if (name is not None):
        fig.savefig(name + '.pdf')
        fig.close()
        
def compare_batch_detailed(batch, params, batch_r=None, params_r=None, synth_r=None, wave=None, synth_wave=None, name=None):
    # Create one big image for plot
    if len(batch.shape)>3:
        batch = batch[:,0]
        if not batch_r is None:
            batch_r = batch_r[:,0]
    fig, axes = plt.subplots(batch.shape[0], 6, figsize=(10, 20))
    for b in range(batch.shape[0]):
        axes[b, 0].imshow(batch[b], aspect='auto')
        if (batch_r is not None):
            axes[b, 1].imshow(batch_r[b], aspect='auto')
        axes[b, 2].bar(np.linspace(0, params[b].shape[0]-1, params[b].shape[0]), params[b], color='b')
        if (params is not None):
            axes[b, 3].bar(np.linspace(0, params[b].shape[0]-1, params_r[b].shape[0]), params_r[b], color='r')
        if (wave is not None):
            axes[b, 4].plot(wave[b].numpy())
        if (synth_wave is not None):
            axes[b, 5].plot(synth_wave[b])
        #for w in range(4):
        #    axes[b, w+2].plot(batch[b, int(float((w+1.)/5) * batch[b].shape[0])].numpy())
    if (name is not None):
        fig.savefig(name + '.pdf')
        plt.close()
        
