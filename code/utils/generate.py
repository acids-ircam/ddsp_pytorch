# -*- coding: utf-8 -*-

#%%

"""
###################

Subscript to generate MEL and MFCC from the synth dataset

###################
"""

import os
import glob
import numpy as np
import librosa

datadir = '/fast-2/datasets/flow_synthesizer_new/toy'
mel_dir = datadir + '/mel/'
mfcc_dir = datadir + '/mfcc/'

param_files = sorted(glob.glob(datadir + '/raw/*.npz'))
for p in range(len(param_files)):
    print(param_files[p])
    audio = np.load(param_files[p])['audio']
    f_name, _ = os.path.splitext(os.path.basename(param_files[p]))
    print(f_name)
    # Compute mel
    mel_file = mel_dir + f_name + '.npy'
    b_mel = librosa.feature.melspectrogram(audio, sr=22050, n_fft=2048, n_mels=128, hop_length=1024, fmin=30, fmax=11000)
    np.save(mel_file, b_mel)
    mfcc_file = mfcc_dir + f_name + '.npy'
    b_mfcc = librosa.feature.mfcc(audio, sr=22050, n_mfcc=13)
    np.save(mfcc_file, b_mfcc)
    
#%%  

"""
###################

Subscript to find 4 seconds sub-windows that fit a given pitch

###################
"""

import glob
import librosa
import numpy as np

def extract_max(pitches,magnitudes, shape):
    new_pitches = []
    new_magnitudes = []
    for i in range(0, shape[1]):
        new_pitches.append(np.max(pitches[:,i]))
        new_magnitudes.append(np.max(magnitudes[:,i]))
    return (new_pitches,new_magnitudes)

def freq2midi(freq):
    """ Given a frequency in Hz, returns its MIDI pitch number. """
    MIDI_A4 = 69   # MIDI Pitch number
    FREQ_A4 = 440. # Hz
    return int(12 * (np.log2(freq) - np.log2(FREQ_A4)) + MIDI_A4)

def smooth(x, window_len=11, window='hanning'):
    if window_len < 3:
            return x
    s = np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    w = eval('np.'+window+'(window_len)')
    y = np.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]

out_dir = '/Users/esling/Datasets/synth_testing/'
datadir = '/Users/esling/Datasets/instruments_solo_recordings/'
dirs = ['neurofunk', 'violin', 'violin_piano', 'voice']
data_files = []
for d in dirs:
    tmp_files = sorted(glob.glob(datadir + d + '/*.mp3'))
    data_files.extend(tmp_files)
cur_id = 0
# Go through files
for f in range(len(data_files)):
    cur_files = data_files[f]
    print(cur_files)
    y, sr = librosa.core.load(cur_files)
    # Final ramp
    ramp = np.ones(4 * sr)
    ramp[(3*sr):] = np.linspace(1, 0, sr)
    for p in range(1000):
        s_id = int(np.random.rand() * (len(y) - (4 * sr)))
        cur_sig = y[s_id:int(s_id + (4 * sr))].copy()
        pitches, magnitudes = librosa.core.piptrack(y=cur_sig, sr=sr, S=None)
        pitches, mags = extract_max(pitches, magnitudes, pitches.shape)
        pitches = smooth(pitches)
        final_pitches = np.array([freq2midi(p+1e-3) for p in pitches])
        vals = np.sum(final_pitches == 60)
        if (vals > 40):
            cur_sig *= ramp
            librosa.output.write_wav(out_dir + 'synth_' + str(cur_id) + '.wav', cur_sig, sr)
            cur_id += 1
            
#%%  

"""
###################

Subscript to find 4 seconds sub-windows that fit a given pitch

###################
"""

import glob
import librosa

sr = 22050
datadir = '/fast-2/datasets/flow_synthesizer/vocal_testing'
data_files = sorted(glob.glob(datadir + '/*.wav'))
for f in range(len(data_files)):
    cur_files = data_files[f]
    print(cur_files)
    y, f_sr = librosa.core.load(cur_files, sr=sr)
    cur_len = len(y)
    # Final signal
    final_sig = np.zeros(sr * 4)
    if (cur_len <= sr * 4):
        final_sig[:(cur_len)] = y
    else:
        final_sig[:] = y[:(sr * 4)]
    # Final ramp
    ramp = np.ones(4 * sr)
    ramp[(3*sr):] = np.linspace(1, 0, sr)
    final_sig *= ramp
    librosa.output.write_wav(cur_files, final_sig, sr)