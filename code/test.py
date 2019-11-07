#%% -*- coding: utf-8 -*-

import librosa
import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand((100))
plt.plot(x, linewidth=4)
plt.savefig('images/noise.pdf')

plt.figure()
x = np.sin(np.linspace(0, 2 * np.pi, 100))
plt.plot(x, 'k', linewidth=4)
x = np.sin(np.linspace(0, 4 * np.pi, 100))
plt.plot(0.6 * x, 'r', linewidth=4)
x = np.sin(np.linspace(0, 8 * np.pi, 100))
plt.plot(0.4 * x, 'g', linewidth=4)
x = np.sin(np.linspace(0, 9 * np.pi, 100))
plt.plot(0.2 * x, 'b', linewidth=4)
plt.savefig('images/harmonic.pdf')

plt.figure()
x = np.sin(np.linspace(0, 30 * np.pi, 200))
plt.plot(x * (np.exp(np.linspace(3, 0, 200))-1), 'k', linewidth=4)
plt.savefig('images/decay.pdf')

plt.figure()
x, sr = librosa.core.load('/Users/esling/EMILIE/Trip-HopLoop1/Acid2techGnr8Kick/Methlab_PACKS/DatBassTho/Barbarix_MethLab_Bass_01.wav')
plt.plot(x[2000:18000]* (np.exp(np.linspace(2, 0, 16000))), 'k', linewidth=3)
plt.savefig('images/snare.pdf')