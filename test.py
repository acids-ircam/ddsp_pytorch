import torch
import numpy as np
from scipy.io import wavfile
from ddsp.core import extract_centroid, extract_loudness, extract_pitch
import numpy as np
import librosa

# Load the audio file
y, sampling_rate = librosa.load("./violin.wav")

block_size = 160

centroid = extract_centroid(y, sampling_rate, block_size)
pitch = extract_pitch(y, sampling_rate, block_size)
loudness = extract_loudness(y, block_size)

# Model
model = torch.jit.load("./export/ddsp_gamba_pretrained.ts")

audio = model(torch.tensor(pitch, dtype=torch.float), torch.tensor(centroid, dtype=torch.float), torch.tensor(loudness, dtype=torch.float))
audio = audio.squeeze().detach().numpy()

# Scale the audio data to the appropriate range for the desired data type (e.g., int16)
scaled_data = np.int16(audio *
                       32767)  # Scale to the range of 16-bit signed integers

# Write the WAV file
file_path = 'output.wav'
wavfile.write(file_path, sampling_rate, scaled_data)
