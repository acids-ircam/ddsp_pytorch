import numpy as np
import soundfile as sf
import librosa as li
from tqdm import tqdm
import os
from torch_ddsp.hparams import preprocess
from torch_ddsp.ddsp import NeuralSynth
import torch
from glob import glob
from pyworld import dio
import crepe

multiScaleFFT = NeuralSynth().multiScaleFFT
amp = lambda x: x[:,:,0]**2 + x[:,:,1]**2

def getSmoothLoudness(x, block_size, kernel_size=8):
    win = np.hamming(kernel_size)

    x = x.reshape(-1, block_size)
    lo = np.log(np.mean(x**2, -1) + 1e-15)
    lo = lo.reshape(-1)

    lo = np.convolve(lo, win, "same")
    return lo

def getFundamentalFrequency(x):
    sr         = preprocess.samplerate
    block_size = preprocess.block_size
    hop = int(1000 * block_size / sr)

    if preprocess.f0_estimation == "dio":
        f0 = dio(x.astype(np.float64), sr,
                 frame_period=hop,
                 f0_floor=50,
                 f0_ceil=2000)[0]
    elif preprocess.f0_estimation == "crepe":
        f0 = crepe.predict(x, sr, step_size=hop, verbose=False)[1]

    return f0[:preprocess.sequence_size].astype(np.float)

class BatchSoundFiles:
    def __init__(self, wav_list):
        self.wavs = wav_list
    def read(self):
        mod = preprocess.block_size * preprocess.sequence_size
        for head,wav in enumerate(self.wavs):
            wav = li.load(wav, preprocess.samplerate)[0]
            wav = wav[:mod*(len(wav)//mod)].reshape(-1,mod)
            for i in range(wav.shape[0]):
                yield wav[i]

    def __len__(self):
        mod = preprocess.block_size * preprocess.sequence_size
        return sum([mod*(len(wav)//mod) for wav in self.wavs])


def process(filename, block_size, sequence_size):
    output = preprocess.output_dir
    os.makedirs(output, exist_ok=True)

    sound = BatchSoundFiles(glob(filename))
    batch = len(sound) // (block_size * sequence_size)
    print(f"Splitting data into {batch} examples of {sequence_size}-deep sequences of {block_size} samples.")


    scales = preprocess.fft_scales
    sp = []
    for scale, ex_sp in zip(scales,multiScaleFFT(torch.randn(block_size * sequence_size),amp=amp)):
        sp.append(np.memmap(f"{output}/sp_{scale}.npy",
                            dtype=np.float32,
                            shape=(batch, ex_sp.shape[0], ex_sp.shape[1]),
                            mode="w+"))

    lo        = np.zeros([batch,sequence_size])
    f0        = np.zeros([batch, sequence_size])
    raw_audio = np.zeros([batch, sequence_size*block_size])
    index     = np.zeros([batch])

    in_point  = 0
    last_file = 0

    for file_index, x in sound.read():
        index[b] = file_index

        for i,msstft in enumerate(multiScaleFFT(torch.from_numpy(x).float(), amp=amp)):
            sp[i][b,:,:] = msstft.detach().numpy()

        lo[b,:] = getSmoothLoudness(x, preprocess.block_size, preprocess.kernel_size)
        f0[b,:] = getFundamentalFrequency(x)

        raw_audio[b,:] = x

        if (file_index != last_file) or (b==batch-1):
            # print(f"\nnormalization", in_point, b)
            last_file = file_index

            mean_loudness = np.mean(lo[in_point:b])
            std_loudness  = np.std(lo[in_point:b])

            lo[in_point:b] -= mean_loudness
            lo[in_point:b] /= std_loudness

            in_point = b



    np.save(f"{output}/lo.npy", lo)
    np.save(f"{output}/f0.npy", f0)
    np.save(f"{output}/index.npy", index)
    np.save(f"{output}/raw_audio.npy", raw_audio)


if __name__ == '__main__':
    process(preprocess.input_filename,
            preprocess.block_size,
            preprocess.sequence_size)
