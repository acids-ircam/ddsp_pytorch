from torch_ddsp.ddsp import NeuralSynth
from torch_ddsp.hparams import preprocess
import numpy as np
import torch
import librosa as li
from argparse import ArgumentParser
import soundfile as sf
from os import path
from pyworld import dio


parser = ArgumentParser(description="Reconstruction of an input audio sample.")
parser.add_argument("input",type=str, help="Audio to reconstruct")
parser.add_argument("--state", type=str, default=None, help="Model state to load")
parser.add_argument("--transpose", type=int, default=0, help="Transposition amount (semitone)")
parser.add_argument("--std-factor", type=float, default=1, help="Standard deviation modulation")
args = parser.parse_args()

x,fs = li.load(args.input, preprocess.samplerate)

f0,_ = dio(x.astype(np.float64), fs, frame_period=10)


lo = np.asarray([np.mean(x[i*preprocess.block_size:(i+1)*preprocess.block_size]**2)\
      for i in range(len(x)//preprocess.block_size)])
lo = np.log(lo + 1e-15)
lo = lo.reshape(1,-1)



mean, std = np.mean(lo), np.std(lo)

f0 *= 2**(args.transpose/12)
std *= args.std_factor

lo -= mean
lo /= std

N = min(f0.shape[-1],lo.shape[-1])
f0, lo = f0[:N],lo[:,:N]

f0 = torch.from_numpy(f0).float().reshape(1,-1,1)
lo = torch.from_numpy(lo).float().reshape(1,-1,1)


NS = NeuralSynth()
if args.state is not None:
    state = torch.load(args.state, map_location="cpu")[1]
    NS.load_state_dict(state)

with torch.no_grad():
    out,_,_,_ = NS(f0,lo, True, True)

out = out.detach().cpu().numpy().reshape(-1)

sf.write("reconstruction.wav", out, preprocess.samplerate)
