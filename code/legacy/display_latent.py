import seaborn
seaborn.set()

import torch
from torch.utils.data import DataLoader
from torch_ddsp.loader import Loader
from torch_ddsp.hparams import preprocess
from torch_ddsp.ddsp import NeuralSynth
from argparse import ArgumentParser
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser(description="Display latent points for a dataset")
    parser.add_argument("--state", type=str, default=None, help="State to load")
    args = parser.parse_args()


    loader = Loader(preprocess.output_dir)
    dataloader = DataLoader(loader, batch_size=1, shuffle=True)

    synth  = NeuralSynth()

    if args.state is not None:
        synth.load_state_dict(torch.load(args.state, map_location="cpu")[1])
        print(f"State {args.state} loaded!")

    for i,batch in enumerate(tqdm(dataloader)):
        if i>0:
            break

        index = batch.pop(0)
        raw   = batch.pop(0).unsqueeze(1)
        lo    = batch.pop(0).unsqueeze(-1)
        f0    = batch.pop(0).unsqueeze(-1)


        with torch.no_grad():
            z = synth(raw, f0, lo)[0]
            index = index.reshape(-1,).repeat_interleave(z.shape[1])
            z = z.reshape(-1,2)

            plt.ion()
            plt.figure(figsize=(10,5))
            plt.subplot(121)
            plt.plot(raw.reshape(-1))
            plt.title(index.reshape(-1)[0].item())

            avancement = np.linspace(0,raw.reshape(-1).shape[0],z.shape[0])

            for i,point in enumerate(z):
                plt.subplot(121)
                plt.plot(avancement[i], 0, "ro")
                plt.subplot(122)
                plt.xlim([-4,4])
                plt.ylim([-4,4])
                plt.plot(point[0], point[1], "r.")
                plt.pause(1/60)
