import torch_ddsp.central_training as ct
from torch_ddsp.ddsp import NeuralSynth
from torch_ddsp.loader import Loader
from torch_ddsp import hparams
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ==========[------]======[------]==========
#           |      |      |      | warmup_noise + warmup_synth
#           |      |      | warmup_noise
#           |      | warmup_conv + warmup_synth
#           | warmup_conv


def learning_scheme(step):
    amp_pass   = True if step > hparams.train.warmup_amp else False
    synth_pass = True
    noise_pass = True if step > hparams.train.warmup_noise else False
    conv_pass  = True if step > hparams.train.warmup_conv  else False

    if (step > hparams.train.warmup_noise) and\
    (step <= hparams.train.warmup_noise + hparams.train.warmup_synth):
        synth_pass = False

    elif (step > hparams.train.warmup_conv) and\
    (step <= hparams.train.warmup_conv + hparams.train.warmup_synth):
        synth_pass = False

    return amp_pass, synth_pass, noise_pass, conv_pass




def train_step(model, opt_list, step, data_list):
    amp_pass, synth_pass, noise_pass, conv_pass = learning_scheme(step)

    opt_list[0].zero_grad()

    idx       = data_list.pop(0)
    raw_audio = data_list.pop(0)
    lo        = data_list.pop(0)
    f0        = data_list.pop(0)
    stfts     = data_list

    z, output, amp, alpha, S_noise = model(raw_audio.unsqueeze(1),
                                           f0.unsqueeze(-1),
                                           lo.unsqueeze(-1),
                                           amp_pass,
                                           synth_pass,
                                           noise_pass,
                                           conv_pass)

    stfts_rec = model.multiScaleFFT(output)

    lin_loss = sum([torch.mean(abs(stfts[i] - stfts_rec[i])) for i in range(len(stfts_rec))])
    log_loss = sum([torch.mean(abs(torch.log(stfts[i]+1e-4) - torch.log(stfts_rec[i] + 1e-4))) for i in range(len(stfts_rec))])

    z_mean,z_var = z
    reg_loss = torch.mean(torch.exp(z_var)**2 + z_mean**2 - z_var - 1)

    collapse_loss = torch.mean(-torch.log(amp + 1e-10))

    loss = lin_loss + log_loss + .1 * reg_loss

    # Used to avoid a silence collapse during early stage of training
    if step < 1000:
        loss += .1 * collapse_loss

    loss.backward()
    opt_list[0].step()

    if step % 50 == 0:
        # INFERED PARAMETERS PLOT ##############################################

        plt.figure(figsize=(15,5))
        plt.subplot(131)
        plt.plot(np.log(amp[0].detach().cpu().numpy()))
        plt.title("Infered instrument amplitude")
        plt.xlabel("Time (ua)")
        plt.ylabel("Log amplitude (dB)")

        plt.subplot(132)

        alpha_n = alpha.cpu().detach().numpy()[0]
        histogram = [np.histogram(alpha_n[:,i], bins=100, range=(0,1))[0] for i in range(alpha_n.shape[-1])]
        histogram = np.asarray(histogram)
        plt.imshow(np.log(histogram.T+1e-3), origin="lower", aspect="auto", cmap="magma")
        plt.xlabel("Harmonic number")
        plt.ylabel("Density")
        plt.title("Harmonic repartition")

        plt.subplot(133)

        S_noise = S_noise[0].cpu().detach().numpy()
        S_noise = S_noise[:,:,0] ** 2 + S_noise[:,:,1] ** 2
        plt.imshow(S_noise.T, origin="lower", aspect="auto", cmap="magma")
        plt.title("Noise output")
        plt.xlabel("Time (ua)")
        plt.ylabel("Frequency (ua)")

        writer.add_figure("Infered parameters", plt.gcf(), step)
        plt.close()

        # RECONSTRUCTION PLOT ##################################################
        plt.figure(figsize=(15,5))
        plt.subplot(131)
        plt.plot(output[0].detach().cpu().numpy().reshape(-1))
        plt.title("Rec waveform")

        plt.subplot(132)
        plt.imshow(np.log(stfts[2][0].cpu().detach().numpy()+1e-4), cmap="magma", origin="lower", aspect="auto")
        plt.title("Original spectrogram")
        plt.colorbar()

        plt.subplot(133)
        plt.imshow(np.log(stfts_rec[2][0].cpu().detach().numpy()+1e-4), cmap="magma", origin="lower", aspect="auto")
        plt.title("Reconstructed spectrogram")
        plt.colorbar()
        writer.add_figure("reconstruction", plt.gcf(), step)
        plt.close()

        try:
            writer.add_audio("Reconstruction", output[0].reshape(-1)/torch.max(output[0].reshape(-1)), step, 16000)
            writer.add_audio("Original", raw_audio[0].reshape(-1), step, 16000)
        except:
            print("Could not export audio (NaN ?)")

    return {"lin_loss":lin_loss.item(),
            "log_loss":log_loss.item(),
            "kl_loss":reg_loss.item(),
            "collapse_loss":collapse_loss.item()}

trainer = ct.Trainer(**ct.args.__dict__)

trainer.set_model(NeuralSynth)
trainer.setup_model()

trainer.add_optimizer(torch.optim.Adam(trainer.model.parameters()))
trainer.setup_optim()

trainer.set_dataset_loader(Loader)
trainer.set_lr(np.linspace(1e-3, 1e-4, ct.args.step))

trainer.set_train_step(train_step)

writer = SummaryWriter(f"runs/{ct.args.name}/")

for i,losses in enumerate(trainer.train_loop()):
    for loss in losses:
        writer.add_scalar(loss, losses[loss], i)
