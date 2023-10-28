import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from ddsp.model import DDSP
from effortless_config import Config
from os import path
from preprocess import Dataset
from tqdm import tqdm
from ddsp.core import multiscale_fft, safe_log, mean_std_loudness
import soundfile as sf
from einops import rearrange
from ddsp.utils import get_scheduler
import numpy as np


class args(Config):
    CONFIG = "config.yaml"
    ROOT = "models"


def main():
    with open(args.CONFIG, "r") as config:
        config = yaml.safe_load(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("GPU Is Available: ", torch.cuda.is_available())

    model = DDSP(**config["model"]).to(device)

    dataset = Dataset(config["preprocess"]["out_dir"])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        config["train"]["batch"],
        True,
        drop_last=True,
    )

    mean_loudness, std_loudness = mean_std_loudness(dataloader)

    config["data"]["mean_loudness"] = mean_loudness
    config["data"]["std_loudness"] = std_loudness

    writer = SummaryWriter(path.join(args.ROOT, config["train"]["name"]),
                           flush_secs=20)

    with open(path.join(args.ROOT, config["train"]["name"], "config.yaml"),
              "w") as out_config:
        yaml.safe_dump(config, out_config)

    opt = torch.optim.Adam(model.parameters(), lr=config["train"]["start_lr"])

    schedule = get_scheduler(
        len(dataloader),
        config["train"]["start_lr"],
        config["train"]["stop_lr"],
        config["train"]["decay"],
    )

    # scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule)

    best_loss = float("inf")
    mean_loss = 0
    n_element = 0
    step = 0
    epochs = int(np.ceil(config["train"]["steps"] / len(dataloader)))

    for e in tqdm(range(epochs)):
        for s, p, c, l in dataloader:
            s = s.to(device)
            p = p.unsqueeze(-1).to(device)
            c = c.unsqueeze(-1).to(device)
            l = l.unsqueeze(-1).to(device)

            l = (l - mean_loudness) / std_loudness

            y = model(p, c, l).squeeze(-1)

            ori_stft = multiscale_fft(
                s,
                config["train"]["scales"],
                config["train"]["overlap"],
            )
            rec_stft = multiscale_fft(
                y,
                config["train"]["scales"],
                config["train"]["overlap"],
            )

            loss = 0
            for s_x, s_y in zip(ori_stft, rec_stft):
                lin_loss = (s_x - s_y).abs().mean()
                log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
                loss = loss + lin_loss + log_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            writer.add_scalar("loss", loss.item(), step)

            step += 1

            n_element += 1
            mean_loss += (loss.item() - mean_loss) / n_element

        if not e % 10:
            writer.add_scalar("lr", schedule(e), e)
            writer.add_scalar("reverb_decay", model.reverb.decay.item(), e)
            writer.add_scalar("reverb_wet", model.reverb.wet.item(), e)
            # scheduler.step()
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(
                    model.state_dict(),
                    path.join(args.ROOT, config["train"]["name"], "state.pth"),
                )

            mean_loss = 0
            n_element = 0

            audio = torch.cat([s, y], -1).reshape(-1).detach().cpu().numpy()

            sf.write(
                path.join(args.ROOT, config["train"]["name"],
                          f"eval_{e:06d}.wav"),
                audio,
                config["preprocess"]["sampling_rate"],
            )


if __name__ == "__main__":
    main()
