import torch
import numpy as np
from scipy.io import wavfile
from effortless_config import Config
from ddsp.core import extract_centroid, extract_loudness, extract_pitch
import numpy as np
import yaml
import librosa


@torch.no_grad()
def mean_std_loudness_arr(loudness):
    mean = 0
    std = 0
    n = 0
    for l in loudness:
        n += 1
        mean += (l.mean().item() - mean) / n
        std += (l.std().item() - std) / n
    return mean, std


def extract_features(
    signal,
    sampling_rate,
    block_size,
):

    p = extract_pitch(signal, sampling_rate, block_size)
    c = extract_centroid(signal, sampling_rate, block_size)
    l = extract_loudness(signal, block_size)

    return p, c, l


def main():

    class args(Config):
        CONFIG = "config.yaml"

    with open(args.CONFIG, "r") as config:
        config = yaml.safe_load(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("GPU Is Available: ", torch.cuda.is_available())

    # Load the model
    model = torch.jit.load(config["timbre_transfer"]["model"])

    # Load the audio file
    signal, sampling_rate = librosa.load(
        config["timbre_transfer"]["input_file"])

    p, c, l = extract_features(signal, sampling_rate,
                               config["model"]["block_size"])

    mean_loudness, std_loudness = mean_std_loudness_arr(l)

    p = torch.tensor(p, dtype=torch.float).unsqueeze(0).unsqueeze(2).to(device)
    c = torch.tensor(c, dtype=torch.float).unsqueeze(0).unsqueeze(2).to(device)
    l = torch.tensor(l, dtype=torch.float).unsqueeze(0).unsqueeze(2).to(device)

    l = (l - mean_loudness) / std_loudness

    y = model(p, c, l).squeeze(-1)

    audio = y.squeeze().detach().numpy()

    # Scale the audio data to the appropriate range for the desired data type (e.g., int16)
    scaled_data = np.int16(
        audio * 32767)  # Scale to the range of 16-bit signed integers

    # Write the WAV file
    file_path = 'output.wav'
    wavfile.write(file_path, sampling_rate, scaled_data)


if __name__ == "__main__":
    main()
