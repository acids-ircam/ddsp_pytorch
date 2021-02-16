import yaml
from scipy.io import wavfile
import numpy as np
from effortless_config import Config
from types import SimpleNamespace
from einops import rearrange
from os import path, makedirs
from ddsp.core import extract_pitch

with open("config.yaml", "r") as config:
    config = SimpleNamespace(**yaml.safe_load(config))

sr, x = wavfile.read(config.data["data_location"])
assert sr == config.preprocess["sampling_rate"]

n_signal = config.preprocess["signal_length"]
pad = (n_signal - (x.shape[0] % n_signal)) % n_signal

x = np.pad(x, ((0, pad), (0, 0)))

# pitch = extract_pitch(x[:, 0], sr, config.preprocess["block_size"])

x = rearrange(
    x,
    "(batch n_signal) channel -> channel batch n_signal",
    n_signal=n_signal,
)

# pitch = pitch.reshape(x.shape[1], -1).astype(np.float32)

out_dir = config.preprocess["out_dir"]
makedirs(out_dir, exist_ok=True)

np.save(path.join(out_dir, "signals.npy"), x[0])

np.save(
    path.join(out_dir, "pitchs.npy"),
    x[1, :, ::config.preprocess["block_size"]],
)

np.save(
    path.join(out_dir, "loudness.npy"),
    x[2, :, ::config.preprocess["block_size"]],
)
