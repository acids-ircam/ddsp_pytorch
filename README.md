# Differentiable Digital Signal Processing

Implementation of the [DDSP model](https://github.com/magenta/ddsp) using PyTorch. This implementation can be exported to a torchscript model, ready to be used inside a realtime environment (see [this video](https://www.youtube.com/watch?v=_U6Bn-1FDHc)).

## Usage

Edit the `config.yaml` file to fit your needs (audio location, preprocess folder, sampling rate, model parameters...), then preprocess your data using

```bash
python preprocess.py
```

You can then train your model using

```bash
python train.py --name mytraining --steps 10000000 --batch 16 --lr .001
```

Each flag is an override of the configuration provided in `config.yaml`.

You can monitor the progress with tensorboard

```bash
tensorboard models/train
```

Once trained, export it using

```bash
python export.py --run models/mytraining
```

It will produce a file named `ddsp_pretrained_mytraining.ts`, that you can use inside a python environment like that

```python
import torch

model = torch.jit.load("ddsp_pretrained_mytraining.ts")

pitch = torch.randn(1, 200, 1)
loudness = torch.randn(1, 200, 1)

audio = model(pitch, loudness)
```
