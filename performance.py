#%%
import torch
torch.set_grad_enabled(False)
from time import time
import math
from tqdm import tqdm
from effortless_config import Config


class args(Config):
    MODEL = None
    N_RUN = 10


args.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.jit.load(args.MODEL).eval().to(device)
sr = model.ddsp.sampling_rate

N = 2**(math.ceil(math.log2(sr)))
x = torch.randn(1, N, 1).to(device)

n_run = args.N_RUN
mean = 0
nel = 0

for i in tqdm(range(n_run), desc="testing..."):
    st = time()
    y = model(x, x)
    nel += 1
    mean += (time() - st - mean) / nel

realtime = N / (mean * sr)
smiley = ":)" if realtime >= 1 else ":("

print("\n")
print(
    f"average of {1000*mean:.2f}ms to generate {1000*N/sr:.2f}ms over {n_run} trials on device {device}"
)
print(f"generation is {realtime:.2f}x realtime {smiley}")
print(80 * "-")
# %%
