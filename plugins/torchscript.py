import torch
from torch_ddsp.ddsp import NeuralSynth, IncrementalNS
import argparse

parser = argparse.ArgumentParser(description="Conversion of a pretrained DDSP model to torchscript")
parser.add_argument("--state", type=str, default=None, help="State to load")
args = parser.parse_args()

NS = NeuralSynth()
NS.load_state_dict(torch.load(args.state, map_location="cpu")[1])
INS = IncrementalNS(NS)

f0 = torch.randn(1,1,1)
lo = torch.randn(1,1,1)
hx = torch.randn(1,1,512)

traced = torch.jit.trace(INS, [f0, lo, hx])
traced.save("ddsp.torchscript")
