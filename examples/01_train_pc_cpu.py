"""
Train a PC on the CPU (Triton interpreter)
==========================================

A scaled-down CPU variant of ``01_train_pc.py`` for machines without a GPU. It runs PyJuice's
Triton kernels through Triton's CPU interpreter, which is orders of magnitude slower than a GPU,
so the settings are shrunk to finish in minutes:

* MNIST downsampled to 14x14 (196 variables, still 256 pixel values each),
* an HCLT with 16 latents (vs. 128),
* batches of 128 and an epoch of ``--train-size`` images (vs. all 60K).

Run (the interpreter must be enabled before ``pyjuice``/``triton`` are imported)::

    TRITON_INTERPRET=1 python examples/01_train_pc_cpu.py

The pipeline -- HCLT construction, ``juice.compile``, the mini-batch EM loop and the full-batch EM
epoch -- is the same as in ``01_train_pc.py``; only the CUDA-graph dry run is dropped (CPU only).
"""

import argparse
import math
import os
import time
import warnings

assert os.environ.get("TRITON_INTERPRET", "0") not in ("", "0"), \
    "Run with TRITON_INTERPRET=1 (Triton's CPU interpreter); this example has no GPU code path."

import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader

import pyjuice as juice

# The interpreter evaluates masked-out lanes too (e.g. `-inf - -inf`), which NumPy warns about;
# those lanes are discarded, exactly as on the GPU.
warnings.filterwarnings("ignore", category = RuntimeWarning, module = "triton")

parser = argparse.ArgumentParser()
parser.add_argument("--side", type = int, default = 14, help = "downsample MNIST to side x side pixels")
parser.add_argument("--num-latents", type = int, default = 16)
parser.add_argument("--batch-size", type = int, default = 128)
parser.add_argument("--train-size", type = int, default = 1280, help = "training images per epoch")
parser.add_argument("--valid-size", type = int, default = 512)
parser.add_argument("--epochs", type = int, default = 5, help = "mini-batch EM epochs")
parser.add_argument("--data-root", type = str, default = "../data")
parser.add_argument("--save-dir", type = str, default = ".",
                    help = "where to save the trained PCs (`pc_minibatch_em.jpc` and `pc_final.jpc`)")
args = parser.parse_args()

torch.manual_seed(0)

# %%
# Load (and downsample) MNIST
# ---------------------------

def load(train: bool):
    ds = torchvision.datasets.MNIST(root = args.data_root, train = train, download = True)
    x = ds.data.float()
    if args.side != 28:
        # Area-average to side x side, then round back to the 256 integer pixel values
        x = torch.nn.functional.interpolate(x[:, None], size = (args.side, args.side), mode = "area")[:, 0]
    return x.round().long().reshape(x.size(0), -1)

train_data = load(train = True)
valid_data = load(train = False)
num_vars = train_data.size(1)

train_loader = DataLoader(TensorDataset(train_data[torch.randperm(train_data.size(0))[:args.train_size]]),
                          batch_size = args.batch_size, shuffle = True, drop_last = True)
valid_loader = DataLoader(TensorDataset(valid_data[:args.valid_size]),
                          batch_size = args.batch_size, shuffle = False, drop_last = True)

def bpd(ll):
    # bits per dimension: the usual way to report image log-likelihoods
    return -ll / (num_vars * math.log(2))

# %%
# Create and compile the PC
# -------------------------

t0 = time.time()
# The Chow-Liu tree backbone is estimated from (a subset of) the training data
ns = juice.structures.HCLT(train_data[:10000].float(), num_latents = args.num_latents)
pc = juice.compile(ns)
pc.to(torch.device("cpu"))
print(f"HCLT over {num_vars} variables with {args.num_latents} latents: "
      f"{pc.num_nodes} nodes, {pc.num_sum_params} sum params; built + compiled in {time.time() - t0:.1f}s")

def evaluate():
    with torch.no_grad():
        return sum(pc(batch[0]).mean().item() for batch in valid_loader) / len(valid_loader)

# %%
# Mini-batch EM
# -------------

optimizer = juice.optim.MiniBatchEM(pc, step_size = 0.1, pseudocount = 0.1)

for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    train_ll = 0.0
    for batch in train_loader:
        x = batch[0]
        lls = pc(x)                 # forward: log p(x) per sample
        lls.mean().backward()       # backward: accumulates parameter flows (not autograd gradients)
        train_ll += lls.mean().item()
        optimizer.step()            # EM update from the flows, then the accumulator is reset
    train_ll /= len(train_loader)

    t1 = time.time()
    test_ll = evaluate()
    t2 = time.time()
    print(f"[Mini-batch EM epoch {epoch}/{args.epochs}][train LL: {train_ll:.2f} ({bpd(train_ll):.3f} bpd); "
          f"val LL: {test_ll:.2f} ({bpd(test_ll):.3f} bpd)].....[train {t1-t0:.1f}s; val {t2-t1:.1f}s]", flush = True)

# Save the PC after mini-batch EM (the full-batch step below can overfit a small training set)
os.makedirs(args.save_dir, exist_ok = True)
juice.save(os.path.join(args.save_dir, "pc_minibatch_em.jpc"), pc)

# %%
# Full-batch EM (one epoch)
# -------------------------

t0 = time.time()
pc.init_param_flows(flows_memory = 0.0)      # zero the flow accumulator manually
train_ll = 0.0
for batch in train_loader:
    lls = pc(batch[0])
    lls.mean().backward()                    # flows accumulate over the whole epoch
    train_ll += lls.mean().item()
pc.mini_batch_em(step_size = 1.0, pseudocount = 0.01)   # step size 1.0 == a plain EM step
train_ll /= len(train_loader)

t1 = time.time()
test_ll = evaluate()
t2 = time.time()
print(f"[Full-batch EM epoch 1/1][train LL: {train_ll:.2f} ({bpd(train_ll):.3f} bpd); "
      f"val LL: {test_ll:.2f} ({bpd(test_ll):.3f} bpd)].....[train {t1-t0:.1f}s; val {t2-t1:.1f}s]", flush = True)

juice.save(os.path.join(args.save_dir, "pc_final.jpc"), pc)
print(f"Saved the PCs to {os.path.abspath(args.save_dir)}")
