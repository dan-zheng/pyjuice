"""
Use a trained PC: inference on the CPU (Triton interpreter)
===========================================================

Loads the HCLT trained by ``01_train_pc_cpu.py`` (14x14 MNIST) and demonstrates the queries a
probabilistic circuit answers EXACTLY -- the reason to use a PC over other generative models:

1. Likelihood scoring: detect out-of-distribution inputs, find the most typical / unusual digits.
2. Marginals: score images with missing pixels (pixels are summed out exactly, no imputation).
3. Conditionals: inpaint the hidden half of a digit, with a per-pixel distribution (uncertainty).
4. Sampling: draw new digits, unconditionally and conditioned on evidence.

Run (after training; the interpreter must be enabled before ``pyjuice``/``triton`` are imported)::

    TRITON_INTERPRET=1 python examples/01_train_pc_cpu.py --save-dir pcs
    TRITON_INTERPRET=1 python examples/01_inference_pc_cpu.py --model pcs/pc_minibatch_em.jpc --out-dir inference_out
"""

import argparse
import math
import os
import time
import warnings

assert os.environ.get("TRITON_INTERPRET", "0") not in ("", "0"), \
    "Run with TRITON_INTERPRET=1 (Triton's CPU interpreter); this example has no GPU code path."

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision

import pyjuice as juice

warnings.filterwarnings("ignore", category = RuntimeWarning, module = "triton")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, default = "pc_minibatch_em.jpc", help = "a PC saved by 01_train_pc_cpu.py")
parser.add_argument("--side", type = int, default = 14, help = "must match the side the PC was trained with")
parser.add_argument("--data-root", type = str, default = "../data")
parser.add_argument("--out-dir", type = str, default = "inference_out")
parser.add_argument("--num-score", type = int, default = 128, help = "images per group in the scoring demos")
parser.add_argument("--num-inpaint", type = int, default = 16, help = "digits to inpaint")
args = parser.parse_args()

torch.manual_seed(0)
os.makedirs(args.out_dir, exist_ok = True)
S = args.side
D = S * S
LN2 = math.log(2)


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def load(dataset_cls, train: bool):
    # The SAME preprocessing as 01_train_pc_cpu.py: area-average to SxS, round to 0..255
    ds = dataset_cls(root = args.data_root, train = train, download = True)
    x = ds.data.float()
    if S != 28:
        x = torch.nn.functional.interpolate(x[:, None], size = (S, S), mode = "area")[:, 0]
    return x.round().long().reshape(x.size(0), -1), ds.targets


def bpd(ll):
    return -ll / (D * LN2)


def auroc(pos_scores, neg_scores):
    # P(score of a random positive > score of a random negative), ties counted as half
    diff = pos_scores[:, None] - neg_scores[None, :]
    return ((diff > 0).float() + 0.5 * (diff == 0).float()).mean().item()


def show_grid(rows, row_titles, fname, col_titles = None, cmap = "gray", vmax = 255):
    n_rows, n_cols = len(rows), rows[0].size(0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize = (1.1 * n_cols, 1.2 * n_rows + 0.3), squeeze = False)
    for r, (imgs, title) in enumerate(zip(rows, row_titles)):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(imgs[c].reshape(S, S).float().numpy(), cmap = cmap, vmin = 0, vmax = vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(title, rotation = 0, ha = "right", va = "center", fontsize = 9)
            if col_titles is not None and r == 0:
                ax.set_title(col_titles[c], fontsize = 8)
    plt.tight_layout()
    path = os.path.join(args.out_dir, fname)
    plt.savefig(path, dpi = 110)
    plt.close(fig)
    print(f"  -> saved {path}")


# %%
# Load the trained PC and the data
# --------------------------------

t0 = time.time()
ns = juice.load(args.model)            # the saved node DAG (structure + parameters) ...
pc = juice.compile(ns, verbose = False)  # ... compiled again into a TensorCircuit
print(f"Loaded {args.model}: {pc.num_vars} variables, {pc.num_nodes} nodes ({time.time() - t0:.1f}s)")
assert pc.num_vars == D, f"the PC has {pc.num_vars} variables but --side {S} gives {D}"

mnist_train, _ = load(torchvision.datasets.MNIST, train = True)
mnist_test, mnist_labels = load(torchvision.datasets.MNIST, train = False)
fashion_test, _ = load(torchvision.datasets.FashionMNIST, train = False)

# Held-out digits: training used a random 1,280-image TRAINING subset and validated on test[:512],
# so test[512:] was never looked at.
N = args.num_score
digits = mnist_test[512:512 + N]


@torch.no_grad()
def log_likelihood(x, batch_size = 128):
    return torch.cat([pc(x[i:i + batch_size])[:, 0] for i in range(0, x.size(0), batch_size)])


# %%
# 1. Likelihood scoring: out-of-distribution detection
# ----------------------------------------------------

section("1. Likelihood scoring -- is this input like the training data?")
print("pc(x) returns the exact log p(x). A low score flags inputs unlike the training digits.\n")

groups = {
    "MNIST digits (held out)": digits,
    "Fashion-MNIST (clothes)": fashion_test[:N],
    "Inverted digits (255 - x)": 255 - digits,
    "Uniform noise": torch.randint(0, 256, (N, D)),
    "Blank image (all 0)": torch.zeros(N, D, dtype = torch.long),
}

t0 = time.time()
scores = {name: log_likelihood(x) for name, x in groups.items()}
print(f"  {'group':<28}{'mean log p(x)':>15}{'bits/pixel':>12}{'AUROC vs digits':>17}")
for name, ll in scores.items():
    au = "" if name.startswith("MNIST") else f"{auroc(scores['MNIST digits (held out)'], ll):.3f}"
    print(f"  {name:<28}{ll.mean().item():>15.1f}{bpd(ll.mean().item()):>12.3f}{au:>17}")
print(f"  (AUROC 1.0 = the score separates the group from real digits perfectly; {time.time() - t0:.1f}s)")

# The most typical / most unusual held-out digits, by their likelihood
ll_digits = scores["MNIST digits (held out)"]
order = torch.argsort(ll_digits, descending = True)
k = 8
show_grid([digits[order[:k]], digits[order[-k:]]],
          ["most likely", "least likely"], "1_typical_vs_unusual_digits.png",
          col_titles = None)
print(f"  most likely digits:  bits/pixel {', '.join(f'{bpd(v):.2f}' for v in ll_digits[order[:k]].tolist())}")
print(f"  least likely digits: bits/pixel {', '.join(f'{bpd(v):.2f}' for v in ll_digits[order[-k:]].tolist())}")


# %%
# 2. Marginals: missing pixels are summed out exactly
# ---------------------------------------------------

section("2. Marginals -- likelihood of a PARTIAL observation")
print("marginal(pc, x, missing_mask) = log p(observed pixels), summing the missing ones out exactly.\n")

# 2a. Exactness checks
x = digits[:16]
everything = torch.ones(D, dtype = torch.bool)
ll_none = juice.queries.marginal(pc, data = x, missing_mask = everything)[:, 0]
print(f"  check: every pixel missing -> log p() = {ll_none.abs().max().item():.2e} (exactly 0 = log 1)")

v = D // 2 + S // 2                                  # a pixel near the image centre
one = torch.zeros(D, dtype = torch.bool); one[v] = True
ll_marg = juice.queries.marginal(pc, data = x[:1], missing_mask = one)[0, 0].item()
all_values = x[:1].repeat(256, 1); all_values[:, v] = torch.arange(256)
ll_brute = torch.logsumexp(log_likelihood(all_values, batch_size = 256), dim = 0).item()
print(f"  check: pixel {v} missing -> marginal {ll_marg:.4f} vs brute-force sum over its 256 values "
      f"{ll_brute:.4f} (diff {abs(ll_marg - ll_brute):.1e})")

# 2b. Use case: out-of-distribution detection when half of each image is missing
#     (a per-example mask: a different random half for every image)
print("\n  Use case: flag clothes vs digits when a RANDOM 50% of each image's pixels is missing")
masks = torch.rand(2 * N, D) < 0.5
ll_digit_half = juice.queries.marginal(pc, data = digits, missing_mask = masks[:N])[:, 0]
ll_cloth_half = juice.queries.marginal(pc, data = fashion_test[:N], missing_mask = masks[N:])[:, 0]
n_obs = (~masks).sum(dim = 1).float()
print(f"  {'':<28}{'bits per OBSERVED pixel':>24}")
print(f"  {'MNIST digits':<28}{(-ll_digit_half / (n_obs[:N] * LN2)).mean().item():>24.3f}")
print(f"  {'Fashion-MNIST':<28}{(-ll_cloth_half / (n_obs[N:] * LN2)).mean().item():>24.3f}")
print(f"  AUROC digits vs clothes with 50% missing: {auroc(ll_digit_half / n_obs[:N], ll_cloth_half / n_obs[N:]):.3f} "
      f"(all pixels: {auroc(scores['MNIST digits (held out)'], scores['Fashion-MNIST (clothes)']):.3f})")


# %%
# 3. Conditionals: inpainting with uncertainty
# --------------------------------------------

section("3. Conditionals -- inpaint the bottom half of a digit")
print("conditional(pc, x, missing_mask, target_vars) = p(x_i | observed pixels) for every hidden pixel i,\n"
      "a full 256-way distribution per pixel, from one forward + backward pass.\n")

M = args.num_inpaint
x = mnist_test[512:512 + M]
hidden = torch.zeros(D, dtype = torch.bool); hidden[D // 2:] = True        # the bottom half
hidden_vars = hidden.nonzero()[:, 0].tolist()

t0 = time.time()
probs = juice.queries.conditional(pc, data = x, missing_mask = hidden, target_vars = hidden_vars)  # [M, |hidden|, 256]
print(f"  conditional: output {tuple(probs.shape)} = [images, hidden pixels, pixel values]; "
      f"rows sum to {probs.sum(-1).min().item():.4f}..{probs.sum(-1).max().item():.4f} ({time.time() - t0:.1f}s)")

values = torch.arange(256, dtype = torch.float)
cond_mean = (probs * values).sum(-1)                                        # minimises expected squared error
cond_std = ((probs * values ** 2).sum(-1) - cond_mean ** 2).clamp(min = 0).sqrt()
cond_mode = probs.argmax(-1).float()

def fill(img, bottom):
    out = img.clone().float(); out[:, hidden] = bottom; return out

truth = x[:, hidden].float()
pixel_mean = mnist_train.float().mean(dim = 0)[hidden]                      # baseline: average training digit
print(f"\n  RMSE on the hidden half (pixel values 0..255; lower is better):")
print(f"    PC conditional mean          {((cond_mean - truth) ** 2).mean().sqrt().item():6.1f}")
print(f"    PC conditional mode          {((cond_mode - truth) ** 2).mean().sqrt().item():6.1f}")
print(f"    baseline: mean training image {((pixel_mean - truth) ** 2).mean().sqrt().item():6.1f}")
print(f"    baseline: all black (0)       {(truth ** 2).mean().sqrt().item():6.1f}")
print(f"  log p(true bottom half | top half), mean: {(probs.gather(-1, x[:, hidden][..., None]).log().sum(dim = (1, 2))).mean().item():.1f}")

# Conditional SAMPLES: the forward pass on the evidence is cached, then the top-down pass draws the
# circuit's latent choices from their posterior. Every input node then emits a value, so the observed
# pixels are copied back in; the hidden ones are then exact draws from p(hidden | observed).
t0 = time.time()
cond_samples = []
for _ in range(2):
    juice.queries.marginal(pc, data = x, missing_mask = hidden)
    s = juice.queries.sample(pc, conditional = True)
    s[:, ~hidden] = x[:, ~hidden]
    cond_samples.append(s)
print(f"  drew 2 conditional samples per digit ({time.time() - t0:.1f}s)")

masked = x.clone().float(); masked[:, hidden] = 128
show_grid([x, masked, fill(x, cond_mean), fill(x, cond_mode), *cond_samples],
          ["original", "observed (top)", "cond. mean", "cond. mode", "sample 1", "sample 2"],
          "3_inpainting.png")
std_img = torch.zeros(M, D); std_img[:, hidden] = cond_std
show_grid([std_img], ["uncertainty\n(std of p(x_i|top))"], "3_inpainting_uncertainty.png", cmap = "magma", vmax = float(cond_std.max()))


# %%
# 4. Sampling: new digits
# -----------------------

section("4. Sampling -- draw new digits from p(x)")
t0 = time.time()
samples = juice.queries.sample(pc, num_samples = 32)
print(f"  sample(pc, num_samples = 32): {tuple(samples.shape)}, values {samples.min().item()}..{samples.max().item()} "
      f"({time.time() - t0:.1f}s)")
ll_samples = log_likelihood(samples)
print(f"  their own bits/pixel under the PC: {bpd(ll_samples.mean().item()):.3f} "
      f"(held-out digits: {bpd(ll_digits.mean().item()):.3f})")
show_grid([samples[:16], samples[16:]], ["samples", "samples"], "4_samples.png")

print(f"\nDone. Figures are in {os.path.abspath(args.out_dir)}")
