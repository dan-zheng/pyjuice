"""
The CPU path through Triton's interpreter (`TRITON_INTERPRET=1`): compile, forward, backward and EM
on a small HCLT, checked against exact facts -- the circuit's probabilities sum to 1 over every
assignment (in both precisions, and at batch sizes on both sides of the bf16 threshold), and
full-batch EM never decreases the data log-likelihood.

The interpreter has to be enabled before `triton`/`pyjuice` are imported, so the checks run in a
subprocess.
"""

import os
import subprocess
import sys
import textwrap

import pytest


_SCRIPT = textwrap.dedent("""
    import itertools, warnings, torch
    import pyjuice as juice
    warnings.filterwarnings("ignore", category = RuntimeWarning)
    torch.manual_seed(0)

    V, C = 5, 3
    data = torch.randint(0, C, (200, V))
    data[:, 1] = data[:, 0]
    allx = torch.tensor(list(itertools.product(range(C), repeat = V)))

    # num_latents = 16 gives block size 16: the size at which "auto" would pick bf16 products
    pc = juice.compile(juice.structures.HCLT(data.float(), num_latents = 16, input_node_params = {"num_cats": C}),
                       verbose = False)
    assert pc.device.type == "cpu"

    def total_prob(B):
        with torch.no_grad():
            return sum(pc(allx[i:i+B]).exp().sum().item() for i in range(0, len(allx), B))

    for precision in ("auto", "fp32"):
        pc.precision = precision
        for B in (8, 64):
            s = total_prob(B)
            assert abs(s - 1.0) < 1e-4, (precision, B, s)

    def data_ll():
        with torch.no_grad():
            return pc(data).mean().item()

    prev = data_ll()
    for _ in range(3):
        pc.init_param_flows(flows_memory = 0.0)
        for s in range(0, data.size(0), 100):
            pc(data[s:s+100]).mean().backward()
        pc.mini_batch_em(step_size = 1.0, pseudocount = 0.0)
        curr = data_ll()
        assert curr >= prev - 1e-4, (prev, curr)
        prev = curr
        assert abs(total_prob(243) - 1.0) < 1e-4

    opt = juice.optim.MiniBatchEM(pc, step_size = 0.1, pseudocount = 0.1)
    pc(data[:64]).mean().backward()
    opt.step()
    assert abs(total_prob(243) - 1.0) < 1e-4
    print("OK")
""")


@pytest.mark.slow
def test_cpu_interpret_hclt():
    env = dict(os.environ, TRITON_INTERPRET = "1", CUDA_VISIBLE_DEVICES = "")
    res = subprocess.run([sys.executable, "-c", _SCRIPT], env = env, capture_output = True, text = True, timeout = 1200)
    assert res.returncode == 0 and "OK" in res.stdout, res.stdout[-3000:] + res.stderr[-3000:]


if __name__ == "__main__":
    test_cpu_interpret_hclt()
