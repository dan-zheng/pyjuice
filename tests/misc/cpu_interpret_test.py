"""
The CPU path through Triton's interpreter (`TRITON_INTERPRET=1`): compile, forward, backward and EM
on a small HCLT, checked against exact facts -- the circuit's probabilities sum to 1 over every
assignment (in both precisions, and at batch sizes on both sides of the bf16 threshold), and
full-batch EM never decreases the data log-likelihood; and the marginal, conditional and
sampling queries agree with brute force over all assignments.

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

    # Queries, against brute force over all 243 assignments
    with torch.no_grad():
        p = pc(allx)[:, 0].exp()
    code = lambda x: (x * (C ** torch.arange(V - 1, -1, -1))).sum(-1)

    # conditional: p(x_3, x_4 | x_0, x_1, x_2)
    hidden = torch.tensor([False, False, False, True, True])
    ev = allx[torch.tensor([0, 50, 120, 200])]
    cond = juice.queries.conditional(pc, data = ev, missing_mask = hidden, target_vars = [3, 4])
    for r in range(ev.size(0)):
        match = (allx[:, :3] == ev[r, :3]).all(-1)
        pj, xs = p[match], allx[match]
        for j, v in enumerate([3, 4]):
            brute = torch.stack([pj[xs[:, v] == c].sum() for c in range(C)]) / pj.sum()
            assert torch.allclose(cond[r, j], brute, atol = 1e-4), (r, v, cond[r, j], brute)

    # marginal: one variable summed out == logsumexp over its values
    m = juice.queries.marginal(pc, data = allx[:1], missing_mask = torch.tensor([False, False, True, False, False]))
    rows = allx[:1].repeat(C, 1); rows[:, 2] = torch.arange(C)
    assert abs(m.item() - torch.logsumexp(pc(rows)[:, 0].detach(), 0).item()) < 1e-4

    # unconditional sampling: empirical distribution close to p (total variation)
    n = 8000
    s = juice.queries.sample(pc, num_samples = n)
    emp = torch.bincount(code(s), minlength = allx.size(0)).float() / n
    assert 0.5 * (emp - p).abs().sum().item() < 0.08

    # conditional sampling: hidden pixels drawn from p(x_3, x_4 | evidence)
    e = allx[50:51].repeat(n, 1)
    juice.queries.marginal(pc, data = e, missing_mask = hidden)
    s = juice.queries.sample(pc, conditional = True)
    match = (allx[:, :3] == e[0, :3]).all(-1)
    exact = torch.zeros(C * C); exact[allx[match][:, 3] * C + allx[match][:, 4]] = p[match] / p[match].sum()
    emp = torch.bincount(s[:, 3] * C + s[:, 4], minlength = C * C).float() / n
    assert 0.5 * (emp - exact).abs().sum().item() < 0.04
    print("OK")
""")


@pytest.mark.slow
def test_cpu_interpret_hclt():
    env = dict(os.environ, TRITON_INTERPRET = "1", CUDA_VISIBLE_DEVICES = "")
    res = subprocess.run([sys.executable, "-c", _SCRIPT], env = env, capture_output = True, text = True, timeout = 1200)
    assert res.returncode == 0 and "OK" in res.stdout, res.stdout[-3000:] + res.stderr[-3000:]


if __name__ == "__main__":
    test_cpu_interpret_hclt()
