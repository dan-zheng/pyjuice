"""
Stand-ins for the `libdevice` math functions PyJuice's Triton kernels use, for Triton's CPU
interpreter (`TRITON_INTERPRET=1`).

The interpreter cannot execute `extern_elementwise` calls, so every kernel that touches
`tl.extra.cuda.libdevice` (PyJuice's `tlmath`) fails there. :func:`install` replaces that module
with this one -- only when the interpreter is enabled -- before PyJuice's kernels are imported.
"""
import triton.language as tl


# Plain functions, not `@triton.jit`: inside an interpreted kernel `tl` is already patched to the
# interpreter's ops, and a nested jit call would re-patch it on every call (~1 ms each).
def log1p(x):
    # Taylor branch keeps precision near 0, where `log(1 + x)` rounds `1 + x` to 1
    return tl.where(tl.abs(x) < 1e-4, x - 0.5 * x * x, tl.log(1.0 + x))


def pow(x, y):
    # Kernels only raise non-negative values (probabilities/flows); x = 0 gives exp(-inf) = 0 for y > 0
    return tl.exp(y * tl.log(x))


exp = tl.exp
ceil = tl.math.ceil


def install():
    import types
    from pyjuice.utils.util import triton_interpret_enabled
    if not triton_interpret_enabled():
        return
    shim = types.ModuleType("pyjuice_interpret_libdevice")
    for name in ("log1p", "pow", "exp", "ceil"):
        setattr(shim, name, globals()[name])
    import triton.language.extra.cuda as tl_cuda
    tl_cuda.libdevice = shim
