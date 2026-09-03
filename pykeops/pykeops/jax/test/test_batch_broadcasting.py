#!/usr/bin/env python3
"""
KeOps JAX Batch Broadcasting Tests
==================================
A batch axis of size one must be reused for every sample, exactly as NumPy and
`pykeops.torch` do. A side of shape (1, M, D) used against a side of shape
(B, N, D) has to produce B results, not one.

Before the fix, the JAX binder took the batch size from the first argument
alone and its ranges launcher gave every "i" variable a batch stride of nx and
every "j" variable a stride of ny. So (1, M, D) against (B, N, D) returned a
single sample in the shape (1, M, 1), and (B, M, D) against (1, N, D) returned
the right shape with out-of-bounds reads on every row past the first.

Three neighbouring shapes are also checked here, because the same assumption
made each of them fail silently: batch sizes that are genuinely incompatible,
a parameter that carries a batch axis, and a 3D argument mixed with a 2D one.
Each must raise rather than compute something wrong.

PyTorch KeOps is the ground truth throughout.
"""

import sys
import numpy as np

from test_utils import (
    TestSuite, print_header, print_subheader, print_warning,
    compare_arrays, run_test, print_environment_info,
    setup_jax_float64, get_np_dtype, get_dtype_str,
)

# Setup float64 mode BEFORE importing JAX
setup_jax_float64()

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError as e:
    print(f"Error: JAX not found: {e}")
    JAX_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from pykeops.jax import Genred as JaxGenred, LazyTensor as JaxLazyTensor
    from pykeops.torch import Genred as TorchGenred, LazyTensor as TorchLazyTensor
    KEOPS_AVAILABLE = True
except ImportError as e:
    print(f"Error: KeOps not found: {e}")
    KEOPS_AVAILABLE = False


SEED = 42
RTOL = 1e-4
ATOL = 1e-5

FORMULA = "Exp(-SqDist(x,y))"
ALIASES = ["x=Vi(3)", "y=Vj(3)"]


def make_pair(bx, by, m=5, n=7, d=3, seed=SEED):
    """Random (bx, m, d) and (by, n, d) arrays."""
    rng = np.random.default_rng(seed)
    dtype = get_np_dtype()
    return (rng.random((bx, m, d)).astype(dtype),
            rng.random((by, n, d)).astype(dtype))


def torch_genred(x, y, formula=FORMULA, aliases=ALIASES):
    op = TorchGenred(formula, aliases, reduction_op="Sum", axis=1)
    out = op(torch.as_tensor(x).cuda(), torch.as_tensor(y).cuda())
    return out.cpu().numpy()


def jax_genred(x, y, formula=FORMULA, aliases=ALIASES):
    op = JaxGenred(formula, aliases, reduction_op="Sum", axis=1,
                   dtype=get_dtype_str())
    return np.asarray(op(jnp.asarray(x), jnp.asarray(y)))


def fail(message):
    """run_test formats the second element as a float, so report detail here."""
    print_warning(message)
    return False, float("inf")


def expect_raises(fn, exc_types):
    """Return (passed, 0.0) when fn raises one of exc_types."""
    try:
        fn()
    except exc_types as e:
        return True, 0.0
    except Exception as e:
        return fail(f"raised {type(e).__name__} instead: {str(e)[:60]}")
    return fail("no exception raised")


# =============================================================================
# 1. Forward pass
# =============================================================================

def test_broadcast_i_side(batch=3):
    """(1, M, D) against (B, N, D) must give B results."""
    x, y = make_pair(1, batch)
    got = jax_genred(x, y)
    ref = torch_genred(x, y)
    if got.shape != ref.shape:
        return fail(f"shape {got.shape}, expected {ref.shape}")
    return compare_arrays(got, ref, rtol=RTOL, atol=ATOL, squeeze=False)


def test_broadcast_j_side(batch=3):
    """(B, M, D) against (1, N, D) must reuse the single j sample."""
    x, y = make_pair(batch, 1)
    got = jax_genred(x, y)
    ref = torch_genred(x, y)
    if got.shape != ref.shape:
        return fail(f"shape {got.shape}, expected {ref.shape}")
    return compare_arrays(got, ref, rtol=RTOL, atol=ATOL, squeeze=False)


def test_broadcast_large_batch():
    """Broadcasting across more samples than one CUDA block covers."""
    x, y = make_pair(1, 8, m=200, n=193)
    got = jax_genred(x, y)
    ref = torch_genred(x, y)
    if got.shape != ref.shape:
        return fail(f"shape {got.shape}, expected {ref.shape}")
    return compare_arrays(got, ref, rtol=RTOL, atol=ATOL, squeeze=False)


def test_equal_batch_still_works(batch=3):
    """Regression guard: equal batch sizes were already correct."""
    x, y = make_pair(batch, batch)
    got = jax_genred(x, y)
    ref = torch_genred(x, y)
    if got.shape != ref.shape:
        return fail(f"shape {got.shape}, expected {ref.shape}")
    return compare_arrays(got, ref, rtol=RTOL, atol=ATOL, squeeze=False)


def test_lazytensor_broadcast(batch=3):
    """The LazyTensor front end broadcasts the same way."""
    x, y = make_pair(1, batch)

    x_i = JaxLazyTensor(jnp.asarray(x)[:, :, None, :])
    y_j = JaxLazyTensor(jnp.asarray(y)[:, None, :, :])
    got = np.asarray((-((x_i - y_j) ** 2).sum(-1)).exp().sum(2))

    x_t = TorchLazyTensor(torch.as_tensor(x).cuda()[:, :, None, :])
    y_t = TorchLazyTensor(torch.as_tensor(y).cuda()[:, None, :, :])
    ref = (-((x_t - y_t) ** 2).sum(-1)).exp().sum(2).cpu().numpy()

    if got.shape != ref.shape:
        return fail(f"shape {got.shape}, expected {ref.shape}")
    return compare_arrays(got, ref, rtol=RTOL, atol=ATOL, squeeze=False)


# =============================================================================
# 2. Gradients through a broadcast argument
# =============================================================================

def test_gradient_wrt_broadcast_i(batch=3):
    """d/dx with x of shape (1, M, D) sums the cotangent over the batch."""
    x, y = make_pair(1, batch)

    def loss(x_in):
        op = JaxGenred(FORMULA, ALIASES, reduction_op="Sum", axis=1,
                       dtype=get_dtype_str())
        return jnp.sum(op(x_in, jnp.asarray(y)))

    got = np.asarray(jax.grad(loss)(jnp.asarray(x)))

    x_t = torch.as_tensor(x).cuda().requires_grad_(True)
    op_t = TorchGenred(FORMULA, ALIASES, reduction_op="Sum", axis=1)
    op_t(x_t, torch.as_tensor(y).cuda()).sum().backward()
    ref = x_t.grad.cpu().numpy()

    if got.shape != ref.shape:
        return fail(f"grad shape {got.shape}, expected {ref.shape}")
    return compare_arrays(got, ref, rtol=RTOL, atol=ATOL, squeeze=False)


def test_gradient_wrt_broadcast_j(batch=3):
    """d/dy with y of shape (1, N, D)."""
    x, y = make_pair(batch, 1)

    def loss(y_in):
        op = JaxGenred(FORMULA, ALIASES, reduction_op="Sum", axis=1,
                       dtype=get_dtype_str())
        return jnp.sum(op(jnp.asarray(x), y_in))

    got = np.asarray(jax.grad(loss)(jnp.asarray(y)))

    y_t = torch.as_tensor(y).cuda().requires_grad_(True)
    op_t = TorchGenred(FORMULA, ALIASES, reduction_op="Sum", axis=1)
    op_t(torch.as_tensor(x).cuda(), y_t).sum().backward()
    ref = y_t.grad.cpu().numpy()

    if got.shape != ref.shape:
        return fail(f"grad shape {got.shape}, expected {ref.shape}")
    return compare_arrays(got, ref, rtol=RTOL, atol=ATOL, squeeze=False)


# =============================================================================
# 3. Shapes that must raise instead of computing something wrong
# =============================================================================

def test_incompatible_batch_raises():
    """(2, M, D) against (3, N, D) is not broadcastable."""
    x, y = make_pair(2, 3)
    return expect_raises(lambda: jax_genred(x, y), ValueError)


def test_mixed_rank_raises():
    """A 3D argument mixed with a 2D one needs explicit unit dimensions."""
    x, y = make_pair(3, 1)
    return expect_raises(lambda: jax_genred(x, y[0]), ValueError)


def test_batched_parameter_raises():
    """A Pm carrying a batch axis is not supported by the ranges launcher."""
    x, y = make_pair(3, 3)
    s = np.array([0.25, 1.0, 4.0], dtype=get_np_dtype()).reshape(3, 1, 1)
    formula = "Exp(-SqDist(x,y) * s)"
    aliases = ["x=Vi(3)", "y=Vj(3)", "s=Pm(1)"]

    def call():
        op = JaxGenred(formula, aliases, reduction_op="Sum", axis=1,
                       dtype=get_dtype_str())
        return op(jnp.asarray(x), jnp.asarray(y), jnp.asarray(s))

    return expect_raises(call, (ValueError, NotImplementedError))


def test_shared_parameter_still_works():
    """A Pm with a unit batch axis is broadcast, as before."""
    x, y = make_pair(3, 3)
    s = np.array([0.5], dtype=get_np_dtype()).reshape(1, 1, 1)
    formula = "Exp(-SqDist(x,y) * s)"
    aliases = ["x=Vi(3)", "y=Vj(3)", "s=Pm(1)"]

    op = JaxGenred(formula, aliases, reduction_op="Sum", axis=1,
                   dtype=get_dtype_str())
    got = np.asarray(op(jnp.asarray(x), jnp.asarray(y), jnp.asarray(s)))

    op_t = TorchGenred(formula, aliases, reduction_op="Sum", axis=1)
    ref = op_t(torch.as_tensor(x).cuda(), torch.as_tensor(y).cuda(),
               torch.as_tensor(s).cuda()).cpu().numpy()

    if got.shape != ref.shape:
        return fail(f"shape {got.shape}, expected {ref.shape}")
    return compare_arrays(got, ref, rtol=RTOL, atol=ATOL, squeeze=False)


# =============================================================================

def main():
    print_header("Batch Broadcasting Tests",
                 "A batch axis of size one must be reused for every sample")
    print_environment_info()

    if not JAX_AVAILABLE:
        print_warning("JAX not available, skipping tests")
        return 1
    if not TORCH_AVAILABLE:
        print_warning("PyTorch CUDA not available, skipping tests")
        return 1
    if not KEOPS_AVAILABLE:
        print_warning("KeOps not available, skipping tests")
        return 1

    suite = TestSuite("Batch Broadcasting", "Size-one batch axes vs PyTorch")

    print_subheader("1. Forward pass")
    run_test("(1,M) vs (B,N)", test_broadcast_i_side, suite)
    run_test("(B,M) vs (1,N)", test_broadcast_j_side, suite)
    run_test("(1,M) vs (8,N), M=200 N=193", test_broadcast_large_batch, suite)
    run_test("(B,M) vs (B,N) unchanged", test_equal_batch_still_works, suite)
    run_test("LazyTensor (1,M) vs (B,N)", test_lazytensor_broadcast, suite)

    print_subheader("2. Gradients through a broadcast argument")
    run_test("grad wrt (1,M,D) source", test_gradient_wrt_broadcast_i, suite)
    run_test("grad wrt (1,N,D) target", test_gradient_wrt_broadcast_j, suite)

    print_subheader("3. Shapes that must raise")
    run_test("(2,M) vs (3,N) raises", test_incompatible_batch_raises, suite)
    run_test("3D vs 2D raises", test_mixed_rank_raises, suite)
    run_test("batched Pm raises", test_batched_parameter_raises, suite)
    run_test("shared Pm still works", test_shared_parameter_still_works, suite)

    suite.print_summary()
    return 0 if suite.all_passed() else 1


if __name__ == "__main__":
    sys.exit(main())
