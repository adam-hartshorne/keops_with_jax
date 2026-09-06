#!/usr/bin/env python3
"""KernelSolve on the JAX backend.

Covers the conjugate-gradient solve of (alpha I + K) a = b: its residual, agreement with
pykeops.torch, behaviour across regularizations and kernels, and a kernel ridge fit.

Nothing here runs at import. Until 2026-09-06 the whole file was module-level code, so pytest
executed it while collecting and then collected no tests from it, and every check printed its
verdict without ever failing the run.
"""

import sys

import numpy as np
import jax.numpy as jnp
import pytest

from pykeops.jax import Genred, KernelSolve

# Every test here needs a GPU and the torch comparison needs PyTorch with CUDA. conftest.py
# registers these markers and skips on missing hardware, so declaring them at module level is what
# makes `pytest -m pytorch` and `pytest -m gpu` select anything.
pytestmark = [pytest.mark.gpu, pytest.mark.pytorch]

N, D = 100, 3
EPS = 1e-6
FORMULA = "Exp(-SqDist(x,y)) * a"
ALIASES = ["x=Vi(3)", "y=Vj(3)", "a=Vj(3)"]

# A conjugate-gradient residual only means something against the scale of the right-hand side, and
# the conditioning of (alpha I + K) grows as alpha shrinks, so the check is relative. Measured
# 2026-09-06 across the four alphas below: 2.1e-5 at alpha=0.01, then 3.8e-6, 1.0e-6, 6.3e-7. The
# absolute 1e-4 this file used before put alpha=0.01 over the line (3.5e-4 against ||b|| = 16.6)
# and printed PASSED regardless, because the summary line was unconditional.
RTOL_RESIDUAL = 1e-4

# JAX and torch run the same kernel but reduce with different libraries, so agreement is checked
# at single-precision scale rather than bitwise.
ATOL_VS_TORCH = 1e-4


@pytest.fixture(scope="module")
def data():
    """The x and b of (alpha I + K) a = b, shared by every test in the file."""
    np.random.seed(42)
    x = np.random.randn(N, D).astype(np.float32)
    b = np.random.randn(N, D).astype(np.float32)
    return jnp.array(x), jnp.array(b), x, b


@pytest.fixture(scope="module")
def solver():
    return KernelSolve(FORMULA, ALIASES, "a", axis=1)


@pytest.fixture(scope="module")
def kernel_op():
    return Genred(FORMULA, ALIASES, reduction_op="Sum", axis=1)


def relative_residual(kernel_op, x, a, b, alpha):
    """||(alpha I + K) a - b|| / ||b||."""
    Ka = kernel_op(x, x, a)
    return float(jnp.linalg.norm(Ka + alpha * a - b) / jnp.linalg.norm(b))


def test_basic_solve(data, solver, kernel_op):
    """A Gaussian kernel solve returns the right shape and a small residual."""
    x_jax, b_jax, _, _ = data
    alpha = 0.1

    a_star = solver(x_jax, x_jax, b_jax, alpha=alpha, eps=EPS)

    assert a_star.shape == (N, D)
    residual = relative_residual(kernel_op, x_jax, a_star, b_jax, alpha)
    print(f"  relative residual at alpha={alpha}: {residual:.2e}")
    assert residual < RTOL_RESIDUAL


def test_matches_torch(data):
    """The JAX solve agrees with pykeops.torch on the same system."""
    import torch
    from pykeops.torch import KernelSolve as KernelSolve_torch

    x_jax, b_jax, x_np, b_np = data
    alpha = 0.1

    solver_jax = KernelSolve(FORMULA, ALIASES, "a", axis=1)
    a_star_jax = np.array(solver_jax(x_jax, x_jax, b_jax, alpha=alpha, eps=EPS))

    solver_torch = KernelSolve_torch(FORMULA, ALIASES, "a", axis=1)
    x_torch = torch.tensor(x_np, device="cuda")
    b_torch = torch.tensor(b_np, device="cuda")
    a_star_torch = solver_torch(x_torch, x_torch, b_torch, alpha=alpha, eps=EPS)

    max_diff = np.abs(a_star_jax - a_star_torch.cpu().numpy()).max()
    print(f"  max |jax - torch|: {max_diff:.2e}")
    assert max_diff < ATOL_VS_TORCH


@pytest.mark.parametrize("alpha", [0.01, 0.1, 1.0, 10.0])
def test_regularization(data, solver, kernel_op, alpha):
    """The residual stays small across four orders of magnitude of regularization."""
    x_jax, b_jax, _, _ = data

    a_star = solver(x_jax, x_jax, b_jax, alpha=alpha, eps=EPS)

    residual = relative_residual(kernel_op, x_jax, a_star, b_jax, alpha)
    print(f"  alpha={alpha:5.2f}: relative residual={residual:.2e}")
    assert residual < RTOL_RESIDUAL


def test_laplacian_kernel(data):
    """A kernel other than the Gaussian solves to the same accuracy."""
    x_jax, b_jax, _, _ = data
    alpha = 0.1
    formula = "Exp(-Sqrt(SqDist(x,y)+IntCst(1e-6))) * a"
    aliases = ["x=Vi(3)", "y=Vj(3)", "a=Vj(3)"]

    solver_lap = KernelSolve(formula, aliases, "a", axis=1)
    a_star = solver_lap(x_jax, x_jax, b_jax, alpha=alpha, eps=EPS)

    kernel_op = Genred(formula, aliases, reduction_op="Sum", axis=1)
    residual = relative_residual(kernel_op, x_jax, a_star, b_jax, alpha)
    print(f"  Laplacian relative residual: {residual:.2e}")
    assert residual < RTOL_RESIDUAL


def test_kernel_ridge_regression():
    """The solve is usable as the fitting step of a kernel ridge regression."""
    np.random.seed(123)
    n_train = 200
    x_train = np.random.randn(n_train, 1).astype(np.float32)
    y_train = (np.sin(3 * x_train) + 0.1 * np.random.randn(n_train, 1)).astype(
        np.float32
    )

    sigma = 0.5
    oos2 = jnp.array(np.array([1.0 / (2 * sigma**2)], dtype=np.float32))
    x_jax = jnp.array(x_train)
    y_jax = jnp.array(y_train)

    formula = "Exp(-SqDist(x,y) * oos2) * a"
    aliases = ["x=Vi(1)", "y=Vj(1)", "a=Vj(1)", "oos2=Pm(1)"]

    coeffs = KernelSolve(formula, aliases, "a", axis=1)(
        x_jax, x_jax, y_jax, oos2, alpha=0.01
    )
    y_pred = Genred(formula, aliases, reduction_op="Sum", axis=1)(
        x_jax, x_jax, coeffs, oos2
    )

    mse = float(jnp.mean((y_pred - y_jax) ** 2))
    print(f"  training MSE: {mse:.6f}")
    assert mse < 0.1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
