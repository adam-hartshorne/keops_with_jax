#!/usr/bin/env python3
"""
KeOps JAX Edge Case Tests
=========================
Tests for edge cases discovered during development.

All tests compare JAX KeOps against PyTorch KeOps as the ground truth reference.
Tests cover multiple matrix sizes to catch size-dependent bugs.

These tests specifically target bugs that were found and fixed:
1. Scalar multiplication variable ordering (LazyTensor._var_ids fix)
2. Batched vs non-batched operations
3. Complex multi-variable varifold kernels with lengthscales
4. Pm parameter gradient handling
5. High-dimensional chunking
6. Sparse variable indices (gradient formula variable dropout)
7. Size-dependent bugs (small, medium, large matrices)
"""

import sys
import numpy as np

# =============================================================================
# Imports and Setup
# =============================================================================

import jax
import jax.numpy as jnp

from test_utils import (
    TestSuite, TestResult, Status,
    print_header, print_subheader, print_info, print_success, print_error,
    compare_arrays, run_test, print_environment_info
)

# Import JAX KeOps
try:
    from pykeops.jax import Genred, LazyTensor, Vi, Vj, Pm
    JAX_AVAILABLE = True
except ImportError as e:
    print(f"Error: pykeops.jax not found: {e}")
    JAX_AVAILABLE = False
    sys.exit(1)

# Import PyTorch KeOps for ground truth - REQUIRED for all tests
try:
    import torch
    from pykeops.torch import Genred as Genred_torch, LazyTensor as LazyTensor_torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

if not TORCH_AVAILABLE:
    print("Error: PyTorch KeOps with CUDA is required for these tests")
    print("All tests compare against PyTorch KeOps as ground truth")
    sys.exit(1)

# =============================================================================
# Configuration
# =============================================================================

SEED = 42
RTOL = 1e-4
ATOL = 1e-5

# Test sizes to catch size-dependent bugs
# Previously had issues with: 200x200 OK, slightly smaller broken, >500x500 broken
TEST_SIZES = [
    # (N, M) pairs - small, medium around problem areas, large
    (10, 10),
    (50, 40),
    (100, 80),
    (150, 120),
    (200, 200),
    (250, 200),
    (300, 250),
    (500, 400),
    (750, 600),
    (1000, 800),
]

BATCH_SIZES = [1, 2, 4]


# =============================================================================
# Test 1: Scalar Multiplication Variable Ordering (vs PyTorch)
# =============================================================================

def test_scalar_multiplication_sizes():
    """Test scalar multiplication across multiple sizes against PyTorch."""
    max_diff = 0.0
    all_passed = True

    for N, M in TEST_SIZES:
        np.random.seed(SEED)
        D = 3

        x_np = np.random.randn(N, D).astype(np.float32)
        y_np = np.random.randn(M, D).astype(np.float32)

        # JAX KeOps
        x_jax = jnp.array(x_np)
        y_jax = jnp.array(y_np)
        x_i = LazyTensor(x_jax[:, None, :])
        y_j = LazyTensor(y_jax[None, :, :])
        result_jax = (2.0 * (x_i - y_j)).sum(-1).sum(axis=1)

        # PyTorch KeOps (ground truth)
        x_torch = torch.tensor(x_np, device='cuda')
        y_torch = torch.tensor(y_np, device='cuda')
        x_i_t = LazyTensor_torch(x_torch[:, None, :])
        y_j_t = LazyTensor_torch(y_torch[None, :, :])
        result_torch = (2.0 * (x_i_t - y_j_t)).sum(-1).sum(axis=1)

        passed, diff = compare_arrays(
            np.squeeze(np.array(result_jax)),
            np.squeeze(result_torch.cpu().numpy()),
            rtol=RTOL, atol=ATOL
        )

        if not passed:
            print(f"    FAIL at size {N}x{M}: diff={diff:.2e}")
            all_passed = False
        max_diff = max(max_diff, diff)

    return all_passed, max_diff


def test_scalar_multiplication_batched():
    """Test batched scalar multiplication against PyTorch."""
    max_diff = 0.0
    all_passed = True

    for B in BATCH_SIZES:
        for N, M in [(50, 40), (200, 150), (500, 400)]:
            np.random.seed(SEED)
            D = 3

            x_np = np.random.randn(B, N, D).astype(np.float32)
            y_np = np.random.randn(B, M, D).astype(np.float32)

            # JAX KeOps
            x_jax = jnp.array(x_np)
            y_jax = jnp.array(y_np)
            x_i = LazyTensor(x_jax[:, :, None, :])
            y_j = LazyTensor(y_jax[:, None, :, :])
            result_jax = (2.0 * (x_i - y_j)).sum(-1).sum(axis=2)

            # PyTorch KeOps
            x_torch = torch.tensor(x_np, device='cuda')
            y_torch = torch.tensor(y_np, device='cuda')
            x_i_t = LazyTensor_torch(x_torch[:, :, None, :])
            y_j_t = LazyTensor_torch(y_torch[:, None, :, :])
            result_torch = (2.0 * (x_i_t - y_j_t)).sum(-1).sum(axis=2)

            passed, diff = compare_arrays(
                np.array(result_jax), result_torch.cpu().numpy(),
                rtol=RTOL, atol=ATOL, squeeze=True
            )

            if not passed:
                print(f"    FAIL at B={B}, size {N}x{M}: diff={diff:.2e}")
                all_passed = False
            max_diff = max(max_diff, diff)

    return all_passed, max_diff


# =============================================================================
# Test 2: Gaussian Kernel Forward Pass (vs PyTorch)
# =============================================================================

def test_gaussian_kernel_sizes():
    """Test Gaussian kernel across multiple sizes against PyTorch."""
    max_diff = 0.0
    all_passed = True

    for N, M in TEST_SIZES:
        np.random.seed(SEED)
        D = 3

        x_np = np.random.randn(N, D).astype(np.float32)
        y_np = np.random.randn(M, D).astype(np.float32)

        # JAX KeOps
        x_jax = jnp.array(x_np)
        y_jax = jnp.array(y_np)
        x_i = LazyTensor(x_jax[:, None, :])
        y_j = LazyTensor(y_jax[None, :, :])
        K = (-((x_i - y_j) ** 2).sum(-1)).exp()
        result_jax = K.sum(axis=1)

        # PyTorch KeOps
        x_torch = torch.tensor(x_np, device='cuda')
        y_torch = torch.tensor(y_np, device='cuda')
        x_i_t = LazyTensor_torch(x_torch[:, None, :])
        y_j_t = LazyTensor_torch(y_torch[None, :, :])
        K_t = (-((x_i_t - y_j_t) ** 2).sum(-1)).exp()
        result_torch = K_t.sum(axis=1)

        passed, diff = compare_arrays(
            np.array(result_jax), result_torch.cpu().numpy(),
            rtol=RTOL, atol=ATOL, squeeze=True
        )

        if not passed:
            print(f"    FAIL at size {N}x{M}: diff={diff:.2e}")
            all_passed = False
        max_diff = max(max_diff, diff)

    return all_passed, max_diff


def test_gaussian_kernel_gradient_sizes():
    """Test Gaussian kernel gradient across multiple sizes against PyTorch."""
    max_diff = 0.0
    all_passed = True

    for N, M in [(50, 40), (150, 120), (200, 200), (300, 250), (500, 400)]:
        np.random.seed(SEED)
        D = 3

        x_np = np.random.randn(N, D).astype(np.float32)
        y_np = np.random.randn(M, D).astype(np.float32)

        # JAX KeOps gradient
        x_jax = jnp.array(x_np)
        y_jax = jnp.array(y_np)

        def jax_forward(x_var):
            x_i = LazyTensor(x_var[:, None, :])
            y_j = LazyTensor(y_jax[None, :, :])
            K = (-((x_i - y_j) ** 2).sum(-1)).exp()
            return K.sum(axis=1).sum()

        grad_jax = jax.grad(jax_forward)(x_jax)

        # PyTorch KeOps gradient
        x_torch = torch.tensor(x_np, device='cuda', requires_grad=True)
        y_torch = torch.tensor(y_np, device='cuda')

        x_i_t = LazyTensor_torch(x_torch[:, None, :])
        y_j_t = LazyTensor_torch(y_torch[None, :, :])
        K_t = (-((x_i_t - y_j_t) ** 2).sum(-1)).exp()
        loss_t = K_t.sum(axis=1).sum()
        loss_t.backward()
        grad_torch = x_torch.grad

        passed, diff = compare_arrays(
            np.array(grad_jax), grad_torch.cpu().numpy(),
            rtol=1e-3, atol=1e-4
        )

        if not passed:
            print(f"    FAIL at size {N}x{M}: diff={diff:.2e}")
            all_passed = False
        max_diff = max(max_diff, diff)

    return all_passed, max_diff


# =============================================================================
# Test 3: Kernel-Vector Product (K @ v) and Gradient (vs PyTorch)
# =============================================================================

def test_kernel_matmul_sizes():
    """Test kernel @ vector across multiple sizes against PyTorch."""
    max_diff = 0.0
    all_passed = True

    for N, M in TEST_SIZES:
        np.random.seed(SEED)
        D = 3

        x_np = np.random.randn(N, D).astype(np.float32)
        y_np = np.random.randn(M, D).astype(np.float32)
        v_np = np.random.randn(M, 1).astype(np.float32)

        # JAX KeOps
        x_jax = jnp.array(x_np)
        y_jax = jnp.array(y_np)
        v_jax = jnp.array(v_np)

        x_i = LazyTensor(x_jax[:, None, :])
        y_j = LazyTensor(y_jax[None, :, :])
        K = (-x_i.sqdist(y_j)).exp()
        result_jax = K @ v_jax

        # PyTorch KeOps
        x_torch = torch.tensor(x_np, device='cuda')
        y_torch = torch.tensor(y_np, device='cuda')
        v_torch = torch.tensor(v_np, device='cuda')

        x_i_t = LazyTensor_torch(x_torch[:, None, :])
        y_j_t = LazyTensor_torch(y_torch[None, :, :])
        K_t = (-x_i_t.sqdist(y_j_t)).exp()
        result_torch = K_t @ v_torch

        passed, diff = compare_arrays(
            np.array(result_jax), result_torch.cpu().numpy(),
            rtol=RTOL, atol=ATOL
        )

        if not passed:
            print(f"    FAIL at size {N}x{M}: diff={diff:.2e}")
            all_passed = False
        max_diff = max(max_diff, diff)

    return all_passed, max_diff


def test_kernel_matmul_gradient_sizes():
    """Test gradient of K @ v w.r.t. v across multiple sizes against PyTorch.

    This is critical for the sparse variable bug - d(K@v)/dv = K^T, where v drops out.
    """
    max_diff = 0.0
    all_passed = True

    for N, M in [(50, 40), (150, 120), (200, 200), (300, 250), (500, 400)]:
        np.random.seed(SEED)
        D = 3

        x_np = np.random.randn(N, D).astype(np.float32) * 0.1
        y_np = np.random.randn(M, D).astype(np.float32) * 0.1
        v_np = np.abs(np.random.randn(M, 1).astype(np.float32)) + 0.1

        # JAX KeOps gradient
        x_jax = jnp.array(x_np)
        y_jax = jnp.array(y_np)
        v_jax = jnp.array(v_np)

        def jax_forward(v_var):
            x_i = LazyTensor(x_jax[:, None, :])
            y_j = LazyTensor(y_jax[None, :, :])
            K = (-x_i.sqdist(y_j)).exp()
            return (K @ v_var).sum()

        grad_jax = jax.grad(jax_forward)(v_jax)

        # PyTorch KeOps gradient
        x_torch = torch.tensor(x_np, device='cuda')
        y_torch = torch.tensor(y_np, device='cuda')
        v_torch = torch.tensor(v_np, device='cuda', requires_grad=True)

        x_i_t = LazyTensor_torch(x_torch[:, None, :])
        y_j_t = LazyTensor_torch(y_torch[None, :, :])
        K_t = (-x_i_t.sqdist(y_j_t)).exp()
        loss_t = (K_t @ v_torch).sum()
        loss_t.backward()
        grad_torch = v_torch.grad

        passed, diff = compare_arrays(
            np.array(grad_jax), grad_torch.cpu().numpy(),
            rtol=1e-3, atol=1e-4
        )

        if not passed:
            print(f"    FAIL at size {N}x{M}: diff={diff:.2e}")
            all_passed = False
        max_diff = max(max_diff, diff)

    return all_passed, max_diff


# =============================================================================
# Test 4: Varifold Kernel (Complex Multi-Variable) vs PyTorch
# =============================================================================

def test_varifold_forward():
    """Test varifold kernel forward pass against PyTorch."""
    max_diff = 0.0
    all_passed = True

    for B in [1, 2]:
        for n_i, n_j in [(10, 8), (50, 40), (200, 150)]:
            np.random.seed(SEED)

            x_np = np.random.randn(B, n_i, 3).astype(np.float32)
            y_np = np.random.randn(B, n_j, 3).astype(np.float32)
            nx_np = np.random.randn(B, n_i, 3).astype(np.float32)
            ny_np = np.random.randn(B, n_j, 3).astype(np.float32)
            wi_np = np.random.randn(B, n_i, 1).astype(np.float32)
            wj_np = np.random.randn(B, n_j, 1).astype(np.float32)

            nx_np = nx_np / (np.linalg.norm(nx_np, axis=-1, keepdims=True) + 1e-8)
            ny_np = ny_np / (np.linalg.norm(ny_np, axis=-1, keepdims=True) + 1e-8)

            sigma_pos, sigma_norm = 1.5, 0.8

            # JAX KeOps
            x, y = jnp.array(x_np), jnp.array(y_np)
            nx, ny = jnp.array(nx_np), jnp.array(ny_np)
            wi, wj = jnp.array(wi_np), jnp.array(wj_np)

            x_i = LazyTensor(x.reshape(B, n_i, 1, 3))
            y_j = LazyTensor(y.reshape(B, 1, n_j, 3))
            nx_i = LazyTensor(nx.reshape(B, n_i, 1, 3))
            ny_j = LazyTensor(ny.reshape(B, 1, n_j, 3))
            wi_i = LazyTensor(wi.reshape(B, n_i, 1, 1))
            wj_j = LazyTensor(wj.reshape(B, 1, n_j, 1))

            sq_dist_pos = ((x_i - y_j) ** 2).sum(-1)
            sq_dist_norm = ((nx_i - ny_j) ** 2).sum(-1)
            K_ij = wi_i * wj_j * (-(sq_dist_pos / (sigma_pos ** 2))).exp() * (-(sq_dist_norm / (sigma_norm ** 2))).exp()
            result_jax = K_ij.sum(dim=2)

            # PyTorch KeOps
            x_t = torch.tensor(x_np, device='cuda')
            y_t = torch.tensor(y_np, device='cuda')
            nx_t = torch.tensor(nx_np, device='cuda')
            ny_t = torch.tensor(ny_np, device='cuda')
            wi_t = torch.tensor(wi_np, device='cuda')
            wj_t = torch.tensor(wj_np, device='cuda')

            x_i_t = LazyTensor_torch(x_t.view(B, n_i, 1, 3))
            y_j_t = LazyTensor_torch(y_t.view(B, 1, n_j, 3))
            nx_i_t = LazyTensor_torch(nx_t.view(B, n_i, 1, 3))
            ny_j_t = LazyTensor_torch(ny_t.view(B, 1, n_j, 3))
            wi_i_t = LazyTensor_torch(wi_t.view(B, n_i, 1, 1))
            wj_j_t = LazyTensor_torch(wj_t.view(B, 1, n_j, 1))

            sq_dist_pos_t = ((x_i_t - y_j_t) ** 2).sum(-1)
            sq_dist_norm_t = ((nx_i_t - ny_j_t) ** 2).sum(-1)
            K_ij_t = wi_i_t * wj_j_t * (-(sq_dist_pos_t / (sigma_pos ** 2))).exp() * (-(sq_dist_norm_t / (sigma_norm ** 2))).exp()
            result_torch = K_ij_t.sum(dim=2)

            passed, diff = compare_arrays(
                np.array(result_jax), result_torch.cpu().numpy(),
                rtol=RTOL, atol=ATOL, squeeze=True
            )

            if not passed:
                print(f"    FAIL at B={B}, size {n_i}x{n_j}: diff={diff:.2e}")
                all_passed = False
            max_diff = max(max_diff, diff)

    return all_passed, max_diff


def test_varifold_gradient():
    """Test varifold kernel gradient against PyTorch."""
    max_diff = 0.0
    all_passed = True

    for B in [1, 2]:
        for n_i, n_j in [(10, 8), (50, 40), (150, 120)]:
            np.random.seed(SEED)

            x_np = np.random.randn(B, n_i, 3).astype(np.float32)
            y_np = np.random.randn(B, n_j, 3).astype(np.float32)
            nx_np = np.random.randn(B, n_i, 3).astype(np.float32)
            ny_np = np.random.randn(B, n_j, 3).astype(np.float32)
            wi_np = np.random.randn(B, n_i, 1).astype(np.float32)
            wj_np = np.random.randn(B, n_j, 1).astype(np.float32)

            nx_np = nx_np / (np.linalg.norm(nx_np, axis=-1, keepdims=True) + 1e-8)
            ny_np = ny_np / (np.linalg.norm(ny_np, axis=-1, keepdims=True) + 1e-8)

            sigma_pos, sigma_norm = 1.5, 0.8

            # JAX KeOps gradient
            x, y = jnp.array(x_np), jnp.array(y_np)
            nx, ny = jnp.array(nx_np), jnp.array(ny_np)
            wi, wj = jnp.array(wi_np), jnp.array(wj_np)

            def jax_forward(x_var):
                x_i = LazyTensor(x_var.reshape(B, n_i, 1, 3))
                y_j = LazyTensor(y.reshape(B, 1, n_j, 3))
                nx_i = LazyTensor(nx.reshape(B, n_i, 1, 3))
                ny_j = LazyTensor(ny.reshape(B, 1, n_j, 3))
                wi_i = LazyTensor(wi.reshape(B, n_i, 1, 1))
                wj_j = LazyTensor(wj.reshape(B, 1, n_j, 1))

                sq_dist_pos = ((x_i - y_j) ** 2).sum(-1)
                sq_dist_norm = ((nx_i - ny_j) ** 2).sum(-1)
                K_ij = wi_i * wj_j * (-(sq_dist_pos / (sigma_pos ** 2))).exp() * (-(sq_dist_norm / (sigma_norm ** 2))).exp()
                return K_ij.sum(dim=2).sum()

            grad_jax = jax.grad(jax_forward)(x)

            # PyTorch KeOps gradient
            x_t = torch.tensor(x_np, device='cuda', requires_grad=True)
            y_t = torch.tensor(y_np, device='cuda')
            nx_t = torch.tensor(nx_np, device='cuda')
            ny_t = torch.tensor(ny_np, device='cuda')
            wi_t = torch.tensor(wi_np, device='cuda')
            wj_t = torch.tensor(wj_np, device='cuda')

            x_i_t = LazyTensor_torch(x_t.view(B, n_i, 1, 3))
            y_j_t = LazyTensor_torch(y_t.view(B, 1, n_j, 3))
            nx_i_t = LazyTensor_torch(nx_t.view(B, n_i, 1, 3))
            ny_j_t = LazyTensor_torch(ny_t.view(B, 1, n_j, 3))
            wi_i_t = LazyTensor_torch(wi_t.view(B, n_i, 1, 1))
            wj_j_t = LazyTensor_torch(wj_t.view(B, 1, n_j, 1))

            sq_dist_pos_t = ((x_i_t - y_j_t) ** 2).sum(-1)
            sq_dist_norm_t = ((nx_i_t - ny_j_t) ** 2).sum(-1)
            K_ij_t = wi_i_t * wj_j_t * (-(sq_dist_pos_t / (sigma_pos ** 2))).exp() * (-(sq_dist_norm_t / (sigma_norm ** 2))).exp()
            loss_t = K_ij_t.sum(dim=2).sum()
            loss_t.backward()
            grad_torch = x_t.grad

            passed, diff = compare_arrays(
                np.array(grad_jax), grad_torch.cpu().numpy(),
                rtol=1e-3, atol=1e-3
            )

            if not passed:
                print(f"    FAIL at B={B}, size {n_i}x{n_j}: diff={diff:.2e}")
                all_passed = False
            max_diff = max(max_diff, diff)

    return all_passed, max_diff


# =============================================================================
# Test 5: Pm Parameter Gradient (vs PyTorch)
# =============================================================================

def test_pm_gradient():
    """Test Pm parameter gradient across sizes against PyTorch."""
    max_diff = 0.0
    all_passed = True

    for N, M in [(50, 40), (150, 120), (300, 250), (500, 400)]:
        np.random.seed(SEED)
        D = 3

        x_np = np.random.randn(N, D).astype(np.float32)
        y_np = np.random.randn(M, D).astype(np.float32)

        formula = "Exp(-SqNorm2(x-y) * s)"
        aliases = [f"x=Vi({D})", f"y=Vj({D})", "s=Pm(1)"]

        # JAX KeOps
        x_jax = jnp.array(x_np)
        y_jax = jnp.array(y_np)
        sigma_jax = jnp.array([0.5])

        op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1)
        grad_jax = jax.grad(lambda s: op_jax(x_jax, y_jax, s).sum())(sigma_jax)

        # PyTorch KeOps
        x_torch = torch.tensor(x_np, device='cuda')
        y_torch = torch.tensor(y_np, device='cuda')
        sigma_torch = torch.tensor([0.5], device='cuda', requires_grad=True)

        op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
        result_torch = op_torch(x_torch, y_torch, sigma_torch).sum()
        result_torch.backward()
        grad_torch = sigma_torch.grad

        passed, diff = compare_arrays(
            np.array(grad_jax), grad_torch.cpu().numpy(),
            rtol=1e-3, atol=1e-4
        )

        if not passed:
            print(f"    FAIL at size {N}x{M}: diff={diff:.2e}")
            all_passed = False
        max_diff = max(max_diff, diff)

    return all_passed, max_diff


# =============================================================================
# Test 6: Genred API (vs PyTorch)
# =============================================================================

def test_genred_forward_sizes():
    """Test Genred API across sizes against PyTorch."""
    max_diff = 0.0
    all_passed = True

    for N, M in TEST_SIZES:
        np.random.seed(SEED)
        D = 3

        x_np = np.random.randn(N, D).astype(np.float32)
        y_np = np.random.randn(M, D).astype(np.float32)
        b_np = np.random.randn(M, D).astype(np.float32)
        sigma_np = np.array([0.5], dtype=np.float32)

        formula = "Exp(-SqNorm2(x-y) * s) * b"
        aliases = [f"x=Vi({D})", f"y=Vj({D})", f"b=Vj({D})", "s=Pm(1)"]

        # JAX KeOps
        x_jax = jnp.array(x_np)
        y_jax = jnp.array(y_np)
        b_jax = jnp.array(b_np)
        sigma_jax = jnp.array(sigma_np)

        op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1)
        result_jax = op_jax(x_jax, y_jax, b_jax, sigma_jax)

        # PyTorch KeOps
        x_torch = torch.tensor(x_np, device='cuda')
        y_torch = torch.tensor(y_np, device='cuda')
        b_torch = torch.tensor(b_np, device='cuda')
        sigma_torch = torch.tensor(sigma_np, device='cuda')

        op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
        result_torch = op_torch(x_torch, y_torch, b_torch, sigma_torch)

        passed, diff = compare_arrays(
            np.array(result_jax), result_torch.cpu().numpy(),
            rtol=RTOL, atol=ATOL
        )

        if not passed:
            print(f"    FAIL at size {N}x{M}: diff={diff:.2e}")
            all_passed = False
        max_diff = max(max_diff, diff)

    return all_passed, max_diff


def test_genred_gradient_sizes():
    """Test Genred gradient across sizes against PyTorch."""
    max_diff = 0.0
    all_passed = True

    for N, M in [(50, 40), (150, 120), (300, 250), (500, 400)]:
        np.random.seed(SEED)
        D = 3

        x_np = np.random.randn(N, D).astype(np.float32)
        y_np = np.random.randn(M, D).astype(np.float32)
        b_np = np.random.randn(M, D).astype(np.float32)
        sigma_np = np.array([0.5], dtype=np.float32)

        formula = "Exp(-SqNorm2(x-y) * s) * b"
        aliases = [f"x=Vi({D})", f"y=Vj({D})", f"b=Vj({D})", "s=Pm(1)"]

        # JAX KeOps gradient w.r.t. x
        x_jax = jnp.array(x_np)
        y_jax = jnp.array(y_np)
        b_jax = jnp.array(b_np)
        sigma_jax = jnp.array(sigma_np)

        op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1)
        grad_jax = jax.grad(lambda x: op_jax(x, y_jax, b_jax, sigma_jax).sum())(x_jax)

        # PyTorch KeOps gradient
        x_torch = torch.tensor(x_np, device='cuda', requires_grad=True)
        y_torch = torch.tensor(y_np, device='cuda')
        b_torch = torch.tensor(b_np, device='cuda')
        sigma_torch = torch.tensor(sigma_np, device='cuda')

        op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
        result_torch = op_torch(x_torch, y_torch, b_torch, sigma_torch).sum()
        result_torch.backward()
        grad_torch = x_torch.grad

        passed, diff = compare_arrays(
            np.array(grad_jax), grad_torch.cpu().numpy(),
            rtol=1e-3, atol=1e-4
        )

        if not passed:
            print(f"    FAIL at size {N}x{M}: diff={diff:.2e}")
            all_passed = False
        max_diff = max(max_diff, diff)

    return all_passed, max_diff


# =============================================================================
# Test 7: High-Dimensional Operations (vs PyTorch)
# =============================================================================

def test_high_dim_forward():
    """Test high-dimensional operations against PyTorch."""
    max_diff = 0.0
    all_passed = True

    for D in [32, 64, 128]:
        for N, M in [(100, 80), (300, 250), (500, 400)]:
            np.random.seed(SEED)

            x_np = np.random.randn(N, D).astype(np.float32)
            y_np = np.random.randn(M, D).astype(np.float32)

            formula = "SqNorm2(x-y)"
            aliases = [f"x=Vi({D})", f"y=Vj({D})"]

            # JAX KeOps
            x_jax = jnp.array(x_np)
            y_jax = jnp.array(y_np)
            op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1)
            result_jax = op_jax(x_jax, y_jax)

            # PyTorch KeOps
            x_torch = torch.tensor(x_np, device='cuda')
            y_torch = torch.tensor(y_np, device='cuda')
            op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
            result_torch = op_torch(x_torch, y_torch)

            passed, diff = compare_arrays(
                np.array(result_jax), result_torch.cpu().numpy(),
                rtol=RTOL, atol=ATOL
            )

            if not passed:
                print(f"    FAIL at D={D}, size {N}x{M}: diff={diff:.2e}")
                all_passed = False
            max_diff = max(max_diff, diff)

    return all_passed, max_diff


def test_high_dim_gradient():
    """Test high-dimensional gradient computation against PyTorch.

    This specifically tests the shared memory block size fix for dimensions > 64.
    For D > 64, the block size must be reduced to fit shared memory (48KB limit).
    """
    max_diff = 0.0
    all_passed = True

    # Test dimensions that stress shared memory:
    # D=64: 192*64*4 = 49152 bytes (exactly at limit)
    # D=96: needs block reduction to ~125 threads
    # D=128: needs block reduction to ~93 threads
    # D=192: needs block reduction to ~62 threads
    for D in [64, 96, 128, 192]:
        for N, M in [(100, 80), (300, 250)]:
            np.random.seed(SEED)

            x_np = np.random.randn(N, D).astype(np.float32)
            y_np = np.random.randn(M, D).astype(np.float32)

            formula = "SqNorm2(x-y)"
            aliases = [f"x=Vi({D})", f"y=Vj({D})"]

            # JAX KeOps gradient
            x_jax = jnp.array(x_np)
            y_jax = jnp.array(y_np)
            op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1)

            def jax_forward(x_var):
                return op_jax(x_var, y_jax).sum()

            grad_jax = jax.grad(jax_forward)(x_jax)

            # PyTorch KeOps gradient
            x_torch = torch.tensor(x_np, device='cuda', requires_grad=True)
            y_torch = torch.tensor(y_np, device='cuda')
            op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
            result_torch = op_torch(x_torch, y_torch).sum()
            result_torch.backward()
            grad_torch = x_torch.grad

            passed, diff = compare_arrays(
                np.array(grad_jax), grad_torch.cpu().numpy(),
                rtol=1e-3, atol=1e-4
            )

            if not passed:
                print(f"    FAIL at D={D}, size {N}x{M}: diff={diff:.2e}")
                all_passed = False
            max_diff = max(max_diff, diff)

    return all_passed, max_diff


def test_high_dim_odd_sizes():
    """Test odd/problematic dimension sizes that previously caused issues.

    Before the shared memory block size fix, these dimensions failed due to
    incorrect block size calculation. This test ensures they all work now.

    Previously problematic dimensions included: 66, 68, 72, 74, 82-88, 110,
    118, 122-130, 134, 136, 144, and various odd numbers.
    """
    max_diff = 0.0
    all_passed = True

    # Include previously-problematic dimensions and odd sizes
    # These are dimensions that would fail before the fix due to:
    # - Being just over the 64-thread boundary
    # - Odd numbers that don't divide evenly
    # - Sizes around problematic thresholds
    problematic_dims = [
        65,   # Just over threshold
        66,   # Previously failed
        67,   # Odd number
        72,   # Previously failed
        82,   # Previously failed
        86,   # Previously failed
        88,   # Previously failed
        93,   # Odd - exactly at 48KB/512 boundary
        97,   # Prime number
        110,  # Previously failed
        118,  # Previously failed
        122,  # Previously failed
        127,  # Prime number
        129,  # Just over 128
        133,  # Odd
        144,  # Previously failed
        191,  # Prime, just under 192
        193,  # Prime, just over 192
        200,  # Round number
        255,  # 2^8 - 1
        256,  # 2^8
        257,  # 2^8 + 1
    ]

    N, M = 50, 40  # Use smaller sizes for speed

    for D in problematic_dims:
        np.random.seed(SEED)

        x_np = np.random.randn(N, D).astype(np.float32)
        y_np = np.random.randn(M, D).astype(np.float32)

        formula = "SqNorm2(x-y)"
        aliases = [f"x=Vi({D})", f"y=Vj({D})"]

        # JAX KeOps
        x_jax = jnp.array(x_np)
        y_jax = jnp.array(y_np)
        op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1)
        result_jax = op_jax(x_jax, y_jax)

        # PyTorch KeOps
        x_torch = torch.tensor(x_np, device='cuda')
        y_torch = torch.tensor(y_np, device='cuda')
        op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
        result_torch = op_torch(x_torch, y_torch)

        passed, diff = compare_arrays(
            np.array(result_jax), result_torch.cpu().numpy(),
            rtol=RTOL, atol=ATOL
        )

        if not passed:
            print(f"    FAIL at D={D}: diff={diff:.2e}")
            all_passed = False
        max_diff = max(max_diff, diff)

    return all_passed, max_diff


# =============================================================================
# Test 8: Sparse Variable Indices (Gradient Dropout) vs PyTorch
# =============================================================================

def test_sparse_variable_forward():
    """Test sparse variable indices forward pass against PyTorch."""
    np.random.seed(SEED)
    B, N, M = 2, 15, 20

    x_np = np.random.randn(B, N, 3).astype(np.float32) * 0.1
    y_np = np.random.randn(B, M, 3).astype(np.float32) * 0.1
    v_np = np.abs(np.random.randn(B, M, 1).astype(np.float32)) * 0.01 + 0.001

    # JAX KeOps
    x_jax = jnp.array(x_np)
    y_jax = jnp.array(y_np)
    v_jax = jnp.array(v_np)

    x_i = LazyTensor(x_jax[:, :, None, :])
    y_j = LazyTensor(y_jax[:, None, :, :])
    K = (-x_i.sqdist(y_j)).exp()
    result_jax = K @ v_jax

    # PyTorch KeOps
    x_torch = torch.tensor(x_np, device='cuda')
    y_torch = torch.tensor(y_np, device='cuda')
    v_torch = torch.tensor(v_np, device='cuda')

    x_i_t = LazyTensor_torch(x_torch[:, :, None, :])
    y_j_t = LazyTensor_torch(y_torch[:, None, :, :])
    K_t = (-x_i_t.sqdist(y_j_t)).exp()
    result_torch = K_t @ v_torch

    return compare_arrays(
        np.array(result_jax), result_torch.cpu().numpy(),
        rtol=RTOL, atol=ATOL
    )


def test_sparse_variable_gradient():
    """Test sparse variable gradient (variable dropout) against PyTorch.

    Critical test: d(K@v)/dv = K^T, where 'v' drops out of the gradient formula.
    """
    max_diff = 0.0
    all_passed = True

    for B in [1, 2]:
        for N, M in [(15, 20), (100, 80), (200, 150), (400, 300)]:
            np.random.seed(SEED)

            x_np = np.random.randn(B, N, 3).astype(np.float32) * 0.1
            y_np = np.random.randn(B, M, 3).astype(np.float32) * 0.1
            v_np = np.abs(np.random.randn(B, M, 1).astype(np.float32)) * 0.01 + 0.001

            # JAX KeOps gradient
            x_jax = jnp.array(x_np)
            y_jax = jnp.array(y_np)
            v_jax = jnp.array(v_np)

            def jax_forward(v_var):
                x_i = LazyTensor(x_jax[:, :, None, :])
                y_j = LazyTensor(y_jax[:, None, :, :])
                K = (-x_i.sqdist(y_j)).exp()
                return (K @ v_var).sum()

            grad_jax = jax.grad(jax_forward)(v_jax)

            # PyTorch KeOps gradient
            x_torch = torch.tensor(x_np, device='cuda')
            y_torch = torch.tensor(y_np, device='cuda')
            v_torch = torch.tensor(v_np, device='cuda', requires_grad=True)

            x_i_t = LazyTensor_torch(x_torch[:, :, None, :])
            y_j_t = LazyTensor_torch(y_torch[:, None, :, :])
            K_t = (-x_i_t.sqdist(y_j_t)).exp()
            loss_t = (K_t @ v_torch).sum()
            loss_t.backward()
            grad_torch = v_torch.grad

            passed, diff = compare_arrays(
                np.array(grad_jax), grad_torch.cpu().numpy(),
                rtol=1e-3, atol=1e-4
            )

            if not passed:
                print(f"    FAIL at B={B}, size {N}x{M}: diff={diff:.2e}")
                all_passed = False
            max_diff = max(max_diff, diff)

    return all_passed, max_diff


# =============================================================================
# Test 9: JIT Compilation (vs PyTorch)
# =============================================================================

def test_jit_forward():
    """Test JIT compilation produces correct results vs PyTorch."""
    np.random.seed(SEED)
    N, M, D = 200, 150, 3

    x_np = np.random.randn(N, D).astype(np.float32)
    y_np = np.random.randn(M, D).astype(np.float32)

    formula = "SqNorm2(x-y)"
    aliases = [f"x=Vi({D})", f"y=Vj({D})"]

    # JAX KeOps with JIT
    x_jax = jnp.array(x_np)
    y_jax = jnp.array(y_np)
    op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1)

    @jax.jit
    def compute_jax(x, y):
        return op_jax(x, y)

    result_jax = compute_jax(x_jax, y_jax)

    # PyTorch KeOps
    x_torch = torch.tensor(x_np, device='cuda')
    y_torch = torch.tensor(y_np, device='cuda')
    op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
    result_torch = op_torch(x_torch, y_torch)

    return compare_arrays(
        np.array(result_jax), result_torch.cpu().numpy(),
        rtol=RTOL, atol=ATOL
    )


def test_jit_gradient():
    """Test JIT gradient computation vs PyTorch."""
    np.random.seed(SEED)
    N, M, D = 200, 150, 3

    x_np = np.random.randn(N, D).astype(np.float32)
    y_np = np.random.randn(M, D).astype(np.float32)

    formula = "SqNorm2(x-y)"
    aliases = [f"x=Vi({D})", f"y=Vj({D})"]

    # JAX KeOps with JIT
    x_jax = jnp.array(x_np)
    y_jax = jnp.array(y_np)
    op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1)

    @jax.jit
    def compute_grad_jax(x, y):
        return jax.grad(lambda x: op_jax(x, y).sum())(x)

    grad_jax = compute_grad_jax(x_jax, y_jax)

    # PyTorch KeOps
    x_torch = torch.tensor(x_np, device='cuda', requires_grad=True)
    y_torch = torch.tensor(y_np, device='cuda')
    op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
    result_torch = op_torch(x_torch, y_torch).sum()
    result_torch.backward()
    grad_torch = x_torch.grad

    return compare_arrays(
        np.array(grad_jax), grad_torch.cpu().numpy(),
        rtol=1e-3, atol=1e-4
    )


# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    print_header("KeOps JAX Edge Case Tests",
                "All tests compare JAX KeOps against PyTorch KeOps (ground truth)")

    print_environment_info()

    suite = TestSuite("Edge Case Tests", "JAX KeOps vs PyTorch KeOps")

    # Test 1: Scalar Multiplication
    print_subheader("Scalar Multiplication (Multiple Sizes)")
    run_test("Scalar mult sizes (10x10 to 1000x800)", test_scalar_multiplication_sizes, suite)
    run_test("Scalar mult batched", test_scalar_multiplication_batched, suite)

    # Test 2: Gaussian Kernel
    print_subheader("Gaussian Kernel (Multiple Sizes)")
    run_test("Gaussian forward sizes", test_gaussian_kernel_sizes, suite)
    run_test("Gaussian gradient sizes", test_gaussian_kernel_gradient_sizes, suite)

    # Test 3: Kernel-Vector Product
    print_subheader("Kernel @ Vector Product (Multiple Sizes)")
    run_test("K @ v forward sizes", test_kernel_matmul_sizes, suite)
    run_test("K @ v gradient sizes (sparse var)", test_kernel_matmul_gradient_sizes, suite)

    # Test 4: Varifold Kernel
    print_subheader("Varifold Kernel (6 Variables)")
    run_test("Varifold forward", test_varifold_forward, suite)
    run_test("Varifold gradient", test_varifold_gradient, suite)

    # Test 5: Pm Parameter Gradient
    print_subheader("Pm Parameter Gradient")
    run_test("Pm gradient sizes", test_pm_gradient, suite)

    # Test 6: Genred API
    print_subheader("Genred API (Multiple Sizes)")
    run_test("Genred forward sizes", test_genred_forward_sizes, suite)
    run_test("Genred gradient sizes", test_genred_gradient_sizes, suite)

    # Test 7: High-Dimensional (shared memory block size fix)
    print_subheader("High-Dimensional Operations (Shared Memory Fix)")
    run_test("High-dim forward (D=32,64,128)", test_high_dim_forward, suite)
    run_test("High-dim gradient (D=64,96,128,192)", test_high_dim_gradient, suite)
    run_test("High-dim odd sizes (D=65-257)", test_high_dim_odd_sizes, suite)

    # Test 8: Sparse Variable Indices
    print_subheader("Sparse Variable Indices (Gradient Dropout)")
    run_test("Sparse var forward", test_sparse_variable_forward, suite)
    run_test("Sparse var gradient (critical)", test_sparse_variable_gradient, suite)

    # Test 9: JIT Compilation
    print_subheader("JIT Compilation")
    run_test("JIT forward", test_jit_forward, suite)
    run_test("JIT gradient", test_jit_gradient, suite)

    # Print summary
    suite.print_summary()

    return 0 if suite.all_passed() else 1


if __name__ == "__main__":
    sys.exit(main())
