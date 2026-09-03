#!/usr/bin/env python3
"""
KeOps JAX Batched Gradient Tests
================================
Unit tests for batched (3D tensor) gradient computation in JAX KeOps.

These tests verify that:
1. Batched gradient values match PyTorch KeOps (the reference implementation)
2. Block boundary edge cases (N=192, 193, 194) are handled correctly
3. No NaN values appear at CUDA block boundaries
4. Multi-batch operations produce correct gradients

This is critical for catching regressions in the gradient computation,
particularly for batched operations where axis swapping and offset table
generation can produce subtle bugs.

The tests compare against PyTorch KeOps as the ground truth since it is
the mature, well-tested reference implementation.
"""

import sys
import pytest
import numpy as np

from test_utils import (
    TestSuite, print_header, print_subheader, print_info, print_warning,
    compare_arrays, run_test, print_environment_info, RICH_AVAILABLE, Status,
    setup_jax_float64, get_np_dtype, get_dtype_str, is_float64_mode
)

# Setup float64 mode BEFORE importing JAX
setup_jax_float64()

# =============================================================================
# Import Frameworks
# =============================================================================

# JAX
try:
    import jax
    import jax.numpy as jnp
    from jax import grad
    JAX_AVAILABLE = True
except ImportError as e:
    print(f"Error: JAX not found: {e}")
    JAX_AVAILABLE = False

# PyTorch
try:
    import torch
    torch.set_default_dtype(torch.float64 if is_float64_mode() else torch.float32)
    TORCH_AVAILABLE = torch.cuda.is_available()
    if not TORCH_AVAILABLE:
        print("Warning: PyTorch CUDA not available")
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed")

# KeOps
try:
    from pykeops.jax import Genred as JaxGenred
    from pykeops.torch import LazyTensor as TorchLazyTensor
    KEOPS_AVAILABLE = True
except ImportError as e:
    print(f"Error: KeOps not found: {e}")
    KEOPS_AVAILABLE = False

# =============================================================================
# Configuration
# =============================================================================

# Every test here needs a GPU and compares against PyTorch KeOps. conftest.py registers these
# markers and skips on missing hardware, so declaring them at module level is what makes
# `pytest -m pytorch` and `pytest -m gpu` select anything.
pytestmark = [pytest.mark.gpu, pytest.mark.pytorch]

SEED = 42
# Use tight tolerances since we expect near-exact matches
RTOL = 1e-4
ATOL = 1e-5


# =============================================================================
# Test Data Generator
# =============================================================================

def get_test_data(B, N, D, seed=SEED):
    """Generate identical test data for both frameworks."""
    np.random.seed(seed)
    x_np = np.random.randn(B, N, D).astype(get_np_dtype())
    y_np = np.random.randn(B, N, D).astype(get_np_dtype())
    return x_np, y_np


# =============================================================================
# PyTorch Reference Implementation
# =============================================================================

def pytorch_forward_and_grad(x_np, y_np, formula='sqdist'):
    """
    Compute forward pass and gradients using PyTorch KeOps.

    Returns: (result, grad_x, grad_y) as numpy arrays
    """
    B, N, D = x_np.shape

    x_torch = torch.tensor(x_np, requires_grad=True, device='cuda')
    y_torch = torch.tensor(y_np, requires_grad=True, device='cuda')

    # LazyTensor setup for 3D
    x_i = TorchLazyTensor(x_torch[:, :, None, :])  # (B, N, 1, D)
    y_j = TorchLazyTensor(y_torch[:, None, :, :])  # (B, 1, N, D)

    if formula == 'sqdist':
        # Sum of squared differences: Sum_j (x_i - y_j)^2
        sq_diff = (x_i - y_j) ** 2  # (B, N, N, D)
        result = sq_diff.sum(dim=2)  # Sum over j -> (B, N, D)
    elif formula == 'gaussian':
        # Gaussian kernel: Sum_j exp(-||x_i - y_j||^2)
        D_ij = ((x_i - y_j) ** 2).sum(-1)  # (B, N, N)
        K_ij = (-D_ij).exp()
        result = K_ij.sum(dim=2)  # (B, N, 1)
    else:
        raise ValueError(f"Unknown formula: {formula}")

    # Backward pass
    loss = result.sum()
    loss.backward()

    return (
        result.detach().cpu().numpy(),
        x_torch.grad.cpu().numpy(),
        y_torch.grad.cpu().numpy()
    )


# =============================================================================
# JAX Implementation
# =============================================================================

def jax_forward_and_grad(x_np, y_np, formula='sqdist'):
    """
    Compute forward pass and gradients using JAX KeOps.

    Returns: (result, grad_x, grad_y) as numpy arrays
    """
    B, N, D = x_np.shape

    x_jax = jnp.array(x_np)
    y_jax = jnp.array(y_np)

    if formula == 'sqdist':
        keops_formula = "Square(x - y)"
        aliases = [f"x=Vi({D})", f"y=Vj({D})"]
        op = JaxGenred(keops_formula, aliases, reduction_op='Sum', axis=1, dtype=get_dtype_str())
    elif formula == 'gaussian':
        keops_formula = "Exp(-SqNorm2(x-y))"
        aliases = [f"x=Vi({D})", f"y=Vj({D})"]
        op = JaxGenred(keops_formula, aliases, reduction_op='Sum', axis=1, dtype=get_dtype_str())
    else:
        raise ValueError(f"Unknown formula: {formula}")

    def loss_fn(x, y):
        return jnp.sum(op(x, y))

    # Forward
    result = op(x_jax, y_jax)

    # Gradients
    grad_x, grad_y = grad(loss_fn, argnums=(0, 1))(x_jax, y_jax)

    return (
        np.array(result),
        np.array(grad_x),
        np.array(grad_y)
    )


# =============================================================================
# Comparison Tests
# =============================================================================

def check_forward_match(B, N, D, formula='sqdist'):
    """Test that forward pass matches between JAX and PyTorch."""
    x_np, y_np = get_test_data(B, N, D)

    result_torch, _, _ = pytorch_forward_and_grad(x_np, y_np, formula)
    result_jax, _, _ = jax_forward_and_grad(x_np, y_np, formula)

    return compare_arrays(result_torch, result_jax, rtol=RTOL, atol=ATOL)


def check_gradient_match(B, N, D, formula='sqdist'):
    """Test that gradients match between JAX and PyTorch."""
    x_np, y_np = get_test_data(B, N, D)

    _, grad_x_torch, grad_y_torch = pytorch_forward_and_grad(x_np, y_np, formula)
    _, grad_x_jax, grad_y_jax = jax_forward_and_grad(x_np, y_np, formula)

    # Check both gradients
    match_x, diff_x = compare_arrays(grad_x_torch, grad_x_jax, rtol=RTOL, atol=ATOL)
    match_y, diff_y = compare_arrays(grad_y_torch, grad_y_jax, rtol=RTOL, atol=ATOL)

    # Return overall match and max diff
    return match_x and match_y, max(diff_x, diff_y)


def check_gradient_no_nan(B, N, D, formula='sqdist'):
    """Test that gradients contain no NaN values."""
    x_np, y_np = get_test_data(B, N, D)

    _, grad_x, grad_y = jax_forward_and_grad(x_np, y_np, formula)

    has_nan_x = np.any(np.isnan(grad_x))
    has_nan_y = np.any(np.isnan(grad_y))

    if has_nan_x or has_nan_y:
        return False, float('inf')
    return True, 0.0


def test_block_boundary_gradients():
    """
    Test gradients at CUDA block boundaries.

    This specifically tests the fix for the NaN gradient bug at index 192
    (CUDA block size boundary) for batched 3D tensors.
    """
    # Test various sizes around the block boundary (192)
    test_sizes = [191, 192, 193, 194, 384, 385]
    B, D = 1, 3

    all_passed = True
    max_diff = 0.0

    for N in test_sizes:
        x_np, y_np = get_test_data(B, N, D)

        _, grad_x_torch, grad_y_torch = pytorch_forward_and_grad(x_np, y_np)
        _, grad_x_jax, grad_y_jax = jax_forward_and_grad(x_np, y_np)

        match_x, diff_x = compare_arrays(grad_x_torch, grad_x_jax, rtol=RTOL, atol=ATOL)
        match_y, diff_y = compare_arrays(grad_y_torch, grad_y_jax, rtol=RTOL, atol=ATOL)

        if not (match_x and match_y):
            all_passed = False
        max_diff = max(max_diff, diff_x, diff_y)

        # Also check for NaN at the block boundary
        if N > 192:
            if np.any(np.isnan(grad_x_jax[0, 192, :])) or np.any(np.isnan(grad_y_jax[0, 192, :])):
                all_passed = False
                max_diff = float('inf')

    return all_passed, max_diff


def test_multi_batch_gradients():
    """Test gradients with multiple batches."""
    test_cases = [
        (2, 100, 3),
        (4, 193, 3),
        (8, 50, 3),
    ]

    all_passed = True
    max_diff = 0.0

    for B, N, D in test_cases:
        x_np, y_np = get_test_data(B, N, D)

        _, grad_x_torch, grad_y_torch = pytorch_forward_and_grad(x_np, y_np)
        _, grad_x_jax, grad_y_jax = jax_forward_and_grad(x_np, y_np)

        match_x, diff_x = compare_arrays(grad_x_torch, grad_x_jax, rtol=RTOL, atol=ATOL)
        match_y, diff_y = compare_arrays(grad_y_torch, grad_y_jax, rtol=RTOL, atol=ATOL)

        if not (match_x and match_y):
            all_passed = False
        max_diff = max(max_diff, diff_x, diff_y)

    return all_passed, max_diff


def test_gaussian_kernel_gradient():
    """Test Gaussian kernel gradients match PyTorch."""
    B, N, D = 1, 100, 3
    x_np, y_np = get_test_data(B, N, D)

    _, grad_x_torch, grad_y_torch = pytorch_forward_and_grad(x_np, y_np, 'gaussian')
    _, grad_x_jax, grad_y_jax = jax_forward_and_grad(x_np, y_np, 'gaussian')

    match_x, diff_x = compare_arrays(grad_x_torch, grad_x_jax, rtol=RTOL, atol=ATOL)
    match_y, diff_y = compare_arrays(grad_y_torch, grad_y_jax, rtol=RTOL, atol=ATOL)

    return match_x and match_y, max(diff_x, diff_y)


def test_specific_index_192():
    """
    Specifically test the gradient at index 192 (block boundary).

    This is a regression test for the NaN gradient bug that occurred
    at CUDA block boundaries for batched 3D tensors.
    """
    B, N, D = 1, 193, 3
    x_np, y_np = get_test_data(B, N, D)

    _, grad_x_torch, grad_y_torch = pytorch_forward_and_grad(x_np, y_np)
    _, grad_x_jax, grad_y_jax = jax_forward_and_grad(x_np, y_np)

    # Check specifically at index 192
    grad_x_192_torch = grad_x_torch[0, 192, :]
    grad_x_192_jax = grad_x_jax[0, 192, :]
    grad_y_192_torch = grad_y_torch[0, 192, :]
    grad_y_192_jax = grad_y_jax[0, 192, :]

    # Check for NaN
    if np.any(np.isnan(grad_x_192_jax)) or np.any(np.isnan(grad_y_192_jax)):
        return False, float('inf')

    # Check values match
    match_x, diff_x = compare_arrays(grad_x_192_torch, grad_x_192_jax, rtol=RTOL, atol=ATOL)
    match_y, diff_y = compare_arrays(grad_y_192_torch, grad_y_192_jax, rtol=RTOL, atol=ATOL)

    return match_x and match_y, max(diff_x, diff_y)


# =============================================================================
# Main
# =============================================================================

def main():
    print_header("Batched Gradient Tests",
                 "Verifying 3D tensor gradients against PyTorch reference")
    print_environment_info()

    # Check prerequisites
    if not JAX_AVAILABLE:
        print_warning("JAX not available, skipping tests")
        return 1

    if not TORCH_AVAILABLE:
        print_warning("PyTorch CUDA not available, skipping tests")
        return 1

    if not KEOPS_AVAILABLE:
        print_warning("KeOps not available, skipping tests")
        return 1

    suite = TestSuite("Batched Gradients", "3D Tensor Gradient Verification")

    # 1. Forward Pass Tests
    print_subheader("1. Forward Pass (Batched)")
    run_test("B=1, N=100, D=3 (SqDist)", lambda: check_forward_match(1, 100, 3), suite)
    run_test("B=1, N=193, D=3 (SqDist)", lambda: check_forward_match(1, 193, 3), suite)
    run_test("B=4, N=100, D=3 (SqDist)", lambda: check_forward_match(4, 100, 3), suite)
    run_test("B=1, N=100, D=3 (Gaussian)", lambda: check_forward_match(1, 100, 3, 'gaussian'), suite)

    # 2. Gradient Tests - Basic
    print_subheader("2. Gradient Comparison - Basic Sizes")
    run_test("B=1, N=50, D=3", lambda: check_gradient_match(1, 50, 3), suite)
    run_test("B=1, N=100, D=3", lambda: check_gradient_match(1, 100, 3), suite)
    run_test("B=1, N=200, D=3", lambda: check_gradient_match(1, 200, 3), suite)

    # 3. Block Boundary Tests (Critical for NaN bug)
    print_subheader("3. Block Boundary Edge Cases (N~192)")
    run_test("B=1, N=192, D=3 (exact boundary)", lambda: check_gradient_match(1, 192, 3), suite)
    run_test("B=1, N=193, D=3 (over boundary)", lambda: check_gradient_match(1, 193, 3), suite)
    run_test("B=1, N=194, D=3", lambda: check_gradient_match(1, 194, 3), suite)
    run_test("B=1, N=384, D=3 (two blocks)", lambda: check_gradient_match(1, 384, 3), suite)
    run_test("Index 192 Specific Test", test_specific_index_192, suite)
    run_test("Block Boundary Suite", test_block_boundary_gradients, suite)

    # 4. Multi-Batch Tests
    print_subheader("4. Multi-Batch Gradients")
    run_test("B=2, N=100, D=3", lambda: check_gradient_match(2, 100, 3), suite)
    run_test("B=4, N=193, D=3", lambda: check_gradient_match(4, 193, 3), suite)
    run_test("B=8, N=50, D=3", lambda: check_gradient_match(8, 50, 3), suite)
    run_test("Multi-Batch Suite", test_multi_batch_gradients, suite)

    # 5. NaN Detection Tests
    print_subheader("5. NaN Detection")
    run_test("No NaN: B=1, N=193, D=3", lambda: check_gradient_no_nan(1, 193, 3), suite)
    run_test("No NaN: B=1, N=384, D=3", lambda: check_gradient_no_nan(1, 384, 3), suite)
    run_test("No NaN: B=4, N=193, D=3", lambda: check_gradient_no_nan(4, 193, 3), suite)

    # 6. Different Kernels
    print_subheader("6. Different Kernel Types")
    run_test("Gaussian Kernel Gradient", test_gaussian_kernel_gradient, suite)

    # Final Summary
    suite.print_summary()
    return 0 if suite.all_passed() else 1


if __name__ == "__main__":
    sys.exit(main())
