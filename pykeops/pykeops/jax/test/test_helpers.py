#!/usr/bin/env python3
"""
KeOps JAX Helper Function Tests
===============================

Tests for the generic_sum, generic_logsumexp, etc. helper functions.

These tests verify:
1. Helper functions don't raise errors during creation and execution
2. Dimensions are correctly parsed from the output specification
3. Results match PyTorch KeOps (ground truth)
4. Various dimension configurations work correctly
"""

import sys
import pytest
import numpy as np

from test_utils import (
    TestSuite, print_header, print_subheader,
    compare_arrays, run_test, print_environment_info, RICH_AVAILABLE,
    setup_jax_float64, get_np_dtype, get_dtype_str, is_float64_mode
)

# Setup float64 mode BEFORE importing JAX
setup_jax_float64()

import jax
import jax.numpy as jnp

# =============================================================================
# Import KeOps
# =============================================================================

try:
    from pykeops.jax import (
        generic_sum,
        generic_logsumexp,
        generic_argmin,
        generic_argkmin,
        generic_min,
        generic_max,
    )
    KEOPS_JAX_AVAILABLE = True
except ImportError as e:
    print(f"Error: pykeops.jax helpers not found: {e}")
    sys.exit(1)

# PyTorch KeOps for ground truth
try:
    import torch
    from pykeops.torch import (
        generic_sum as generic_sum_torch,
        generic_logsumexp as generic_logsumexp_torch,
        generic_argmin as generic_argmin_torch,
        generic_argkmin as generic_argkmin_torch,
    )
    from pykeops.torch import Genred as Genred_torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

if not TORCH_AVAILABLE:
    print("Error: PyTorch KeOps with CUDA is required for these tests")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

# Every test here needs a GPU and compares against PyTorch KeOps. conftest.py registers these
# markers and skips on missing hardware, so declaring them at module level is what makes
# `pytest -m pytorch` and `pytest -m gpu` select anything.
pytestmark = [pytest.mark.gpu, pytest.mark.pytorch]

SEED = 42
RTOL = 1e-4
ATOL = 1e-5


# =============================================================================
# Test Functions
# =============================================================================

def test_generic_sum_basic():
    """Test generic_sum with basic SqDist formula."""
    np.random.seed(SEED)
    N, M, D = 100, 80, 3

    x_np = np.random.randn(N, D).astype(get_np_dtype())
    y_np = np.random.randn(M, D).astype(get_np_dtype())

    # JAX
    my_conv = generic_sum(
        'SqDist(x, y)',
        'a = Vi(1)',
        'x = Vi(3)',
        'y = Vj(3)',
        dtype=get_dtype_str()
    )
    result_jax = my_conv(jnp.array(x_np), jnp.array(y_np))

    # PyTorch
    torch_dtype = torch.float64 if is_float64_mode() else torch.float32
    my_conv_torch = generic_sum_torch(
        'SqDist(x, y)',
        'a = Vi(1)',
        'x = Vi(3)',
        'y = Vj(3)',
        dtype=get_dtype_str()
    )
    result_torch = my_conv_torch(
        torch.tensor(x_np, device='cuda', dtype=torch_dtype),
        torch.tensor(y_np, device='cuda', dtype=torch_dtype)
    ).cpu().numpy()

    return compare_arrays(result_jax, result_torch, rtol=RTOL, atol=ATOL)


def test_generic_sum_gaussian():
    """Test generic_sum with Gaussian kernel formula."""
    np.random.seed(SEED)
    N, M, D = 100, 80, 3

    x_np = np.random.randn(N, D).astype(get_np_dtype())
    y_np = np.random.randn(M, D).astype(get_np_dtype())

    # JAX
    my_conv = generic_sum(
        'Exp(-SqNorm2(x - y))',
        'a = Vi(1)',
        'x = Vi(3)',
        'y = Vj(3)',
        dtype=get_dtype_str()
    )
    result_jax = my_conv(jnp.array(x_np), jnp.array(y_np))

    # PyTorch
    torch_dtype = torch.float64 if is_float64_mode() else torch.float32
    my_conv_torch = generic_sum_torch(
        'Exp(-SqNorm2(x - y))',
        'a = Vi(1)',
        'x = Vi(3)',
        'y = Vj(3)',
        dtype=get_dtype_str()
    )
    result_torch = my_conv_torch(
        torch.tensor(x_np, device='cuda', dtype=torch_dtype),
        torch.tensor(y_np, device='cuda', dtype=torch_dtype)
    ).cpu().numpy()

    return compare_arrays(result_jax, result_torch, rtol=RTOL, atol=ATOL)


def test_generic_sum_axis_vj():
    """Test generic_sum with Vj output (reduction along axis 0)."""
    np.random.seed(SEED)
    N, M, D = 100, 80, 3

    x_np = np.random.randn(N, D).astype(get_np_dtype())
    y_np = np.random.randn(M, D).astype(get_np_dtype())

    # JAX - output is Vj, so reduction is along axis 0 (over i)
    my_conv = generic_sum(
        'SqDist(x, y)',
        'a = Vj(1)',  # Output indexed by j
        'x = Vi(3)',
        'y = Vj(3)',
        dtype=get_dtype_str()
    )
    result_jax = my_conv(jnp.array(x_np), jnp.array(y_np))

    # PyTorch
    torch_dtype = torch.float64 if is_float64_mode() else torch.float32
    my_conv_torch = generic_sum_torch(
        'SqDist(x, y)',
        'a = Vj(1)',
        'x = Vi(3)',
        'y = Vj(3)',
        dtype=get_dtype_str()
    )
    result_torch = my_conv_torch(
        torch.tensor(x_np, device='cuda', dtype=torch_dtype),
        torch.tensor(y_np, device='cuda', dtype=torch_dtype)
    ).cpu().numpy()

    # Check shape - should be (M, 1) not (N, 1)
    if result_jax.shape[0] != M:
        return False, float('inf')

    return compare_arrays(result_jax, result_torch, rtol=RTOL, atol=ATOL)


def test_generic_sum_various_dims():
    """Test generic_sum with various input/output dimensions."""
    np.random.seed(SEED)

    test_configs = [
        (50, 40, 1, 1),    # D=1, output_dim=1
        (50, 40, 5, 1),    # D=5, output_dim=1
        (50, 40, 10, 1),   # D=10, output_dim=1
        (50, 40, 32, 1),   # D=32, output_dim=1
        (50, 40, 64, 1),   # D=64, output_dim=1
        (50, 40, 128, 1),  # D=128, output_dim=1
    ]

    all_passed = True
    max_diff = 0.0

    for N, M, D, out_dim in test_configs:
        x_np = np.random.randn(N, D).astype(get_np_dtype())
        y_np = np.random.randn(M, D).astype(get_np_dtype())

        # JAX
        my_conv = generic_sum(
            'SqDist(x, y)',
            f'a = Vi({out_dim})',
            f'x = Vi({D})',
            f'y = Vj({D})',
            dtype=get_dtype_str()
        )
        result_jax = my_conv(jnp.array(x_np), jnp.array(y_np))

        # PyTorch
        torch_dtype = torch.float64 if is_float64_mode() else torch.float32
        my_conv_torch = generic_sum_torch(
            'SqDist(x, y)',
            f'a = Vi({out_dim})',
            f'x = Vi({D})',
            f'y = Vj({D})',
            dtype=get_dtype_str()
        )
        result_torch = my_conv_torch(
            torch.tensor(x_np, device='cuda', dtype=torch_dtype),
            torch.tensor(y_np, device='cuda', dtype=torch_dtype)
        ).cpu().numpy()

        passed, diff = compare_arrays(result_jax, result_torch, rtol=RTOL, atol=ATOL)
        if not passed:
            all_passed = False
        max_diff = max(max_diff, diff)

    return all_passed, max_diff


def test_generic_logsumexp():
    """Test generic_logsumexp."""
    np.random.seed(SEED)
    N, M, D = 100, 80, 3

    x_np = np.random.randn(N, D).astype(get_np_dtype()) * 0.1
    y_np = np.random.randn(M, D).astype(get_np_dtype()) * 0.1

    # JAX
    my_conv = generic_logsumexp(
        '-SqNorm2(x - y)',
        'a = Vi(1)',
        'x = Vi(3)',
        'y = Vj(3)',
        dtype=get_dtype_str()
    )
    result_jax = my_conv(jnp.array(x_np), jnp.array(y_np))

    # PyTorch
    torch_dtype = torch.float64 if is_float64_mode() else torch.float32
    my_conv_torch = generic_logsumexp_torch(
        '-SqNorm2(x - y)',
        'a = Vi(1)',
        'x = Vi(3)',
        'y = Vj(3)',
        dtype=get_dtype_str()
    )
    result_torch = my_conv_torch(
        torch.tensor(x_np, device='cuda', dtype=torch_dtype),
        torch.tensor(y_np, device='cuda', dtype=torch_dtype)
    ).cpu().numpy()

    return compare_arrays(result_jax, result_torch, rtol=RTOL, atol=ATOL)


def test_generic_argmin():
    """Test generic_argmin (nearest neighbor search)."""
    np.random.seed(SEED)
    N, M, D = 50, 200, 10

    x_np = np.random.randn(N, D).astype(get_np_dtype())
    y_np = np.random.randn(M, D).astype(get_np_dtype())

    # JAX
    nn = generic_argmin(
        'SqDist(x, y)',
        'a = Vi(1)',
        f'x = Vi({D})',
        f'y = Vj({D})',
        dtype=get_dtype_str()
    )
    result_jax = nn(jnp.array(x_np), jnp.array(y_np))

    # PyTorch
    torch_dtype = torch.float64 if is_float64_mode() else torch.float32
    nn_torch = generic_argmin_torch(
        'SqDist(x, y)',
        'a = Vi(1)',
        f'x = Vi({D})',
        f'y = Vj({D})',
        dtype=get_dtype_str()
    )
    result_torch = nn_torch(
        torch.tensor(x_np, device='cuda', dtype=torch_dtype),
        torch.tensor(y_np, device='cuda', dtype=torch_dtype)
    ).cpu().numpy()

    return compare_arrays(result_jax, result_torch, rtol=RTOL, atol=ATOL)


def test_generic_argkmin():
    """Test generic_argkmin (K nearest neighbors)."""
    np.random.seed(SEED)
    N, M, D = 50, 200, 10
    K = 5

    x_np = np.random.randn(N, D).astype(get_np_dtype())
    y_np = np.random.randn(M, D).astype(get_np_dtype())

    # JAX
    knn = generic_argkmin(
        'SqDist(x, y)',
        f'a = Vi({K})',  # Find K nearest neighbors
        f'x = Vi({D})',
        f'y = Vj({D})',
        dtype=get_dtype_str()
    )
    result_jax = knn(jnp.array(x_np), jnp.array(y_np))

    # PyTorch
    torch_dtype = torch.float64 if is_float64_mode() else torch.float32
    knn_torch = generic_argkmin_torch(
        'SqDist(x, y)',
        f'a = Vi({K})',
        f'x = Vi({D})',
        f'y = Vj({D})',
        dtype=get_dtype_str()
    )
    result_torch = knn_torch(
        torch.tensor(x_np, device='cuda', dtype=torch_dtype),
        torch.tensor(y_np, device='cuda', dtype=torch_dtype)
    ).cpu().numpy()

    # Check shape - should be (N, K)
    if result_jax.shape != (N, K):
        return False, float('inf')

    return compare_arrays(result_jax, result_torch, rtol=RTOL, atol=ATOL)


def test_generic_min():
    """Test generic_min."""
    np.random.seed(SEED)
    N, M, D = 100, 80, 3

    x_np = np.random.randn(N, D).astype(get_np_dtype())
    y_np = np.random.randn(M, D).astype(get_np_dtype())

    # JAX
    my_conv = generic_min(
        'SqDist(x, y)',
        'a = Vi(1)',
        'x = Vi(3)',
        'y = Vj(3)',
        dtype=get_dtype_str()
    )
    result_jax = my_conv(jnp.array(x_np), jnp.array(y_np))

    # PyTorch - use Genred directly since generic_min may not exist
    torch_dtype = torch.float64 if is_float64_mode() else torch.float32
    my_conv_torch = Genred_torch(
        'SqDist(x, y)',
        ['x = Vi(3)', 'y = Vj(3)'],
        reduction_op='Min',
        axis=1,
        dtype=get_dtype_str()
    )
    result_torch = my_conv_torch(
        torch.tensor(x_np, device='cuda', dtype=torch_dtype),
        torch.tensor(y_np, device='cuda', dtype=torch_dtype)
    ).cpu().numpy()

    return compare_arrays(result_jax, result_torch, rtol=RTOL, atol=ATOL)


def test_generic_max():
    """Test generic_max."""
    np.random.seed(SEED)
    N, M, D = 100, 80, 3

    x_np = np.random.randn(N, D).astype(get_np_dtype())
    y_np = np.random.randn(M, D).astype(get_np_dtype())

    # JAX
    my_conv = generic_max(
        'SqDist(x, y)',
        'a = Vi(1)',
        'x = Vi(3)',
        'y = Vj(3)',
        dtype=get_dtype_str()
    )
    result_jax = my_conv(jnp.array(x_np), jnp.array(y_np))

    # PyTorch
    torch_dtype = torch.float64 if is_float64_mode() else torch.float32
    my_conv_torch = Genred_torch(
        'SqDist(x, y)',
        ['x = Vi(3)', 'y = Vj(3)'],
        reduction_op='Max',
        axis=1,
        dtype=get_dtype_str()
    )
    result_torch = my_conv_torch(
        torch.tensor(x_np, device='cuda', dtype=torch_dtype),
        torch.tensor(y_np, device='cuda', dtype=torch_dtype)
    ).cpu().numpy()

    return compare_arrays(result_jax, result_torch, rtol=RTOL, atol=ATOL)


def test_dimension_parsing():
    """Test that dimensions are correctly parsed from various output specifications."""
    np.random.seed(SEED)

    # Test different ways of specifying dimensions
    test_cases = [
        ('a = Vi(1)', 100, 80, 5, (100, 1)),    # Standard Vi
        ('a = Vj(1)', 100, 80, 5, (80, 1)),     # Standard Vj
        ('out = Vi(1)', 100, 80, 5, (100, 1)),  # Different name
        ('result = Vi(1)', 100, 80, 5, (100, 1)),  # Another name
    ]

    all_passed = True
    max_diff = 0.0

    for output_spec, N, M, D, expected_shape in test_cases:
        x_np = np.random.randn(N, D).astype(get_np_dtype())
        y_np = np.random.randn(M, D).astype(get_np_dtype())

        my_conv = generic_sum(
            'SqDist(x, y)',
            output_spec,
            f'x = Vi({D})',
            f'y = Vj({D})',
            dtype=get_dtype_str()
        )
        result = my_conv(jnp.array(x_np), jnp.array(y_np))

        if result.shape != expected_shape:
            print(f"    Shape mismatch for '{output_spec}': expected {expected_shape}, got {result.shape}")
            all_passed = False
            max_diff = float('inf')

    return all_passed, max_diff


def test_with_pm_parameter():
    """Test helper functions with Pm (parameter) variables."""
    np.random.seed(SEED)
    N, M, D = 100, 80, 3

    x_np = np.random.randn(N, D).astype(get_np_dtype())
    y_np = np.random.randn(M, D).astype(get_np_dtype())
    sigma_np = np.array([0.5], dtype=get_np_dtype())

    # JAX
    my_conv = generic_sum(
        'Exp(-SqNorm2(x - y) * s)',
        'a = Vi(1)',
        'x = Vi(3)',
        'y = Vj(3)',
        's = Pm(1)',
        dtype=get_dtype_str()
    )
    result_jax = my_conv(jnp.array(x_np), jnp.array(y_np), jnp.array(sigma_np))

    # PyTorch
    torch_dtype = torch.float64 if is_float64_mode() else torch.float32
    my_conv_torch = generic_sum_torch(
        'Exp(-SqNorm2(x - y) * s)',
        'a = Vi(1)',
        'x = Vi(3)',
        'y = Vj(3)',
        's = Pm(1)',
        dtype=get_dtype_str()
    )
    result_torch = my_conv_torch(
        torch.tensor(x_np, device='cuda', dtype=torch_dtype),
        torch.tensor(y_np, device='cuda', dtype=torch_dtype),
        torch.tensor(sigma_np, device='cuda', dtype=torch_dtype)
    ).cpu().numpy()

    return compare_arrays(result_jax, result_torch, rtol=RTOL, atol=ATOL)


# =============================================================================
# Main
# =============================================================================

def main():
    print_header("KeOps JAX Helper Function Tests",
                 "Testing generic_sum, generic_logsumexp, etc.")
    print_environment_info()

    suite = TestSuite("Helper Functions", "Convenience wrappers for Genred")

    # Basic functionality
    print_subheader("1. generic_sum")
    run_test("generic_sum: Basic SqDist", test_generic_sum_basic, suite)
    run_test("generic_sum: Gaussian kernel", test_generic_sum_gaussian, suite)
    run_test("generic_sum: Vj output (axis=0)", test_generic_sum_axis_vj, suite)
    run_test("generic_sum: Various dimensions", test_generic_sum_various_dims, suite)

    # Other reductions
    print_subheader("2. Other Reduction Types")
    run_test("generic_logsumexp", test_generic_logsumexp, suite)
    run_test("generic_argmin", test_generic_argmin, suite)
    run_test("generic_argkmin (K=5)", test_generic_argkmin, suite)
    run_test("generic_min", test_generic_min, suite)
    run_test("generic_max", test_generic_max, suite)

    # Dimension parsing
    print_subheader("3. Dimension Parsing")
    run_test("Output spec parsing", test_dimension_parsing, suite)
    run_test("With Pm parameter", test_with_pm_parameter, suite)

    # Print summary
    suite.print_summary()

    return 0 if suite.all_passed() else 1


if __name__ == "__main__":
    sys.exit(main())
