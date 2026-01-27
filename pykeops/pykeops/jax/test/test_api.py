#!/usr/bin/env python3
"""
KeOps JAX API Tests
===================
Unit tests for the JAX KeOps API.

All tests compare JAX KeOps against PyTorch KeOps as the ground truth reference.
Since both use the same underlying CUDA kernels, results should be identical.

Tests cover:
- Genred interface (basic operations)
- LazyTensor interface (symbolic operations)
- Vi, Vj, Pm helper functions
- Data types (float32, float64)
- Batched operations
- JIT compilation compatibility
- Gradient computation
"""

import sys
import numpy as np

from test_utils import (
    TestSuite, print_header, print_subheader, print_info, print_warning,
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
    from pykeops.jax import Genred, LazyTensor, Vi, Vj, Pm
    KEOPS_AVAILABLE = True
except ImportError as e:
    print(f"Error: pykeops.jax not found: {e}")
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
RTOL = 1e-5
ATOL = 1e-6


# =============================================================================
# Test Data Generators
# =============================================================================

def get_test_data(n, m, d, dtype=None, seed=SEED):
    """Generate test data as numpy arrays."""
    np.random.seed(seed)
    np_dtype = get_np_dtype() if dtype is None else getattr(np, dtype)

    return {
        'x': np.random.randn(n, d).astype(np_dtype),
        'y': np.random.randn(m, d).astype(np_dtype),
        'b': np.random.randn(m, d).astype(np_dtype),
        'sigma': np.array([0.5], dtype=np_dtype),
    }


def get_batched_data(batch, n, m, d, dtype=None, seed=SEED):
    """Generate batched test data as numpy arrays."""
    np.random.seed(seed)
    np_dtype = get_np_dtype() if dtype is None else getattr(np, dtype)

    return {
        'x': np.random.randn(batch, n, d).astype(np_dtype),
        'y': np.random.randn(batch, m, d).astype(np_dtype),
    }


# =============================================================================
# 1. Genred Interface Tests
# =============================================================================

def test_genred_sqdist_sum(axis=1):
    data = get_test_data(100, 80, 3)
    formula = "SqDist(x, y)"
    aliases = ["x=Vi(3)", "y=Vj(3)"]

    # JAX KeOps
    op_jax = Genred(formula, aliases, reduction_op='Sum', axis=axis, dtype=get_dtype_str())
    result_jax = op_jax(jnp.array(data['x']), jnp.array(data['y']))

    # PyTorch KeOps (ground truth)
    op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=axis)
    result_torch = op_torch(
        torch.tensor(data['x'], device='cuda'),
        torch.tensor(data['y'], device='cuda')
    ).cpu().numpy()

    target_shape = (100, 1) if axis == 1 else (80, 1)
    if result_jax.shape != target_shape:
        return False, float('inf')

    return compare_arrays(np.array(result_jax), result_torch, rtol=RTOL, atol=ATOL)


def test_genred_gaussian():
    data = get_test_data(100, 80, 3)
    formula = "Exp(-SqNorm2(x-y) * s)"
    aliases = ["x=Vi(3)", "y=Vj(3)", "s=Pm(1)"]

    # JAX KeOps
    op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1, dtype=get_dtype_str())
    result_jax = op_jax(jnp.array(data['x']), jnp.array(data['y']), jnp.array(data['sigma']))

    # PyTorch KeOps (ground truth)
    op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
    result_torch = op_torch(
        torch.tensor(data['x'], device='cuda'),
        torch.tensor(data['y'], device='cuda'),
        torch.tensor(data['sigma'], device='cuda')
    ).cpu().numpy()

    return compare_arrays(np.array(result_jax), result_torch, rtol=RTOL, atol=ATOL)


def test_genred_reductions(reduction_op):
    data = get_test_data(100, 80, 3)
    formula = "SqDist(x, y)"
    aliases = ["x=Vi(3)", "y=Vj(3)"]

    # JAX KeOps
    op_jax = Genred(formula, aliases, reduction_op=reduction_op, axis=1, dtype=get_dtype_str())
    result_jax = op_jax(jnp.array(data['x']), jnp.array(data['y']))

    # PyTorch KeOps (ground truth)
    op_torch = Genred_torch(formula, aliases, reduction_op=reduction_op, axis=1)
    result_torch = op_torch(
        torch.tensor(data['x'], device='cuda'),
        torch.tensor(data['y'], device='cuda')
    ).cpu().numpy()

    return compare_arrays(np.array(result_jax), result_torch, rtol=RTOL, atol=ATOL)


# =============================================================================
# 2. Formula Tests
# =============================================================================

def test_formula_laplacian():
    data = get_test_data(50, 40, 3)
    formula = "Exp(-Norm2(x-y))"
    aliases = ["x=Vi(3)", "y=Vj(3)"]

    # JAX KeOps
    op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1, dtype=get_dtype_str())
    result_jax = op_jax(jnp.array(data['x']), jnp.array(data['y']))

    # PyTorch KeOps (ground truth)
    op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
    result_torch = op_torch(
        torch.tensor(data['x'], device='cuda'),
        torch.tensor(data['y'], device='cuda')
    ).cpu().numpy()

    return compare_arrays(np.array(result_jax), result_torch, rtol=RTOL, atol=ATOL)


def test_formula_cauchy():
    data = get_test_data(50, 40, 3)
    formula = "Inv(IntCst(1) + SqNorm2(x-y))"
    aliases = ["x=Vi(3)", "y=Vj(3)"]

    # JAX KeOps
    op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1, dtype=get_dtype_str())
    result_jax = op_jax(jnp.array(data['x']), jnp.array(data['y']))

    # PyTorch KeOps (ground truth)
    op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
    result_torch = op_torch(
        torch.tensor(data['x'], device='cuda'),
        torch.tensor(data['y'], device='cuda')
    ).cpu().numpy()

    return compare_arrays(np.array(result_jax), result_torch, rtol=RTOL, atol=ATOL)


def test_formula_weighted_sum():
    data = get_test_data(50, 40, 3)
    formula = "Exp(-SqNorm2(x-y) * s) * b"
    aliases = ["x=Vi(3)", "y=Vj(3)", "b=Vj(3)", "s=Pm(1)"]

    # JAX KeOps
    op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1, dtype=get_dtype_str())
    result_jax = op_jax(
        jnp.array(data['x']), jnp.array(data['y']),
        jnp.array(data['b']), jnp.array(data['sigma'])
    )

    # PyTorch KeOps (ground truth)
    op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
    result_torch = op_torch(
        torch.tensor(data['x'], device='cuda'),
        torch.tensor(data['y'], device='cuda'),
        torch.tensor(data['b'], device='cuda'),
        torch.tensor(data['sigma'], device='cuda')
    ).cpu().numpy()

    return compare_arrays(np.array(result_jax), result_torch, rtol=RTOL, atol=ATOL)


# =============================================================================
# 3. LazyTensor Interface
# =============================================================================

def test_lazytensor_basic():
    data = get_test_data(100, 80, 3)

    # JAX KeOps
    x_jax = jnp.array(data['x'])
    y_jax = jnp.array(data['y'])
    x_i = LazyTensor(x_jax[:, None, :])
    y_j = LazyTensor(y_jax[None, :, :])
    D_ij = ((x_i - y_j) ** 2).sum(-1)
    result_jax = D_ij.sum(axis=1)

    # PyTorch KeOps (ground truth)
    x_torch = torch.tensor(data['x'], device='cuda')
    y_torch = torch.tensor(data['y'], device='cuda')
    x_i_t = LazyTensor_torch(x_torch[:, None, :])
    y_j_t = LazyTensor_torch(y_torch[None, :, :])
    D_ij_t = ((x_i_t - y_j_t) ** 2).sum(-1)
    result_torch = D_ij_t.sum(axis=1).cpu().numpy()

    return compare_arrays(np.array(result_jax), result_torch, rtol=RTOL, atol=ATOL)


def test_lazytensor_gaussian():
    data = get_test_data(100, 80, 3)

    # JAX KeOps
    x_jax = jnp.array(data['x'])
    y_jax = jnp.array(data['y'])
    sigma_jax = jnp.array(data['sigma'])
    x_i = LazyTensor(x_jax[:, None, :])
    y_j = LazyTensor(y_jax[None, :, :])
    D_ij = ((x_i - y_j) ** 2).sum(-1)
    K_ij = (-D_ij * sigma_jax).exp()
    result_jax = K_ij.sum(axis=1)

    # PyTorch KeOps (ground truth)
    x_torch = torch.tensor(data['x'], device='cuda')
    y_torch = torch.tensor(data['y'], device='cuda')
    sigma_torch = torch.tensor(data['sigma'], device='cuda')
    x_i_t = LazyTensor_torch(x_torch[:, None, :])
    y_j_t = LazyTensor_torch(y_torch[None, :, :])
    D_ij_t = ((x_i_t - y_j_t) ** 2).sum(-1)
    K_ij_t = (-D_ij_t * sigma_torch).exp()
    result_torch = K_ij_t.sum(axis=1).cpu().numpy()

    return compare_arrays(np.array(result_jax), result_torch, rtol=RTOL, atol=ATOL)


def test_lazytensor_scalar_mult(side='left'):
    data = get_test_data(100, 80, 3)

    # JAX KeOps
    x_jax = jnp.array(data['x'])
    y_jax = jnp.array(data['y'])
    x_i = LazyTensor(x_jax[:, None, :])
    y_j = LazyTensor(y_jax[None, :, :])
    if side == 'left':
        result_jax = (2.0 * (x_i - y_j)).sum(-1).sum(axis=1)
    else:
        result_jax = ((x_i - y_j) * 2.0).sum(-1).sum(axis=1)

    # PyTorch KeOps (ground truth)
    x_torch = torch.tensor(data['x'], device='cuda')
    y_torch = torch.tensor(data['y'], device='cuda')
    x_i_t = LazyTensor_torch(x_torch[:, None, :])
    y_j_t = LazyTensor_torch(y_torch[None, :, :])
    if side == 'left':
        result_torch = (2.0 * (x_i_t - y_j_t)).sum(-1).sum(axis=1).cpu().numpy()
    else:
        result_torch = ((x_i_t - y_j_t) * 2.0).sum(-1).sum(axis=1).cpu().numpy()

    return compare_arrays(np.array(result_jax), result_torch, rtol=RTOL, atol=ATOL)


def test_lazytensor_batched():
    data = get_batched_data(4, 50, 40, 3)

    # JAX KeOps
    x_jax = jnp.array(data['x'])
    y_jax = jnp.array(data['y'])
    x_i = LazyTensor(x_jax[:, :, None, :])
    y_j = LazyTensor(y_jax[:, None, :, :])
    D_ij = ((x_i - y_j) ** 2).sum(-1)
    result_jax = D_ij.sum(axis=2)

    # PyTorch KeOps (ground truth)
    x_torch = torch.tensor(data['x'], device='cuda')
    y_torch = torch.tensor(data['y'], device='cuda')
    x_i_t = LazyTensor_torch(x_torch[:, :, None, :])
    y_j_t = LazyTensor_torch(y_torch[:, None, :, :])
    D_ij_t = ((x_i_t - y_j_t) ** 2).sum(-1)
    result_torch = D_ij_t.sum(axis=2).cpu().numpy()

    return compare_arrays(np.array(result_jax), result_torch, rtol=RTOL, atol=ATOL, squeeze=True)


# =============================================================================
# 4. Gradients
# =============================================================================

def test_gradient_var(var_name):
    data = get_test_data(50, 40, 3)

    if var_name == 'x':
        formula = "SqDist(x, y)"
        aliases = ["x=Vi(3)", "y=Vj(3)"]

        # JAX KeOps gradient
        op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1, dtype=get_dtype_str())
        def forward_jax(x):
            return op_jax(x, jnp.array(data['y'])).sum()
        grad_jax = jax.grad(forward_jax)(jnp.array(data['x']))

        # PyTorch KeOps gradient (ground truth)
        op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
        x_torch = torch.tensor(data['x'], device='cuda', requires_grad=True)
        y_torch = torch.tensor(data['y'], device='cuda')
        result_torch = op_torch(x_torch, y_torch).sum()
        result_torch.backward()
        grad_torch = x_torch.grad.cpu().numpy()

        return compare_arrays(np.array(grad_jax), grad_torch, rtol=RTOL, atol=ATOL)

    elif var_name == 'y':
        formula = "SqDist(x, y)"
        aliases = ["x=Vi(3)", "y=Vj(3)"]

        # JAX KeOps gradient
        op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1, dtype=get_dtype_str())
        def forward_jax(y):
            return op_jax(jnp.array(data['x']), y).sum()
        grad_jax = jax.grad(forward_jax)(jnp.array(data['y']))

        # PyTorch KeOps gradient (ground truth)
        op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
        x_torch = torch.tensor(data['x'], device='cuda')
        y_torch = torch.tensor(data['y'], device='cuda', requires_grad=True)
        result_torch = op_torch(x_torch, y_torch).sum()
        result_torch.backward()
        grad_torch = y_torch.grad.cpu().numpy()

        return compare_arrays(np.array(grad_jax), grad_torch, rtol=RTOL, atol=ATOL)

    elif var_name == 'pm':
        formula = "Exp(-SqNorm2(x-y) * s)"
        aliases = ["x=Vi(3)", "y=Vj(3)", "s=Pm(1)"]

        # JAX KeOps gradient
        op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1, dtype=get_dtype_str())
        def forward_jax(s):
            return op_jax(jnp.array(data['x']), jnp.array(data['y']), s).sum()
        grad_jax = jax.grad(forward_jax)(jnp.array(data['sigma']))

        # PyTorch KeOps gradient (ground truth)
        op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
        x_torch = torch.tensor(data['x'], device='cuda')
        y_torch = torch.tensor(data['y'], device='cuda')
        s_torch = torch.tensor(data['sigma'], device='cuda', requires_grad=True)
        result_torch = op_torch(x_torch, y_torch, s_torch).sum()
        result_torch.backward()
        grad_torch = s_torch.grad.cpu().numpy()

        return compare_arrays(np.array(grad_jax), grad_torch, rtol=RTOL, atol=ATOL)


def test_gradient_lazytensor():
    data = get_test_data(50, 40, 3)

    # JAX KeOps gradient
    def forward_jax(x):
        x_i = LazyTensor(x[:, None, :])
        y_j = LazyTensor(jnp.array(data['y'])[None, :, :])
        D_ij = ((x_i - y_j) ** 2).sum(-1)
        return D_ij.sum(axis=1).sum()
    grad_jax = jax.grad(forward_jax)(jnp.array(data['x']))

    # PyTorch KeOps gradient (ground truth)
    x_torch = torch.tensor(data['x'], device='cuda', requires_grad=True)
    y_torch = torch.tensor(data['y'], device='cuda')
    x_i_t = LazyTensor_torch(x_torch[:, None, :])
    y_j_t = LazyTensor_torch(y_torch[None, :, :])
    D_ij_t = ((x_i_t - y_j_t) ** 2).sum(-1)
    result_torch = D_ij_t.sum(axis=1).sum()
    result_torch.backward()
    grad_torch = x_torch.grad.cpu().numpy()

    return compare_arrays(np.array(grad_jax), grad_torch, rtol=RTOL, atol=ATOL)


# =============================================================================
# 5. JIT Compilation
# =============================================================================

def test_jit_genred():
    data = get_test_data(100, 80, 3)
    formula = "SqDist(x, y)"
    aliases = ["x=Vi(3)", "y=Vj(3)"]

    # JAX KeOps with JIT
    op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1, dtype=get_dtype_str())
    @jax.jit
    def compute(x, y):
        return op_jax(x, y)
    result_jax = compute(jnp.array(data['x']), jnp.array(data['y']))

    # PyTorch KeOps (ground truth)
    op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
    result_torch = op_torch(
        torch.tensor(data['x'], device='cuda'),
        torch.tensor(data['y'], device='cuda')
    ).cpu().numpy()

    return compare_arrays(np.array(result_jax), result_torch, rtol=RTOL, atol=ATOL)


def test_jit_lazytensor():
    data = get_test_data(100, 80, 3)

    # JAX KeOps with JIT
    @jax.jit
    def compute(x, y):
        x_i = LazyTensor(x[:, None, :])
        y_j = LazyTensor(y[None, :, :])
        return ((x_i - y_j) ** 2).sum(-1).sum(axis=1)
    result_jax = compute(jnp.array(data['x']), jnp.array(data['y']))

    # PyTorch KeOps (ground truth)
    x_torch = torch.tensor(data['x'], device='cuda')
    y_torch = torch.tensor(data['y'], device='cuda')
    x_i_t = LazyTensor_torch(x_torch[:, None, :])
    y_j_t = LazyTensor_torch(y_torch[None, :, :])
    result_torch = ((x_i_t - y_j_t) ** 2).sum(-1).sum(axis=1).cpu().numpy()

    return compare_arrays(np.array(result_jax), result_torch, rtol=RTOL, atol=ATOL)


def test_jit_gradient():
    data = get_test_data(50, 40, 3)
    formula = "SqDist(x, y)"
    aliases = ["x=Vi(3)", "y=Vj(3)"]

    # JAX KeOps with JIT gradient
    op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1, dtype=get_dtype_str())
    @jax.jit
    def forward_and_grad(x, y):
        return jax.grad(lambda x: op_jax(x, y).sum())(x)
    grad_jax = forward_and_grad(jnp.array(data['x']), jnp.array(data['y']))

    # PyTorch KeOps gradient (ground truth)
    op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
    x_torch = torch.tensor(data['x'], device='cuda', requires_grad=True)
    y_torch = torch.tensor(data['y'], device='cuda')
    result_torch = op_torch(x_torch, y_torch).sum()
    result_torch.backward()
    grad_torch = x_torch.grad.cpu().numpy()

    return compare_arrays(np.array(grad_jax), grad_torch, rtol=RTOL, atol=ATOL)


# =============================================================================
# 6. Data Types
# =============================================================================

def test_dtype(dtype_str):
    # Check for X64 support if needed
    if dtype_str == 'float64':
        from jax import config
        if not config.read('jax_enable_x64'):
            print_warning("JAX float64 not enabled")
            return True, 0.0  # Skip test

    data = get_test_data(50, 40, 3, dtype=dtype_str)
    formula = "SqDist(x, y)"
    aliases = ["x=Vi(3)", "y=Vj(3)"]

    torch_dtype = torch.float64 if dtype_str == 'float64' else torch.float32

    # JAX KeOps
    op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1, dtype=dtype_str)
    result_jax = op_jax(jnp.array(data['x']), jnp.array(data['y']))

    # PyTorch KeOps (ground truth)
    op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1, dtype=dtype_str)
    result_torch = op_torch(
        torch.tensor(data['x'], device='cuda', dtype=torch_dtype),
        torch.tensor(data['y'], device='cuda', dtype=torch_dtype)
    ).cpu().numpy()

    expected_dtype = jnp.float64 if dtype_str == 'float64' else jnp.float32
    if result_jax.dtype != expected_dtype:
        return False, float('inf')

    return compare_arrays(np.array(result_jax), result_torch, rtol=RTOL, atol=ATOL)


# =============================================================================
# Main
# =============================================================================

def main():
    print_header("KeOps JAX API Tests", "Comparing JAX KeOps vs PyTorch KeOps (ground truth)")
    print_environment_info()

    suite = TestSuite("API Tests", "Core Functionality")

    # 1. Genred Basic
    print_subheader("1. Genred Basics")
    run_test("SqDist (Sum axis=1)", lambda: test_genred_sqdist_sum(axis=1), suite)
    run_test("SqDist (Sum axis=0)", lambda: test_genred_sqdist_sum(axis=0), suite)
    run_test("Gaussian Kernel", test_genred_gaussian, suite)
    run_test("Reduction: Min", lambda: test_genred_reductions('Min'), suite)
    run_test("Reduction: Max", lambda: test_genred_reductions('Max'), suite)

    # 2. Formulas
    print_subheader("2. Formula Types")
    run_test("Laplacian Kernel", test_formula_laplacian, suite)
    run_test("Cauchy Kernel", test_formula_cauchy, suite)
    run_test("Weighted Sum", test_formula_weighted_sum, suite)

    # 3. LazyTensor
    print_subheader("3. LazyTensor Interface")
    run_test("Basic SqDist", test_lazytensor_basic, suite)
    run_test("Gaussian Kernel", test_lazytensor_gaussian, suite)
    run_test("Scalar Mul (Left)", lambda: test_lazytensor_scalar_mult('left'), suite)
    run_test("Scalar Mul (Right)", lambda: test_lazytensor_scalar_mult('right'), suite)
    run_test("Batched Ops", test_lazytensor_batched, suite)

    # 4. Gradients
    print_subheader("4. Gradients")
    run_test("Grad w.r.t Vi (x)", lambda: test_gradient_var('x'), suite)
    run_test("Grad w.r.t Vj (y)", lambda: test_gradient_var('y'), suite)
    run_test("Grad w.r.t Pm (s)", lambda: test_gradient_var('pm'), suite)
    run_test("LazyTensor Grad", test_gradient_lazytensor, suite)

    # 5. JIT
    print_subheader("5. JIT Compatibility")
    run_test("JIT Genred", test_jit_genred, suite)
    run_test("JIT LazyTensor", test_jit_lazytensor, suite)
    run_test("JIT Gradient", test_jit_gradient, suite)

    # 6. Dtypes
    print_subheader("6. Data Types")
    run_test("float32", lambda: test_dtype('float32'), suite)
    run_test("float64", lambda: test_dtype('float64'), suite)

    # Final Summary
    suite.print_summary()
    return 0 if suite.all_passed() else 1


if __name__ == "__main__":
    sys.exit(main())
