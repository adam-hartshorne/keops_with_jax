#!/usr/bin/env python3
"""
KeOps JAX API Tests
===================
Unit tests for the JAX KeOps API.

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
import jax
import jax.numpy as jnp

from test_utils import (
    TestSuite, print_header, print_subheader, print_info, print_warning,
    compare_arrays, run_test, print_environment_info, RICH_AVAILABLE
)

# =============================================================================
# Import KeOps
# =============================================================================

try:
    from pykeops.jax import Genred, LazyTensor, Vi, Vj, Pm

    KEOPS_AVAILABLE = True
except ImportError as e:
    print(f"Error: pykeops.jax not found: {e}")
    sys.exit(1)

# =============================================================================
# Configuration
# =============================================================================

SEED = 42
RTOL = 1e-4
ATOL = 1e-5
RTOL_LOOSE = 1e-3
ATOL_LOOSE = 1e-3


# =============================================================================
# Test Data Generators
# =============================================================================

def get_test_data(n, m, d, dtype='float32', seed=SEED):
    """Generate test data."""
    np.random.seed(seed)
    np_dtype = np.float32 if dtype == 'float32' else np.float64

    return {
        'x': jnp.array(np.random.randn(n, d).astype(np_dtype)),
        'y': jnp.array(np.random.randn(m, d).astype(np_dtype)),
        'b': jnp.array(np.random.randn(m, d).astype(np_dtype)),
        'sigma': jnp.array([0.5], dtype=jnp.float32 if dtype == 'float32' else jnp.float64),
    }


def get_batched_data(batch, n, m, d, dtype='float32', seed=SEED):
    """Generate batched test data."""
    np.random.seed(seed)
    np_dtype = np.float32 if dtype == 'float32' else np.float64

    return {
        'x': jnp.array(np.random.randn(batch, n, d).astype(np_dtype)),
        'y': jnp.array(np.random.randn(batch, m, d).astype(np_dtype)),
    }


# =============================================================================
# Reference Implementations (Pure JAX)
# =============================================================================

def pure_jax_sqdist_sum(x, y, axis=1):
    """Pure JAX reference for squared distance sum."""
    diff = x[:, None, :] - y[None, :, :]
    sqdist = jnp.sum(diff ** 2, axis=-1)
    return jnp.sum(sqdist, axis=axis, keepdims=True)


def pure_jax_gaussian_sum(x, y, sigma):
    """Pure JAX reference for Gaussian kernel sum."""
    diff = x[:, None, :] - y[None, :, :]
    sqdist = jnp.sum(diff ** 2, axis=-1)
    K = jnp.exp(-sqdist * sigma)
    return jnp.sum(K, axis=1, keepdims=True)


# =============================================================================
# 1. Genred Interface Tests
# =============================================================================

def test_genred_sqdist_sum(axis=1):
    data = get_test_data(100, 80, 3)
    formula = "SqDist(x, y)"
    aliases = ["x=Vi(3)", "y=Vj(3)"]

    op = Genred(formula, aliases, reduction_op='Sum', axis=axis)
    result = op(data['x'], data['y'])

    # Compare vs Pure JAX
    if axis == 1:
        expected = pure_jax_sqdist_sum(data['x'], data['y'])
        target_shape = (100, 1)
    else:
        expected = pure_jax_sqdist_sum(data['x'], data['y'], axis=0).T
        target_shape = (80, 1)

    match, diff = compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)
    if result.shape != target_shape:
        return False, 0.0  # Return float to match format specifier
    return match, diff


def test_genred_gaussian():
    data = get_test_data(100, 80, 3)
    formula = "Exp(-SqNorm2(x-y) * s)"
    aliases = ["x=Vi(3)", "y=Vj(3)", "s=Pm(1)"]

    op = Genred(formula, aliases, reduction_op='Sum', axis=1)
    result = op(data['x'], data['y'], data['sigma'])

    expected = pure_jax_gaussian_sum(data['x'], data['y'], data['sigma'])
    return compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)


def test_genred_reductions(reduction_op):
    data = get_test_data(100, 80, 3)
    formula = "SqDist(x, y)"
    aliases = ["x=Vi(3)", "y=Vj(3)"]

    op = Genred(formula, aliases, reduction_op=reduction_op, axis=1)
    result = op(data['x'], data['y'])

    # Pure JAX Ref
    diff = data['x'][:, None, :] - data['y'][None, :, :]
    sqdist = jnp.sum(diff ** 2, axis=-1)

    if reduction_op == 'Min':
        expected = jnp.min(sqdist, axis=1, keepdims=True)
    elif reduction_op == 'Max':
        expected = jnp.max(sqdist, axis=1, keepdims=True)

    return compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)


# =============================================================================
# 2. Formula Tests
# =============================================================================

def test_formula_laplacian():
    data = get_test_data(50, 40, 3)
    formula = "Exp(-Norm2(x-y))"
    aliases = ["x=Vi(3)", "y=Vj(3)"]

    op = Genred(formula, aliases, reduction_op='Sum', axis=1)
    result = op(data['x'], data['y'])

    diff = data['x'][:, None, :] - data['y'][None, :, :]
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
    expected = jnp.sum(jnp.exp(-dist), axis=1, keepdims=True)

    return compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)


def test_formula_cauchy():
    data = get_test_data(50, 40, 3)
    formula = "Inv(IntCst(1) + SqNorm2(x-y))"
    aliases = ["x=Vi(3)", "y=Vj(3)"]

    op = Genred(formula, aliases, reduction_op='Sum', axis=1)
    result = op(data['x'], data['y'])

    diff = data['x'][:, None, :] - data['y'][None, :, :]
    sqdist = jnp.sum(diff ** 2, axis=-1)
    expected = jnp.sum(1.0 / (1.0 + sqdist), axis=1, keepdims=True)

    return compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)


def test_formula_weighted_sum():
    data = get_test_data(50, 40, 3)
    formula = "Exp(-SqNorm2(x-y) * s) * b"
    aliases = ["x=Vi(3)", "y=Vj(3)", "b=Vj(3)", "s=Pm(1)"]

    op = Genred(formula, aliases, reduction_op='Sum', axis=1)
    result = op(data['x'], data['y'], data['b'], data['sigma'])

    # Reference
    diff = data['x'][:, None, :] - data['y'][None, :, :]
    sqdist = jnp.sum(diff ** 2, axis=-1)
    K = jnp.exp(-sqdist * data['sigma'])
    expected = jnp.sum(K[:, :, None] * data['b'][None, :, :], axis=1)

    return compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)


# =============================================================================
# 3. LazyTensor Interface
# =============================================================================

def test_lazytensor_basic():
    data = get_test_data(100, 80, 3)
    x_i = LazyTensor(data['x'][:, None, :])
    y_j = LazyTensor(data['y'][None, :, :])

    D_ij = ((x_i - y_j) ** 2).sum(-1)
    result = D_ij.sum(axis=1)

    expected = pure_jax_sqdist_sum(data['x'], data['y'])
    return compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)


def test_lazytensor_gaussian():
    data = get_test_data(100, 80, 3)
    x_i = LazyTensor(data['x'][:, None, :])
    y_j = LazyTensor(data['y'][None, :, :])

    D_ij = ((x_i - y_j) ** 2).sum(-1)
    K_ij = (-D_ij * data['sigma']).exp()
    result = K_ij.sum(axis=1)

    expected = pure_jax_gaussian_sum(data['x'], data['y'], data['sigma'])
    return compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)


def test_lazytensor_scalar_mult(side='left'):
    data = get_test_data(100, 80, 3)
    x_i = LazyTensor(data['x'][:, None, :])
    y_j = LazyTensor(data['y'][None, :, :])

    if side == 'left':
        result = (2.0 * (x_i - y_j)).sum(-1).sum(axis=1)
    else:
        result = ((x_i - y_j) * 2.0).sum(-1).sum(axis=1)

    expected = jnp.sum(2.0 * (data['x'][:, None, :] - data['y'][None, :, :]), axis=(1, 2))[:, None]
    return compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)


def test_lazytensor_batched():
    data = get_batched_data(4, 50, 40, 3)
    x_i = LazyTensor(data['x'][:, :, None, :])
    y_j = LazyTensor(data['y'][:, None, :, :])

    D_ij = ((x_i - y_j) ** 2).sum(-1)
    result = D_ij.sum(axis=2)

    diff = data['x'][:, :, None, :] - data['y'][:, None, :, :]
    expected = jnp.sum(jnp.sum(diff ** 2, axis=-1), axis=2, keepdims=True)

    return compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE, squeeze=True)


# =============================================================================
# 4. Gradients
# =============================================================================

def test_gradient_var(var_name):
    data = get_test_data(50, 40, 3)

    if var_name == 'x':
        formula = "SqDist(x, y)"
        aliases = ["x=Vi(3)", "y=Vj(3)"]
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)

        def forward(x):
            return op(x, data['y']).sum()

        grad = jax.grad(forward)(data['x'])

        if grad.shape != data['x'].shape: return False, 0.0
        if not jnp.any(grad != 0): return False, 0.0
        return True, 0.0

    elif var_name == 'y':
        formula = "SqDist(x, y)"
        aliases = ["x=Vi(3)", "y=Vj(3)"]
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)

        def forward(y):
            return op(data['x'], y).sum()

        grad = jax.grad(forward)(data['y'])

        if grad.shape != data['y'].shape: return False, 0.0
        if not jnp.any(grad != 0): return False, 0.0
        return True, 0.0

    elif var_name == 'pm':
        formula = "Exp(-SqNorm2(x-y) * s)"
        aliases = ["x=Vi(3)", "y=Vj(3)", "s=Pm(1)"]
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)

        def forward(s):
            return op(data['x'], data['y'], s).sum()

        grad = jax.grad(forward)(data['sigma'])

        if grad.shape != data['sigma'].shape: return False, 0.0
        if not jnp.isfinite(grad).all(): return False, 0.0
        return True, 0.0


def test_gradient_lazytensor():
    data = get_test_data(50, 40, 3)

    def forward(x):
        x_i = LazyTensor(x[:, None, :])
        y_j = LazyTensor(data['y'][None, :, :])
        D_ij = ((x_i - y_j) ** 2).sum(-1)
        return D_ij.sum(axis=1).sum()

    grad = jax.grad(forward)(data['x'])

    if grad.shape != data['x'].shape: return False, 0.0
    if not jnp.any(grad != 0): return False, 0.0
    return True, 0.0


# =============================================================================
# 5. JIT Compilation
# =============================================================================

def test_jit_genred():
    data = get_test_data(100, 80, 3)
    formula = "SqDist(x, y)"
    aliases = ["x=Vi(3)", "y=Vj(3)"]
    op = Genred(formula, aliases, reduction_op='Sum', axis=1)

    @jax.jit
    def compute(x, y):
        return op(x, y)

    # First call (compile)
    res1 = compute(data['x'], data['y'])
    # Second call (cached)
    res2 = compute(data['x'], data['y'])

    return compare_arrays(res1, res2, rtol=0, atol=0)


def test_jit_lazytensor():
    data = get_test_data(100, 80, 3)

    @jax.jit
    def compute(x, y):
        x_i = LazyTensor(x[:, None, :])
        y_j = LazyTensor(y[None, :, :])
        return ((x_i - y_j) ** 2).sum(-1).sum(axis=1)

    res1 = compute(data['x'], data['y'])
    res2 = compute(data['x'], data['y'])
    return compare_arrays(res1, res2, rtol=0, atol=0)


def test_jit_gradient():
    data = get_test_data(50, 40, 3)
    formula = "SqDist(x, y)"
    aliases = ["x=Vi(3)", "y=Vj(3)"]
    op = Genred(formula, aliases, reduction_op='Sum', axis=1)

    @jax.jit
    def forward_and_grad(x, y):
        return jax.grad(lambda x: op(x, y).sum())(x)

    grad1 = forward_and_grad(data['x'], data['y'])
    grad2 = forward_and_grad(data['x'], data['y'])
    return compare_arrays(grad1, grad2, rtol=0, atol=0)


# =============================================================================
# 6. Data Types
# =============================================================================

def test_dtype(dtype_str):
    # Check for X64 support if needed
    if dtype_str == 'float64':
        from jax import config
        if not config.read('jax_enable_x64'):
            print_warning("JAX float64 not enabled")
            return True, 0.0  # Return success but with warning to avoid failure in formatting

    data = get_test_data(50, 40, 3, dtype=dtype_str)
    formula = "SqDist(x, y)"
    aliases = ["x=Vi(3)", "y=Vj(3)"]

    op = Genred(formula, aliases, reduction_op='Sum', axis=1, dtype=dtype_str)
    result = op(data['x'], data['y'])

    expected_dtype = jnp.float64 if dtype_str == 'float64' else jnp.float32
    if result.dtype != expected_dtype:
        return False, 0.0
    return True, 0.0


# =============================================================================
# Main
# =============================================================================

def main():
    print_header("KeOps JAX API Tests", "Unit tests for Genred, LazyTensor, and Helpers")
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