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

Can run standalone or with pytest.
"""

import sys
import unittest
import numpy as np

import jax
import jax.numpy as jnp

from test_utils import (
    TestSuite, Status, print_header, print_subheader, print_info,
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
    KEOPS_AVAILABLE = False
    sys.exit(1)

# Optional PyTorch for comparison
try:
    import torch
    from pykeops.torch import Genred as Genred_torch, LazyTensor as LazyTensor_torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


# =============================================================================
# Configuration
# =============================================================================

SEED = 42
RTOL = 1e-4
ATOL = 1e-5
RTOL_LOOSE = 1e-3  # For comparing against pure JAX (different accumulation)
ATOL_LOOSE = 1e-3


# =============================================================================
# Test Fixtures
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
# Genred Interface Tests
# =============================================================================

class TestGenredBasic(unittest.TestCase):
    """Basic Genred functionality tests."""
    
    def setUp(self):
        self.data = get_test_data(100, 80, 3)
        self.D = 3
    
    def test_sqdist_sum(self):
        """Test SqDist formula with Sum reduction."""
        formula = "SqDist(x, y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)
        result = op(self.data['x'], self.data['y'])
        
        expected = pure_jax_sqdist_sum(self.data['x'], self.data['y'])
        match, diff = compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)
        
        self.assertEqual(result.shape, (100, 1))
        self.assertTrue(match, f"Max diff: {diff}")
    
    def test_sqdist_sum_axis0(self):
        """Test Sum reduction over axis 0."""
        formula = "SqDist(x, y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        
        op = Genred(formula, aliases, reduction_op='Sum', axis=0)
        result = op(self.data['x'], self.data['y'])
        
        expected = pure_jax_sqdist_sum(self.data['x'], self.data['y'], axis=0).T
        match, diff = compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)
        
        self.assertEqual(result.shape, (80, 1))
        self.assertTrue(match, f"Max diff: {diff}")
    
    def test_gaussian_kernel(self):
        """Test Gaussian kernel with Pm parameter."""
        formula = "Exp(-SqNorm2(x-y) * s)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})", "s=Pm(1)"]
        
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)
        result = op(self.data['x'], self.data['y'], self.data['sigma'])
        
        expected = pure_jax_gaussian_sum(self.data['x'], self.data['y'], self.data['sigma'])
        match, diff = compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)
        
        self.assertEqual(result.shape, (100, 1))
        self.assertTrue(match, f"Max diff: {diff}")
    
    def test_min_reduction(self):
        """Test Min reduction."""
        formula = "SqDist(x, y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        
        op = Genred(formula, aliases, reduction_op='Min', axis=1)
        result = op(self.data['x'], self.data['y'])
        
        # Pure JAX reference
        diff = self.data['x'][:, None, :] - self.data['y'][None, :, :]
        sqdist = jnp.sum(diff ** 2, axis=-1)
        expected = jnp.min(sqdist, axis=1, keepdims=True)
        
        match, diff = compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)
        self.assertTrue(match, f"Max diff: {diff}")
    
    def test_max_reduction(self):
        """Test Max reduction."""
        formula = "SqDist(x, y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        
        op = Genred(formula, aliases, reduction_op='Max', axis=1)
        result = op(self.data['x'], self.data['y'])
        
        diff = self.data['x'][:, None, :] - self.data['y'][None, :, :]
        sqdist = jnp.sum(diff ** 2, axis=-1)
        expected = jnp.max(sqdist, axis=1, keepdims=True)
        
        match, diff = compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)
        self.assertTrue(match, f"Max diff: {diff}")


class TestGenredFormulas(unittest.TestCase):
    """Test various formula types."""
    
    def setUp(self):
        self.data = get_test_data(50, 40, 3)
        self.D = 3
    
    def test_laplacian_kernel(self):
        """Test Laplacian kernel: exp(-|x-y|)."""
        formula = "Exp(-Norm2(x-y))"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)
        result = op(self.data['x'], self.data['y'])
        
        # Reference
        diff = self.data['x'][:, None, :] - self.data['y'][None, :, :]
        dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
        expected = jnp.sum(jnp.exp(-dist), axis=1, keepdims=True)
        
        match, diff = compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)
        self.assertTrue(match, f"Max diff: {diff}")
    
    def test_cauchy_kernel(self):
        """Test Cauchy kernel: 1/(1+|x-y|^2)."""
        formula = "Inv(IntCst(1) + SqNorm2(x-y))"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)
        result = op(self.data['x'], self.data['y'])
        
        diff = self.data['x'][:, None, :] - self.data['y'][None, :, :]
        sqdist = jnp.sum(diff ** 2, axis=-1)
        expected = jnp.sum(1.0 / (1.0 + sqdist), axis=1, keepdims=True)
        
        match, diff = compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)
        self.assertTrue(match, f"Max diff: {diff}")
    
    def test_weighted_sum(self):
        """Test weighted kernel sum: K(x,y) @ b."""
        formula = "Exp(-SqNorm2(x-y) * s) * b"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})", f"b=Vj({self.D})", "s=Pm(1)"]
        
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)
        result = op(self.data['x'], self.data['y'], self.data['b'], self.data['sigma'])
        
        # Reference
        diff = self.data['x'][:, None, :] - self.data['y'][None, :, :]
        sqdist = jnp.sum(diff ** 2, axis=-1)
        K = jnp.exp(-sqdist * self.data['sigma'])
        expected = jnp.sum(K[:, :, None] * self.data['b'][None, :, :], axis=1)
        
        match, diff = compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)
        self.assertEqual(result.shape, (50, 3))
        self.assertTrue(match, f"Max diff: {diff}")


# =============================================================================
# LazyTensor Interface Tests
# =============================================================================

class TestLazyTensor(unittest.TestCase):
    """LazyTensor interface tests."""
    
    def setUp(self):
        self.data = get_test_data(100, 80, 3)
    
    def test_basic_sqdist(self):
        """Test basic squared distance with LazyTensor."""
        x_i = LazyTensor(self.data['x'][:, None, :])
        y_j = LazyTensor(self.data['y'][None, :, :])
        
        D_ij = ((x_i - y_j) ** 2).sum(-1)
        result = D_ij.sum(axis=1)
        
        expected = pure_jax_sqdist_sum(self.data['x'], self.data['y'])
        match, diff = compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)
        self.assertTrue(match, f"Max diff: {diff}")
    
    def test_gaussian_kernel(self):
        """Test Gaussian kernel with LazyTensor."""
        x_i = LazyTensor(self.data['x'][:, None, :])
        y_j = LazyTensor(self.data['y'][None, :, :])
        
        D_ij = ((x_i - y_j) ** 2).sum(-1)
        K_ij = (-D_ij * self.data['sigma']).exp()
        result = K_ij.sum(axis=1)
        
        expected = pure_jax_gaussian_sum(self.data['x'], self.data['y'], self.data['sigma'])
        match, diff = compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)
        self.assertTrue(match, f"Max diff: {diff}")
    
    def test_scalar_multiplication_left(self):
        """Test scalar * LazyTensor (left multiplication)."""
        x_i = LazyTensor(self.data['x'][:, None, :])
        y_j = LazyTensor(self.data['y'][None, :, :])
        
        result = (2.0 * (x_i - y_j)).sum(-1).sum(axis=1)
        expected = jnp.sum(2.0 * (self.data['x'][:, None, :] - self.data['y'][None, :, :]), 
                         axis=(1, 2))[:, None]
        
        match, diff = compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)
        self.assertTrue(match, f"Max diff: {diff}")
    
    def test_scalar_multiplication_right(self):
        """Test LazyTensor * scalar (right multiplication)."""
        x_i = LazyTensor(self.data['x'][:, None, :])
        y_j = LazyTensor(self.data['y'][None, :, :])
        
        result = ((x_i - y_j) * 2.0).sum(-1).sum(axis=1)
        expected = jnp.sum(2.0 * (self.data['x'][:, None, :] - self.data['y'][None, :, :]), 
                         axis=(1, 2))[:, None]
        
        match, diff = compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)
        self.assertTrue(match, f"Max diff: {diff}")


class TestLazyTensorBatched(unittest.TestCase):
    """Batched LazyTensor tests."""
    
    def setUp(self):
        self.data = get_batched_data(4, 50, 40, 3)
    
    def test_batched_sqdist(self):
        """Test batched squared distance."""
        x_i = LazyTensor(self.data['x'][:, :, None, :])
        y_j = LazyTensor(self.data['y'][:, None, :, :])
        
        D_ij = ((x_i - y_j) ** 2).sum(-1)
        result = D_ij.sum(axis=2)
        
        # Reference
        diff = self.data['x'][:, :, None, :] - self.data['y'][:, None, :, :]
        expected = jnp.sum(jnp.sum(diff ** 2, axis=-1), axis=2, keepdims=True)
        
        match, diff = compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE, squeeze=True)
        self.assertTrue(match, f"Max diff: {diff}")


# =============================================================================
# Gradient Tests
# =============================================================================

class TestGradients(unittest.TestCase):
    """Gradient computation tests."""
    
    def setUp(self):
        self.data = get_test_data(50, 40, 3)
        self.D = 3
    
    def test_gradient_vi(self):
        """Test gradient w.r.t. Vi variable."""
        formula = "SqDist(x, y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)
        
        def forward(x):
            return op(x, self.data['y']).sum()
        
        grad = jax.grad(forward)(self.data['x'])
        
        # Check gradient is correct shape and non-zero
        self.assertEqual(grad.shape, self.data['x'].shape)
        self.assertTrue(jnp.any(grad != 0), "Gradient should be non-zero")
    
    def test_gradient_vj(self):
        """Test gradient w.r.t. Vj variable."""
        formula = "SqDist(x, y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)
        
        def forward(y):
            return op(self.data['x'], y).sum()
        
        grad = jax.grad(forward)(self.data['y'])
        
        self.assertEqual(grad.shape, self.data['y'].shape)
        self.assertTrue(jnp.any(grad != 0), "Gradient should be non-zero")
    
    def test_gradient_pm(self):
        """Test gradient w.r.t. Pm parameter."""
        formula = "Exp(-SqNorm2(x-y) * s)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})", "s=Pm(1)"]
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)
        
        def forward(s):
            return op(self.data['x'], self.data['y'], s).sum()
        
        grad = jax.grad(forward)(self.data['sigma'])
        
        self.assertEqual(grad.shape, self.data['sigma'].shape)
        self.assertTrue(jnp.isfinite(grad).all(), "Gradient should be finite")
    
    def test_gradient_lazytensor(self):
        """Test LazyTensor gradient."""
        def forward(x):
            x_i = LazyTensor(x[:, None, :])
            y_j = LazyTensor(self.data['y'][None, :, :])
            D_ij = ((x_i - y_j) ** 2).sum(-1)
            return D_ij.sum(axis=1).sum()
        
        grad = jax.grad(forward)(self.data['x'])
        
        self.assertEqual(grad.shape, self.data['x'].shape)
        self.assertTrue(jnp.any(grad != 0), "Gradient should be non-zero")


# =============================================================================
# JIT Compilation Tests
# =============================================================================

class TestJIT(unittest.TestCase):
    """JIT compilation tests."""
    
    def setUp(self):
        self.data = get_test_data(100, 80, 3)
        self.D = 3
    
    def test_jit_genred(self):
        """Test JIT with Genred."""
        formula = "SqDist(x, y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)
        
        @jax.jit
        def compute(x, y):
            return op(x, y)
        
        # First call (compile)
        result1 = compute(self.data['x'], self.data['y'])
        # Second call (cached)
        result2 = compute(self.data['x'], self.data['y'])
        
        match, diff = compare_arrays(result1, result2, rtol=0, atol=0)
        self.assertTrue(match)
    
    def test_jit_lazytensor(self):
        """Test JIT with LazyTensor."""
        @jax.jit
        def compute(x, y):
            x_i = LazyTensor(x[:, None, :])
            y_j = LazyTensor(y[None, :, :])
            return ((x_i - y_j) ** 2).sum(-1).sum(axis=1)
        
        result1 = compute(self.data['x'], self.data['y'])
        result2 = compute(self.data['x'], self.data['y'])
        
        match, diff = compare_arrays(result1, result2, rtol=0, atol=0)
        self.assertTrue(match)
    
    def test_jit_gradient(self):
        """Test JIT with gradient computation."""
        formula = "SqDist(x, y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)
        
        @jax.jit
        def forward_and_grad(x, y):
            return jax.grad(lambda x: op(x, y).sum())(x)
        
        grad1 = forward_and_grad(self.data['x'], self.data['y'])
        grad2 = forward_and_grad(self.data['x'], self.data['y'])
        
        match, diff = compare_arrays(grad1, grad2, rtol=0, atol=0)
        self.assertTrue(match)


# =============================================================================
# Vi/Vj/Pm Helper Tests
# =============================================================================

class TestHelpers(unittest.TestCase):
    """Test helper functions and LazyTensor creation."""
    
    def test_lazytensor_creation(self):
        """Test LazyTensor creation with different shapes."""
        data = get_test_data(50, 40, 3)
        
        # Create LazyTensors with explicit shapes
        x_i = LazyTensor(data['x'][:, None, :])  # (N, 1, D) - "i" indexed
        y_j = LazyTensor(data['y'][None, :, :])  # (1, M, D) - "j" indexed
        
        # Should be able to do operations
        D_ij = ((x_i - y_j) ** 2).sum(-1)
        result = D_ij.sum(axis=1)
        
        # Verify result is valid
        self.assertEqual(result.shape[0], 50)
        self.assertTrue(jnp.all(jnp.isfinite(result)))
    
    def test_genred_string_aliases(self):
        """Test Genred with string aliases."""
        data = get_test_data(50, 40, 3)
        
        formula = "Exp(-SqNorm2(x-y) * s)"
        aliases = ["x=Vi(3)", "y=Vj(3)", "s=Pm(1)"]
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)
        result = op(data['x'], data['y'], data['sigma'])
        
        # Verify result shape and validity
        self.assertEqual(result.shape, (50, 1))
        self.assertTrue(jnp.all(jnp.isfinite(result)))


# =============================================================================
# Data Type Tests
# =============================================================================

class TestDataTypes(unittest.TestCase):
    """Test different data types."""
    
    def test_float32(self):
        """Test float32 operations."""
        data = get_test_data(50, 40, 3, dtype='float32')
        
        formula = "SqDist(x, y)"
        aliases = ["x=Vi(3)", "y=Vj(3)"]
        op = Genred(formula, aliases, reduction_op='Sum', axis=1, dtype='float32')
        result = op(data['x'], data['y'])
        
        self.assertEqual(result.dtype, jnp.float32)
    
    def test_float64(self):
        """Test float64 operations (requires JAX_ENABLE_X64)."""
        # Check if float64 is enabled in JAX
        from jax import config
        if not config.read('jax_enable_x64'):
            self.skipTest("JAX float64 not enabled (set JAX_ENABLE_X64=True)")
        
        data = get_test_data(50, 40, 3, dtype='float64')
        
        formula = "SqDist(x, y)"
        aliases = ["x=Vi(3)", "y=Vj(3)"]
        op = Genred(formula, aliases, reduction_op='Sum', axis=1, dtype='float64')
        result = op(data['x'], data['y'])
        
        self.assertEqual(result.dtype, jnp.float64)


# =============================================================================
# Main
# =============================================================================

def run_tests():
    """Run all tests with custom output."""
    print_header("KeOps JAX API Tests", "Unit tests for Genred, LazyTensor, and helpers")
    print_environment_info()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestGenredBasic,
        TestGenredFormulas,
        TestLazyTensor,
        TestLazyTensorBatched,
        TestGradients,
        TestJIT,
        TestHelpers,
        TestDataTypes,
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print()
    if result.wasSuccessful():
        print(f"\n{'='*70}")
        print(f"{'ALL TESTS PASSED':^70}")
        print(f"{'='*70}\n")
        return 0
    else:
        print(f"\n{'='*70}")
        print(f"{'SOME TESTS FAILED':^70}")
        print(f"{'='*70}\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
