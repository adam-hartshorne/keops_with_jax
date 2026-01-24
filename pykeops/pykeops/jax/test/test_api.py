#!/usr/bin/env python3
"""
KeOps JAX API Tests
===================
Comprehensive unit tests for the JAX KeOps API.

Tests cover:
- Genred interface (all reduction operations)
- LazyTensor interface (operations and reductions)
- Vi, Vj, Pm helper functions
- Parameter handling
- Data types (float32, float64)
- Batched and unbatched operations
- JIT compilation
- Gradients
"""

import sys
import time
import unittest
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

# Import test utilities
from test_utils import (
    Colors, Status, TestResult, TestSuite,
    print_header, print_subheader, print_test_start, print_test_end,
    ASCIITable, TableColumn, compare_arrays, format_comparison_result
)

# Import KeOps JAX API
try:
    from pykeops.jax import Genred, LazyTensor, Vi, Vj, Pm
    KEOPS_AVAILABLE = True
except ImportError as e:
    print(f"{Colors.RED}Error: pykeops.jax not found: {e}{Colors.RESET}")
    KEOPS_AVAILABLE = False


# =============================================================================
# Test Configuration
# =============================================================================

# Random seed for reproducibility
SEED = 42

# Default tolerances
RTOL = 1e-4
ATOL = 1e-5

# Test sizes
SMALL_N = 50
SMALL_M = 30
MEDIUM_N = 500
MEDIUM_M = 300


# =============================================================================
# Test Fixtures
# =============================================================================

def get_test_data(n: int, m: int, d: int, dtype='float32', seed=SEED):
    """Generate test data."""
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 4)
    
    jnp_dtype = jnp.float32 if dtype == 'float32' else jnp.float64
    
    x = jax.random.normal(keys[0], (n, d), dtype=jnp_dtype)
    y = jax.random.normal(keys[1], (m, d), dtype=jnp_dtype)
    b = jax.random.normal(keys[2], (m, d), dtype=jnp_dtype)
    sigma = jnp.array([0.5], dtype=jnp_dtype)
    
    return x, y, b, sigma


def get_batched_test_data(batch: int, n: int, m: int, d: int, dtype='float32', seed=SEED):
    """Generate batched test data."""
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 4)
    
    jnp_dtype = jnp.float32 if dtype == 'float32' else jnp.float64
    
    x = jax.random.normal(keys[0], (batch, n, d), dtype=jnp_dtype)
    y = jax.random.normal(keys[1], (batch, m, d), dtype=jnp_dtype)
    b = jax.random.normal(keys[2], (batch, m, d), dtype=jnp_dtype)
    sigma = jnp.array([0.5], dtype=jnp_dtype)
    
    return x, y, b, sigma


# =============================================================================
# Genred Tests
# =============================================================================

class TestGenredBasics(unittest.TestCase):
    """Test basic Genred functionality."""
    
    def setUp(self):
        self.N, self.M, self.D = SMALL_N, SMALL_M, 3
        self.x, self.y, self.b, self.sigma = get_test_data(self.N, self.M, self.D)
    
    def test_sum_reduction_basic(self):
        """Test basic Sum reduction with squared distance."""
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Sum', axis=1)
        result = op(self.x, self.y)
        
        # Compute expected result with pure JAX
        diff = self.x[:, None, :] - self.y[None, :, :]  # (N, M, D)
        expected = jnp.sum(jnp.sum(diff**2, axis=-1), axis=1, keepdims=True)  # (N, 1)
        
        self.assertEqual(result.shape, (self.N, 1))
        match, max_diff = compare_arrays(result, expected, rtol=RTOL)
        self.assertTrue(match, f"Max diff: {max_diff}")
    
    def test_sum_reduction_axis0(self):
        """Test Sum reduction over axis 0."""
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Sum', axis=0)
        result = op(self.x, self.y)
        
        # Expected: sum over i
        diff = self.x[:, None, :] - self.y[None, :, :]
        expected = jnp.sum(jnp.sum(diff**2, axis=-1), axis=0, keepdims=True).T
        
        self.assertEqual(result.shape, (self.M, 1))
        match, max_diff = compare_arrays(result, expected, rtol=RTOL)
        self.assertTrue(match, f"Max diff: {max_diff}")
    
    def test_min_reduction(self):
        """Test Min reduction."""
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Min', axis=1)
        result = op(self.x, self.y)
        
        diff = self.x[:, None, :] - self.y[None, :, :]
        expected = jnp.min(jnp.sum(diff**2, axis=-1), axis=1, keepdims=True)
        
        self.assertEqual(result.shape, (self.N, 1))
        match, max_diff = compare_arrays(result, expected, rtol=RTOL)
        self.assertTrue(match, f"Max diff: {max_diff}")
    
    def test_max_reduction(self):
        """Test Max reduction."""
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Max', axis=1)
        result = op(self.x, self.y)
        
        diff = self.x[:, None, :] - self.y[None, :, :]
        expected = jnp.max(jnp.sum(diff**2, axis=-1), axis=1, keepdims=True)
        
        self.assertEqual(result.shape, (self.N, 1))
        match, max_diff = compare_arrays(result, expected, rtol=RTOL)
        self.assertTrue(match, f"Max diff: {max_diff}")
    
    def test_logsumexp_reduction(self):
        """Test LogSumExp reduction."""
        op = Genred("-SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'LogSumExp', axis=1)
        result = op(self.x, self.y)
        
        diff = self.x[:, None, :] - self.y[None, :, :]
        neg_sqdist = -jnp.sum(diff**2, axis=-1)
        expected = jax.scipy.special.logsumexp(neg_sqdist, axis=1, keepdims=True)
        
        self.assertEqual(result.shape, (self.N, 1))
        match, max_diff = compare_arrays(result, expected, rtol=1e-3)  # Looser tolerance for logsumexp
        self.assertTrue(match, f"Max diff: {max_diff}")


class TestGenredFormulas(unittest.TestCase):
    """Test various formula types."""
    
    def setUp(self):
        self.N, self.M, self.D = SMALL_N, SMALL_M, 3
        self.x, self.y, self.b, self.sigma = get_test_data(self.N, self.M, self.D)
    
    def test_gaussian_kernel(self):
        """Test Gaussian kernel formula."""
        formula = "Exp(-SqNorm2(x-y) * s)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})", "s=Pm(1)"]
        op = Genred(formula, aliases, 'Sum', axis=1)
        
        inv_sigma_sq = jnp.array([1.0 / (2 * self.sigma[0]**2)])
        result = op(self.x, self.y, inv_sigma_sq)
        
        # Expected
        diff = self.x[:, None, :] - self.y[None, :, :]
        sqdist = jnp.sum(diff**2, axis=-1)
        K = jnp.exp(-sqdist * float(inv_sigma_sq[0]))
        expected = jnp.sum(K, axis=1, keepdims=True)
        
        self.assertEqual(result.shape, (self.N, 1))
        match, max_diff = compare_arrays(result, expected, rtol=RTOL)
        self.assertTrue(match, f"Max diff: {max_diff}")
    
    def test_kernel_weighted_sum(self):
        """Test weighted kernel sum: K(x,y) @ b."""
        formula = "Exp(-SqNorm2(x-y) * s) * b"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})", f"b=Vj({self.D})", "s=Pm(1)"]
        op = Genred(formula, aliases, 'Sum', axis=1)
        
        inv_sigma_sq = jnp.array([1.0])
        result = op(self.x, self.y, self.b, inv_sigma_sq)
        
        # Expected
        diff = self.x[:, None, :] - self.y[None, :, :]
        sqdist = jnp.sum(diff**2, axis=-1)
        K = jnp.exp(-sqdist * float(inv_sigma_sq[0]))
        expected = K @ self.b
        
        self.assertEqual(result.shape, (self.N, self.D))
        match, max_diff = compare_arrays(result, expected, rtol=RTOL)
        self.assertTrue(match, f"Max diff: {max_diff}")
    
    def test_laplacian_kernel(self):
        """Test Laplacian kernel: exp(-|x-y|)."""
        formula = "Exp(-Norm2(x-y))"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        op = Genred(formula, aliases, 'Sum', axis=1)
        
        result = op(self.x, self.y)
        
        # Expected
        diff = self.x[:, None, :] - self.y[None, :, :]
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))
        K = jnp.exp(-dist)
        expected = jnp.sum(K, axis=1, keepdims=True)
        
        self.assertEqual(result.shape, (self.N, 1))
        match, max_diff = compare_arrays(result, expected, rtol=RTOL)
        self.assertTrue(match, f"Max diff: {max_diff}")
    
    def test_cauchy_kernel(self):
        """Test Cauchy kernel: 1/(1 + |x-y|^2)."""
        formula = "Inv(IntCst(1) + SqNorm2(x-y))"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        op = Genred(formula, aliases, 'Sum', axis=1)
        
        result = op(self.x, self.y)
        
        # Expected
        diff = self.x[:, None, :] - self.y[None, :, :]
        sqdist = jnp.sum(diff**2, axis=-1)
        K = 1.0 / (1.0 + sqdist)
        expected = jnp.sum(K, axis=1, keepdims=True)
        
        self.assertEqual(result.shape, (self.N, 1))
        match, max_diff = compare_arrays(result, expected, rtol=RTOL)
        self.assertTrue(match, f"Max diff: {max_diff}")


class TestGenredParameters(unittest.TestCase):
    """Test parameter handling in Genred."""
    
    def setUp(self):
        self.N, self.M, self.D = SMALL_N, SMALL_M, 3
        self.x, self.y, self.b, self.sigma = get_test_data(self.N, self.M, self.D)
    
    def test_scalar_parameter(self):
        """Test scalar parameter (Pm(1))."""
        formula = "SqDist(x, y) * s"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})", "s=Pm(1)"]
        op = Genred(formula, aliases, 'Sum', axis=1)
        
        scale = jnp.array([2.5])
        result = op(self.x, self.y, scale)
        
        # Expected
        diff = self.x[:, None, :] - self.y[None, :, :]
        sqdist = jnp.sum(diff**2, axis=-1) * float(scale[0])
        expected = jnp.sum(sqdist, axis=1, keepdims=True)
        
        match, max_diff = compare_arrays(result, expected, rtol=RTOL)
        self.assertTrue(match, f"Max diff: {max_diff}")
    
    def test_vector_parameter(self):
        """Test vector parameter (Pm(D))."""
        formula = "SqDist(x * w, y * w)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})", f"w=Pm({self.D})"]
        op = Genred(formula, aliases, 'Sum', axis=1)
        
        weights = jnp.array([1.0, 2.0, 0.5])
        result = op(self.x, self.y, weights)
        
        # Expected
        diff = (self.x * weights)[:, None, :] - (self.y * weights)[None, :, :]
        sqdist = jnp.sum(diff**2, axis=-1)
        expected = jnp.sum(sqdist, axis=1, keepdims=True)
        
        match, max_diff = compare_arrays(result, expected, rtol=RTOL)
        self.assertTrue(match, f"Max diff: {max_diff}")
    
    def test_multiple_parameters(self):
        """Test multiple parameters."""
        formula = "Exp(-SqDist(x, y) * s1) * s2"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})", "s1=Pm(1)", "s2=Pm(1)"]
        op = Genred(formula, aliases, 'Sum', axis=1)
        
        s1 = jnp.array([0.5])
        s2 = jnp.array([2.0])
        result = op(self.x, self.y, s1, s2)
        
        # Expected
        diff = self.x[:, None, :] - self.y[None, :, :]
        sqdist = jnp.sum(diff**2, axis=-1)
        K = jnp.exp(-sqdist * float(s1[0])) * float(s2[0])
        expected = jnp.sum(K, axis=1, keepdims=True)
        
        match, max_diff = compare_arrays(result, expected, rtol=RTOL)
        self.assertTrue(match, f"Max diff: {max_diff}")


class TestGenredDTypes(unittest.TestCase):
    """Test different data types."""
    
    def setUp(self):
        self.N, self.M, self.D = SMALL_N, SMALL_M, 3
    
    def test_float32(self):
        """Test float32 computation."""
        x, y, _, _ = get_test_data(self.N, self.M, self.D, dtype='float32')
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Sum', axis=1, dtype='float32')
        
        result = op(x, y)
        self.assertEqual(result.dtype, jnp.float32)
    
    def test_float64(self):
        """Test float64 computation."""
        x, y, _, _ = get_test_data(self.N, self.M, self.D, dtype='float64')
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Sum', axis=1, dtype='float64')
        
        result = op(x, y)
        self.assertEqual(result.dtype, jnp.float64)


# =============================================================================
# LazyTensor Tests
# =============================================================================

class TestLazyTensorBasics(unittest.TestCase):
    """Test basic LazyTensor functionality."""
    
    def setUp(self):
        self.N, self.M, self.D = SMALL_N, SMALL_M, 3
        self.x, self.y, self.b, self.sigma = get_test_data(self.N, self.M, self.D)
    
    def test_lazy_creation_vi(self):
        """Test LazyTensor creation with Vi pattern."""
        x_i = LazyTensor(self.x[:, None, :])
        self.assertIsNotNone(x_i)
    
    def test_lazy_creation_vj(self):
        """Test LazyTensor creation with Vj pattern."""
        y_j = LazyTensor(self.y[None, :, :])
        self.assertIsNotNone(y_j)
    
    def test_lazy_subtraction(self):
        """Test LazyTensor subtraction."""
        x_i = LazyTensor(self.x[:, None, :])
        y_j = LazyTensor(self.y[None, :, :])
        diff = x_i - y_j
        self.assertIsNotNone(diff)
    
    def test_lazy_squared_distance(self):
        """Test LazyTensor squared distance computation."""
        x_i = LazyTensor(self.x[:, None, :])
        y_j = LazyTensor(self.y[None, :, :])
        
        result = ((x_i - y_j)**2).sum(-1).sum(1)
        
        # Expected
        diff = self.x[:, None, :] - self.y[None, :, :]
        expected = jnp.sum(jnp.sum(diff**2, axis=-1), axis=1, keepdims=True)
        
        self.assertEqual(result.shape, (self.N, 1))
        match, max_diff = compare_arrays(result, expected, rtol=RTOL)
        self.assertTrue(match, f"Max diff: {max_diff}")
    
    def test_lazy_gaussian_kernel(self):
        """Test LazyTensor Gaussian kernel."""
        x_i = LazyTensor(self.x[:, None, :])
        y_j = LazyTensor(self.y[None, :, :])
        b_j = LazyTensor(self.b[None, :, :])
        
        # Use the pattern: ((-D_ij).exp() * b_j).sum(1)
        D_ij = ((x_i - y_j)**2).sum(-1)
        result = ((-D_ij).exp() * b_j).sum(1)
        
        # Expected
        diff = self.x[:, None, :] - self.y[None, :, :]
        sqdist = jnp.sum(diff**2, axis=-1)
        K = jnp.exp(-sqdist)
        expected = K @ self.b
        
        self.assertEqual(result.shape, (self.N, self.D))
        match, max_diff = compare_arrays(result, expected, rtol=RTOL)
        self.assertTrue(match, f"Max diff: {max_diff}")
        self.assertTrue(match, f"Max diff: {max_diff}")


class TestLazyTensorOperations(unittest.TestCase):
    """Test LazyTensor mathematical operations."""
    
    def setUp(self):
        self.N, self.M, self.D = SMALL_N, SMALL_M, 3
        self.x, self.y, self.b, self.sigma = get_test_data(self.N, self.M, self.D)
    
    def test_lazy_exp(self):
        """Test exp() operation."""
        x_i = LazyTensor(self.x[:, None, :])
        y_j = LazyTensor(self.y[None, :, :])
        
        result = (x_i - y_j).exp().sum(-1).sum(1)
        
        diff = self.x[:, None, :] - self.y[None, :, :]
        expected = jnp.sum(jnp.sum(jnp.exp(diff), axis=-1), axis=1, keepdims=True)
        
        match, max_diff = compare_arrays(result, expected, rtol=RTOL)
        self.assertTrue(match, f"Max diff: {max_diff}")
    
    def test_lazy_log(self):
        """Test log() operation on positive values."""
        # Use absolute values to ensure positive
        x_i = LazyTensor(jnp.abs(self.x[:, None, :]) + 0.1)
        y_j = LazyTensor(jnp.abs(self.y[None, :, :]) + 0.1)
        
        result = (x_i + y_j).log().sum(-1).sum(1)
        
        pos_x = jnp.abs(self.x) + 0.1
        pos_y = jnp.abs(self.y) + 0.1
        sum_xy = pos_x[:, None, :] + pos_y[None, :, :]
        expected = jnp.sum(jnp.sum(jnp.log(sum_xy), axis=-1), axis=1, keepdims=True)
        
        match, max_diff = compare_arrays(result, expected, rtol=RTOL)
        self.assertTrue(match, f"Max diff: {max_diff}")
    
    def test_lazy_multiplication(self):
        """Test element-wise multiplication."""
        x_i = LazyTensor(self.x[:, None, :])
        y_j = LazyTensor(self.y[None, :, :])
        
        result = (x_i * y_j).sum(-1).sum(1)
        
        expected = jnp.sum(jnp.sum(self.x[:, None, :] * self.y[None, :, :], axis=-1), axis=1, keepdims=True)
        
        match, max_diff = compare_arrays(result, expected, rtol=RTOL)
        self.assertTrue(match, f"Max diff: {max_diff}")
    
    def test_lazy_scalar_multiplication(self):
        """Test scalar multiplication."""
        x_i = LazyTensor(self.x[:, None, :])
        y_j = LazyTensor(self.y[None, :, :])
        
        result = ((x_i - y_j) * 2.0).sum(-1).sum(1)
        
        diff = (self.x[:, None, :] - self.y[None, :, :]) * 2.0
        expected = jnp.sum(jnp.sum(diff, axis=-1), axis=1, keepdims=True)
        
        match, max_diff = compare_arrays(result, expected, rtol=RTOL)
        self.assertTrue(match, f"Max diff: {max_diff}")


class TestLazyTensorReductions(unittest.TestCase):
    """Test LazyTensor reduction operations."""
    
    def setUp(self):
        self.N, self.M, self.D = SMALL_N, SMALL_M, 3
        self.x, self.y, self.b, self.sigma = get_test_data(self.N, self.M, self.D)
    
    def test_lazy_sum_dim1(self):
        """Test sum reduction over dim 1 (j)."""
        x_i = LazyTensor(self.x[:, None, :])
        y_j = LazyTensor(self.y[None, :, :])
        
        D_ij = ((x_i - y_j)**2).sum(-1)
        result = D_ij.sum(1)
        
        diff = self.x[:, None, :] - self.y[None, :, :]
        sqdist = jnp.sum(diff**2, axis=-1)
        expected = jnp.sum(sqdist, axis=1, keepdims=True)
        
        self.assertEqual(result.shape, (self.N, 1))
        match, max_diff = compare_arrays(result, expected, rtol=RTOL)
        self.assertTrue(match, f"Max diff: {max_diff}")
    
    def test_lazy_sum_dim0(self):
        """Test sum reduction over dim 0 (i)."""
        x_i = LazyTensor(self.x[:, None, :])
        y_j = LazyTensor(self.y[None, :, :])
        
        D_ij = ((x_i - y_j)**2).sum(-1)
        result = D_ij.sum(0)
        
        diff = self.x[:, None, :] - self.y[None, :, :]
        sqdist = jnp.sum(diff**2, axis=-1)
        expected = jnp.sum(sqdist, axis=0, keepdims=True).T
        
        self.assertEqual(result.shape, (self.M, 1))
        match, max_diff = compare_arrays(result, expected, rtol=RTOL)
        self.assertTrue(match, f"Max diff: {max_diff}")
    
    def test_lazy_min(self):
        """Test min reduction."""
        x_i = LazyTensor(self.x[:, None, :])
        y_j = LazyTensor(self.y[None, :, :])
        
        D_ij = ((x_i - y_j)**2).sum(-1)
        result = D_ij.min(1)
        
        diff = self.x[:, None, :] - self.y[None, :, :]
        sqdist = jnp.sum(diff**2, axis=-1)
        expected = jnp.min(sqdist, axis=1, keepdims=True)
        
        match, max_diff = compare_arrays(result, expected, rtol=RTOL)
        self.assertTrue(match, f"Max diff: {max_diff}")
    
    def test_lazy_max(self):
        """Test max reduction."""
        x_i = LazyTensor(self.x[:, None, :])
        y_j = LazyTensor(self.y[None, :, :])
        
        D_ij = ((x_i - y_j)**2).sum(-1)
        result = D_ij.max(1)
        
        diff = self.x[:, None, :] - self.y[None, :, :]
        sqdist = jnp.sum(diff**2, axis=-1)
        expected = jnp.max(sqdist, axis=1, keepdims=True)
        
        match, max_diff = compare_arrays(result, expected, rtol=RTOL)
        self.assertTrue(match, f"Max diff: {max_diff}")


# =============================================================================
# Vi/Vj/Pm Helper Tests
# =============================================================================

class TestViVjPmHelpers(unittest.TestCase):
    """Test Vi, Vj, Pm helper functions."""
    
    def setUp(self):
        self.N, self.M, self.D = SMALL_N, SMALL_M, 3
        self.x, self.y, self.b, self.sigma = get_test_data(self.N, self.M, self.D)
    
    def test_vi_from_2d(self):
        """Test Vi() with 2D array."""
        x_i = Vi(self.x)
        self.assertIsNotNone(x_i)
    
    def test_vj_from_2d(self):
        """Test Vj() with 2D array."""
        y_j = Vj(self.y)
        self.assertIsNotNone(y_j)
    
    def test_pm_scalar(self):
        """Test Pm() with scalar."""
        s = Pm(self.sigma)
        self.assertIsNotNone(s)
    
    def test_vi_vj_computation(self):
        """Test computation using Vi/Vj helpers."""
        x_i = Vi(self.x)
        y_j = Vj(self.y)
        
        result = ((x_i - y_j)**2).sum(-1).sum(1)
        
        diff = self.x[:, None, :] - self.y[None, :, :]
        expected = jnp.sum(jnp.sum(diff**2, axis=-1), axis=1, keepdims=True)
        
        match, max_diff = compare_arrays(result, expected, rtol=RTOL)
        self.assertTrue(match, f"Max diff: {max_diff}")


# =============================================================================
# JIT Compilation Tests
# =============================================================================

class TestJITCompilation(unittest.TestCase):
    """Test JIT compilation compatibility."""
    
    def setUp(self):
        self.N, self.M, self.D = SMALL_N, SMALL_M, 3
        self.x, self.y, self.b, self.sigma = get_test_data(self.N, self.M, self.D)
    
    def test_genred_jit(self):
        """Test JIT-compiled Genred."""
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Sum', axis=1)
        
        @jax.jit
        def compute(x, y):
            return op(x, y)
        
        result = compute(self.x, self.y)
        self.assertEqual(result.shape, (self.N, 1))
    
    def test_lazy_jit(self):
        """Test JIT-compiled LazyTensor."""
        @jax.jit
        def compute(x, y):
            x_i = LazyTensor(x[:, None, :])
            y_j = LazyTensor(y[None, :, :])
            return ((x_i - y_j)**2).sum(-1).sum(1)
        
        result = compute(self.x, self.y)
        self.assertEqual(result.shape, (self.N, 1))
    
    def test_jit_repeated_calls(self):
        """Test repeated JIT calls use cache."""
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Sum', axis=1)
        
        @jax.jit
        def compute(x, y):
            return op(x, y)
        
        # First call - compiles
        result1 = compute(self.x, self.y)
        
        # Second call - should use cache
        result2 = compute(self.x, self.y)
        
        match, _ = compare_arrays(result1, result2)
        self.assertTrue(match)


# =============================================================================
# Gradient Tests
# =============================================================================

class TestGradients(unittest.TestCase):
    """Test gradient computation."""
    
    def setUp(self):
        self.N, self.M, self.D = 20, 15, 3  # Smaller for gradient tests
        self.x, self.y, self.b, self.sigma = get_test_data(self.N, self.M, self.D)
    
    def test_genred_grad_vi(self):
        """Test gradient w.r.t. Vi variable."""
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Sum', axis=1)
        
        def loss(x):
            return jnp.sum(op(x, self.y))
        
        grad = jax.grad(loss)(self.x)
        
        self.assertEqual(grad.shape, self.x.shape)
        self.assertFalse(jnp.any(jnp.isnan(grad)))
    
    def test_genred_grad_vj(self):
        """Test gradient w.r.t. Vj variable."""
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Sum', axis=1)
        
        def loss(y):
            return jnp.sum(op(self.x, y))
        
        grad = jax.grad(loss)(self.y)
        
        self.assertEqual(grad.shape, self.y.shape)
        self.assertFalse(jnp.any(jnp.isnan(grad)))
    
    def test_genred_grad_both(self):
        """Test gradient w.r.t. both variables."""
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Sum', axis=1)
        
        def loss(x, y):
            return jnp.sum(op(x, y))
        
        grad_x, grad_y = jax.grad(loss, argnums=(0, 1))(self.x, self.y)
        
        self.assertEqual(grad_x.shape, self.x.shape)
        self.assertEqual(grad_y.shape, self.y.shape)
    
    def test_lazy_grad(self):
        """Test LazyTensor gradient."""
        def loss(x):
            x_i = LazyTensor(x[:, None, :])
            y_j = LazyTensor(self.y[None, :, :])
            return jnp.sum(((x_i - y_j)**2).sum(-1).sum(1))
        
        grad = jax.grad(loss)(self.x)
        
        self.assertEqual(grad.shape, self.x.shape)
        self.assertFalse(jnp.any(jnp.isnan(grad)))
    
    def test_value_and_grad(self):
        """Test value_and_grad."""
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Sum', axis=1)
        
        def loss(x):
            return jnp.sum(op(x, self.y))
        
        val, grad = jax.value_and_grad(loss)(self.x)
        
        self.assertTrue(val > 0)
        self.assertEqual(grad.shape, self.x.shape)


# =============================================================================
# Batching Tests
# =============================================================================

class TestBatching(unittest.TestCase):
    """Test batched operations."""
    
    def setUp(self):
        self.B, self.N, self.M, self.D = 5, SMALL_N, SMALL_M, 3
        self.x, self.y, self.b, self.sigma = get_batched_test_data(self.B, self.N, self.M, self.D)
    
    def test_vmap_genred(self):
        """Test vmap over Genred."""
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Sum', axis=1)
        
        result = jax.vmap(op)(self.x, self.y)
        
        self.assertEqual(result.shape, (self.B, self.N, 1))
    
    def test_batched_3d_input(self):
        """Test 3D batched input directly."""
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Sum', axis=1)
        
        # Direct 3D input
        result = op(self.x, self.y)
        
        self.assertEqual(result.shape, (self.B, self.N, 1))
    
    def test_vmap_gradient(self):
        """Test vmap over gradient computation."""
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Sum', axis=1)
        
        def single_loss(x, y):
            return jnp.sum(op(x, y))
        
        grad_fn = jax.grad(single_loss)
        batched_grad = jax.vmap(grad_fn)(self.x, self.y)
        
        self.assertEqual(batched_grad.shape, self.x.shape)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special inputs."""
    
    def test_empty_input_vi(self):
        """Test with empty Vi input."""
        op = Genred("SqDist(x, y)", ["x=Vi(3)", "y=Vj(3)"], 'Sum', axis=1)
        
        x = jnp.zeros((0, 3))
        y = jnp.ones((5, 3))
        result = op(x, y)
        
        self.assertEqual(result.shape, (0, 1))
    
    def test_single_point(self):
        """Test with single point."""
        op = Genred("SqDist(x, y)", ["x=Vi(3)", "y=Vj(3)"], 'Sum', axis=1)
        
        x = jnp.array([[1.0, 2.0, 3.0]])
        y = jnp.array([[0.0, 0.0, 0.0]])
        result = op(x, y)
        
        expected = jnp.array([[14.0]])  # 1^2 + 2^2 + 3^2
        match, max_diff = compare_arrays(result, expected, rtol=1e-5)
        self.assertTrue(match, f"Max diff: {max_diff}")
    
    def test_high_dimension(self):
        """Test with high-dimensional data."""
        D = 64
        N, M = 20, 15
        
        x, y, _, _ = get_test_data(N, M, D)
        op = Genred("SqDist(x, y)", [f"x=Vi({D})", f"y=Vj({D})"], 'Sum', axis=1)
        
        result = op(x, y)
        self.assertEqual(result.shape, (N, 1))


# =============================================================================
# Test Runner
# =============================================================================

def run_api_tests():
    """Run all API tests with nice output."""
    if not KEOPS_AVAILABLE:
        print(f"{Colors.RED}KeOps JAX not available. Cannot run tests.{Colors.RESET}")
        return False
    
    print_header("KeOps JAX API Tests")
    
    # Create test suite
    suite = TestSuite("API Test Results")
    
    # Collect all test classes
    test_classes = [
        TestGenredBasics,
        TestGenredFormulas,
        TestGenredParameters,
        TestGenredDTypes,
        TestLazyTensorBasics,
        TestLazyTensorOperations,
        TestLazyTensorReductions,
        TestViVjPmHelpers,
        TestJITCompilation,
        TestGradients,
        TestBatching,
        TestEdgeCases,
    ]
    
    # Run tests
    loader = unittest.TestLoader()
    
    for test_class in test_classes:
        print_subheader(test_class.__name__)
        
        tests = loader.loadTestsFromTestCase(test_class)
        
        for test in tests:
            test_name = test._testMethodName
            start_time = time.time()
            
            try:
                # Run with setUp and tearDown
                test.setUp()
                getattr(test, test_name)()
                if hasattr(test, 'tearDown'):
                    test.tearDown()
                
                duration = (time.time() - start_time) * 1000
                suite.add_result(TestResult(test_name, Status.PASS, duration))
                print(f"  {Colors.GREEN}✓{Colors.RESET} {test_name} ({duration:.1f}ms)")
                
            except unittest.SkipTest as e:
                duration = (time.time() - start_time) * 1000
                suite.add_result(TestResult(test_name, Status.SKIP, duration, str(e)))
                print(f"  {Colors.YELLOW}○{Colors.RESET} {test_name} (skipped: {e})")
                
            except AssertionError as e:
                duration = (time.time() - start_time) * 1000
                suite.add_result(TestResult(test_name, Status.FAIL, duration, str(e)))
                print(f"  {Colors.RED}✗{Colors.RESET} {test_name}: {e}")
                
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                suite.add_result(TestResult(test_name, Status.ERROR, duration, str(e)))
                print(f"  {Colors.RED}💥{Colors.RESET} {test_name}: {e}")
    
    # Print summary
    suite.print_summary()
    
    return suite.all_passed()


if __name__ == '__main__':
    import sys
    success = run_api_tests()
    sys.exit(0 if success else 1)
