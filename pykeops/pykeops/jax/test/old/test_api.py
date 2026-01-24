#!/usr/bin/env python3
"""
KeOps JAX API Tests
===================
Comprehensive unit tests for the JAX KeOps API.

Tests compare JAX KeOps against PyTorch KeOps (ground truth) when available,
falling back to element-wise JAX computation otherwise.

Tests cover:
- Genred interface (all reduction operations)
- LazyTensor interface (operations and reductions)
- Vi, Vj, Pm helper functions
- Parameter handling
- Data types (float32, float64)
- Batched and unbatched operations
- JIT compilation
- Gradients
- Small, medium, and large matrix sizes
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

    KEOPS_JAX_AVAILABLE = True
except ImportError as e:
    print(f"{Colors.RED}Error: pykeops.jax not found: {e}{Colors.RESET}")
    KEOPS_JAX_AVAILABLE = False

# Import PyTorch KeOps for ground truth comparison
try:
    import torch
    from pykeops.torch import Genred as Genred_torch, LazyTensor as LazyTensor_torch

    KEOPS_TORCH_AVAILABLE = torch.cuda.is_available()
    if KEOPS_TORCH_AVAILABLE:
        print(f"{Colors.GREEN}PyTorch KeOps available - using as ground truth{Colors.RESET}")
except ImportError:
    KEOPS_TORCH_AVAILABLE = False
    print(f"{Colors.YELLOW}PyTorch KeOps not available - using element-wise JAX as reference{Colors.RESET}")

# =============================================================================
# Test Configuration
# =============================================================================

# Random seed for reproducibility
SEED = 42

# Default tolerances (tight since comparing two KeOps implementations)
RTOL = 1e-5
ATOL = 1e-6

# Looser tolerance for comparing against element-wise JAX (different FP accumulation)
RTOL_ELEMWISE = 1e-3
ATOL_ELEMWISE = 1e-3

# Test sizes
SMALL_N = 50
SMALL_M = 30
MEDIUM_N = 500
MEDIUM_M = 300
LARGE_N = 5000
LARGE_M = 3000


# =============================================================================
# Test Fixtures
# =============================================================================

def get_test_data(n: int, m: int, d: int, dtype='float32', seed=SEED):
    """Generate test data as numpy arrays (can be converted to JAX or PyTorch)."""
    np.random.seed(seed)

    x = np.random.randn(n, d).astype(np.float32 if dtype == 'float32' else np.float64)
    y = np.random.randn(m, d).astype(np.float32 if dtype == 'float32' else np.float64)
    b = np.random.randn(m, d).astype(np.float32 if dtype == 'float32' else np.float64)
    sigma = np.array([0.5], dtype=np.float32 if dtype == 'float32' else np.float64)

    return x, y, b, sigma


def get_batched_test_data(batch: int, n: int, m: int, d: int, dtype='float32', seed=SEED):
    """Generate batched test data."""
    np.random.seed(seed)

    x = np.random.randn(batch, n, d).astype(np.float32 if dtype == 'float32' else np.float64)
    y = np.random.randn(batch, m, d).astype(np.float32 if dtype == 'float32' else np.float64)
    b = np.random.randn(batch, m, d).astype(np.float32 if dtype == 'float32' else np.float64)
    sigma = np.array([0.5], dtype=np.float32 if dtype == 'float32' else np.float64)

    return x, y, b, sigma


def to_jax(arr):
    """Convert numpy array to JAX array."""
    return jnp.array(arr)


def to_torch(arr):
    """Convert numpy array to PyTorch CUDA tensor."""
    return torch.tensor(arr, device='cuda', dtype=torch.float32)


def torch_to_numpy(t):
    """Convert PyTorch tensor to numpy."""
    return t.detach().cpu().numpy()


# =============================================================================
# PyTorch KeOps Reference Computations
# =============================================================================

def pytorch_genred_reference(formula, aliases, reduction_op, axis, *args_np):
    """Compute reference result using PyTorch KeOps."""
    if not KEOPS_TORCH_AVAILABLE:
        return None

    args_torch = [to_torch(a) for a in args_np]
    op = Genred_torch(formula, aliases, reduction_op=reduction_op, axis=axis)
    result = op(*args_torch)
    torch.cuda.synchronize()
    return torch_to_numpy(result)


def pytorch_lazy_reference(compute_fn, *args_np):
    """Compute reference result using PyTorch LazyTensor."""
    if not KEOPS_TORCH_AVAILABLE:
        return None

    args_torch = [to_torch(a) for a in args_np]
    result = compute_fn(*args_torch)
    torch.cuda.synchronize()
    return torch_to_numpy(result)


# =============================================================================
# Element-wise JAX Reference (fallback)
# =============================================================================

def elemwise_sqdist_sum(x, y):
    """Element-wise squared distance sum."""
    diff = x[:, None, :] - y[None, :, :]
    return jnp.sum(jnp.sum(diff ** 2, axis=-1), axis=1, keepdims=True)


def elemwise_gaussian_kernel_sum(x, y, inv_sigma_sq):
    """Element-wise Gaussian kernel sum."""
    diff = x[:, None, :] - y[None, :, :]
    sqdist = jnp.sum(diff ** 2, axis=-1)
    K = jnp.exp(-sqdist * float(inv_sigma_sq[0]))
    return jnp.sum(K, axis=1, keepdims=True)


def elemwise_kernel_weighted_sum(x, y, b, inv_sigma_sq):
    """Element-wise kernel-weighted sum (matches KeOps reduction order)."""
    diff = x[:, None, :] - y[None, :, :]
    sqdist = jnp.sum(diff ** 2, axis=-1)
    K = jnp.exp(-sqdist * float(inv_sigma_sq[0]))
    # Use element-wise to match KeOps FP accumulation order
    return jnp.sum(K[:, :, None] * b[None, :, :], axis=1)


# =============================================================================
# Genred Tests
# =============================================================================

class TestGenredBasics(unittest.TestCase):
    """Test basic Genred functionality."""

    def setUp(self):
        self.N, self.M, self.D = SMALL_N, SMALL_M, 3
        self.x_np, self.y_np, self.b_np, self.sigma_np = get_test_data(self.N, self.M, self.D)
        self.x = to_jax(self.x_np)
        self.y = to_jax(self.y_np)
        self.b = to_jax(self.b_np)
        self.sigma = to_jax(self.sigma_np)

    def test_sum_reduction_basic(self):
        """Test basic Sum reduction with squared distance."""
        formula = "SqDist(x, y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]

        op = Genred(formula, aliases, 'Sum', axis=1)
        result = op(self.x, self.y)

        # Get reference
        ref = pytorch_genred_reference(formula, aliases, 'Sum', 1, self.x_np, self.y_np)
        if ref is not None:
            rtol, atol = RTOL, ATOL
        else:
            ref = np.array(elemwise_sqdist_sum(self.x, self.y))
            rtol, atol = RTOL_ELEMWISE, ATOL_ELEMWISE

        self.assertEqual(result.shape, (self.N, 1))
        match, max_diff = compare_arrays(result, ref, rtol=rtol, atol=atol)
        self.assertTrue(match, f"Max diff: {max_diff}")

    def test_sum_reduction_axis0(self):
        """Test Sum reduction over axis 0."""
        formula = "SqDist(x, y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]

        op = Genred(formula, aliases, 'Sum', axis=0)
        result = op(self.x, self.y)

        ref = pytorch_genred_reference(formula, aliases, 'Sum', 0, self.x_np, self.y_np)
        if ref is not None:
            rtol, atol = RTOL, ATOL
        else:
            diff = self.x[:, None, :] - self.y[None, :, :]
            ref = np.array(jnp.sum(jnp.sum(diff ** 2, axis=-1), axis=0, keepdims=True).T)
            rtol, atol = RTOL_ELEMWISE, ATOL_ELEMWISE

        self.assertEqual(result.shape, (self.M, 1))
        match, max_diff = compare_arrays(result, ref, rtol=rtol, atol=atol)
        self.assertTrue(match, f"Max diff: {max_diff}")

    def test_min_reduction(self):
        """Test Min reduction."""
        formula = "SqDist(x, y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]

        op = Genred(formula, aliases, 'Min', axis=1)
        result = op(self.x, self.y)

        ref = pytorch_genred_reference(formula, aliases, 'Min', 1, self.x_np, self.y_np)
        if ref is not None:
            rtol, atol = RTOL, ATOL
        else:
            diff = self.x[:, None, :] - self.y[None, :, :]
            ref = np.array(jnp.min(jnp.sum(diff ** 2, axis=-1), axis=1, keepdims=True))
            rtol, atol = RTOL_ELEMWISE, ATOL_ELEMWISE

        self.assertEqual(result.shape, (self.N, 1))
        match, max_diff = compare_arrays(result, ref, rtol=rtol, atol=atol)
        self.assertTrue(match, f"Max diff: {max_diff}")

    def test_max_reduction(self):
        """Test Max reduction."""
        formula = "SqDist(x, y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]

        op = Genred(formula, aliases, 'Max', axis=1)
        result = op(self.x, self.y)

        ref = pytorch_genred_reference(formula, aliases, 'Max', 1, self.x_np, self.y_np)
        if ref is not None:
            rtol, atol = RTOL, ATOL
        else:
            diff = self.x[:, None, :] - self.y[None, :, :]
            ref = np.array(jnp.max(jnp.sum(diff ** 2, axis=-1), axis=1, keepdims=True))
            rtol, atol = RTOL_ELEMWISE, ATOL_ELEMWISE

        self.assertEqual(result.shape, (self.N, 1))
        match, max_diff = compare_arrays(result, ref, rtol=rtol, atol=atol)
        self.assertTrue(match, f"Max diff: {max_diff}")


class TestGenredFormulas(unittest.TestCase):
    """Test various formula types."""

    def setUp(self):
        self.N, self.M, self.D = SMALL_N, SMALL_M, 3
        self.x_np, self.y_np, self.b_np, self.sigma_np = get_test_data(self.N, self.M, self.D)
        self.x = to_jax(self.x_np)
        self.y = to_jax(self.y_np)
        self.b = to_jax(self.b_np)
        self.sigma = to_jax(self.sigma_np)

    def test_gaussian_kernel(self):
        """Test Gaussian kernel formula."""
        formula = "Exp(-SqNorm2(x-y) * s)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})", "s=Pm(1)"]

        inv_sigma_sq = jnp.array([1.0 / (2 * self.sigma[0] ** 2)])
        inv_sigma_sq_np = np.array(inv_sigma_sq)

        op = Genred(formula, aliases, 'Sum', axis=1)
        result = op(self.x, self.y, inv_sigma_sq)

        ref = pytorch_genred_reference(formula, aliases, 'Sum', 1,
                                       self.x_np, self.y_np, inv_sigma_sq_np)
        if ref is not None:
            rtol, atol = RTOL, ATOL
        else:
            ref = np.array(elemwise_gaussian_kernel_sum(self.x, self.y, inv_sigma_sq))
            rtol, atol = RTOL_ELEMWISE, ATOL_ELEMWISE

        self.assertEqual(result.shape, (self.N, 1))
        match, max_diff = compare_arrays(result, ref, rtol=rtol, atol=atol)
        self.assertTrue(match, f"Max diff: {max_diff}")

    def test_kernel_weighted_sum(self):
        """Test weighted kernel sum: K(x,y) @ b."""
        formula = "Exp(-SqNorm2(x-y) * s) * b"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})", f"b=Vj({self.D})", "s=Pm(1)"]

        inv_sigma_sq = jnp.array([1.0])
        inv_sigma_sq_np = np.array(inv_sigma_sq)

        op = Genred(formula, aliases, 'Sum', axis=1)
        result = op(self.x, self.y, self.b, inv_sigma_sq)

        ref = pytorch_genred_reference(formula, aliases, 'Sum', 1,
                                       self.x_np, self.y_np, self.b_np, inv_sigma_sq_np)
        if ref is not None:
            rtol, atol = RTOL, ATOL
        else:
            ref = np.array(elemwise_kernel_weighted_sum(self.x, self.y, self.b, inv_sigma_sq))
            rtol, atol = RTOL_ELEMWISE, ATOL_ELEMWISE

        self.assertEqual(result.shape, (self.N, self.D))
        match, max_diff = compare_arrays(result, ref, rtol=rtol, atol=atol)
        self.assertTrue(match, f"Max diff: {max_diff}")

    def test_laplacian_kernel(self):
        """Test Laplacian kernel: exp(-|x-y|)."""
        formula = "Exp(-Norm2(x-y))"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]

        op = Genred(formula, aliases, 'Sum', axis=1)
        result = op(self.x, self.y)

        ref = pytorch_genred_reference(formula, aliases, 'Sum', 1, self.x_np, self.y_np)
        if ref is not None:
            rtol, atol = RTOL, ATOL
        else:
            diff = self.x[:, None, :] - self.y[None, :, :]
            dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
            K = jnp.exp(-dist)
            ref = np.array(jnp.sum(K, axis=1, keepdims=True))
            rtol, atol = RTOL_ELEMWISE, ATOL_ELEMWISE

        self.assertEqual(result.shape, (self.N, 1))
        match, max_diff = compare_arrays(result, ref, rtol=rtol, atol=atol)
        self.assertTrue(match, f"Max diff: {max_diff}")

    def test_cauchy_kernel(self):
        """Test Cauchy kernel: 1/(1+|x-y|^2)."""
        formula = "Inv(IntCst(1) + SqNorm2(x-y))"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]

        op = Genred(formula, aliases, 'Sum', axis=1)
        result = op(self.x, self.y)

        ref = pytorch_genred_reference(formula, aliases, 'Sum', 1, self.x_np, self.y_np)
        if ref is not None:
            rtol, atol = RTOL, ATOL
        else:
            diff = self.x[:, None, :] - self.y[None, :, :]
            sqdist = jnp.sum(diff ** 2, axis=-1)
            K = 1.0 / (1.0 + sqdist)
            ref = np.array(jnp.sum(K, axis=1, keepdims=True))
            rtol, atol = RTOL_ELEMWISE, ATOL_ELEMWISE

        self.assertEqual(result.shape, (self.N, 1))
        match, max_diff = compare_arrays(result, ref, rtol=rtol, atol=atol)
        self.assertTrue(match, f"Max diff: {max_diff}")


class TestGenredParameters(unittest.TestCase):
    """Test parameter (Pm) handling."""

    def setUp(self):
        self.N, self.M, self.D = SMALL_N, SMALL_M, 3
        self.x_np, self.y_np, self.b_np, self.sigma_np = get_test_data(self.N, self.M, self.D)
        self.x = to_jax(self.x_np)
        self.y = to_jax(self.y_np)

    def test_scalar_parameter(self):
        """Test scalar parameter."""
        formula = "Exp(-SqNorm2(x-y) * s)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})", "s=Pm(1)"]

        s = jnp.array([2.0])
        s_np = np.array(s)

        op = Genred(formula, aliases, 'Sum', axis=1)
        result = op(self.x, self.y, s)

        ref = pytorch_genred_reference(formula, aliases, 'Sum', 1, self.x_np, self.y_np, s_np)
        if ref is not None:
            rtol, atol = RTOL, ATOL
        else:
            ref = np.array(elemwise_gaussian_kernel_sum(self.x, self.y, s))
            rtol, atol = RTOL_ELEMWISE, ATOL_ELEMWISE

        self.assertEqual(result.shape, (self.N, 1))
        match, max_diff = compare_arrays(result, ref, rtol=rtol, atol=atol)
        self.assertTrue(match, f"Max diff: {max_diff}")

    def test_vector_parameter(self):
        """Test vector parameter."""
        formula = "Exp(-WeightedSqNorm(w, x-y))"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})", f"w=Pm({self.D})"]

        w = jnp.array([1.0, 2.0, 0.5])
        w_np = np.array(w)

        op = Genred(formula, aliases, 'Sum', axis=1)
        result = op(self.x, self.y, w)

        ref = pytorch_genred_reference(formula, aliases, 'Sum', 1, self.x_np, self.y_np, w_np)
        if ref is not None:
            rtol, atol = RTOL, ATOL
        else:
            diff = self.x[:, None, :] - self.y[None, :, :]
            weighted_sqdist = jnp.sum(w * diff ** 2, axis=-1)
            ref = np.array(jnp.sum(jnp.exp(-weighted_sqdist), axis=1, keepdims=True))
            rtol, atol = RTOL_ELEMWISE, ATOL_ELEMWISE

        self.assertEqual(result.shape, (self.N, 1))
        match, max_diff = compare_arrays(result, ref, rtol=rtol, atol=atol)
        self.assertTrue(match, f"Max diff: {max_diff}")

    def test_multiple_parameters(self):
        """Test multiple parameters."""
        formula = "Exp(-s1 * SqNorm2(x-y)) + Exp(-s2 * SqNorm2(x-y))"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})", "s1=Pm(1)", "s2=Pm(1)"]

        s1 = jnp.array([1.0])
        s2 = jnp.array([0.5])
        s1_np, s2_np = np.array(s1), np.array(s2)

        op = Genred(formula, aliases, 'Sum', axis=1)
        result = op(self.x, self.y, s1, s2)

        ref = pytorch_genred_reference(formula, aliases, 'Sum', 1,
                                       self.x_np, self.y_np, s1_np, s2_np)
        if ref is not None:
            rtol, atol = RTOL, ATOL
        else:
            diff = self.x[:, None, :] - self.y[None, :, :]
            sqdist = jnp.sum(diff ** 2, axis=-1)
            K = jnp.exp(-float(s1[0]) * sqdist) + jnp.exp(-float(s2[0]) * sqdist)
            ref = np.array(jnp.sum(K, axis=1, keepdims=True))
            rtol, atol = RTOL_ELEMWISE, ATOL_ELEMWISE

        self.assertEqual(result.shape, (self.N, 1))
        match, max_diff = compare_arrays(result, ref, rtol=rtol, atol=atol)
        self.assertTrue(match, f"Max diff: {max_diff}")


class TestGenredDTypes(unittest.TestCase):
    """Test data type handling."""

    def test_float32(self):
        """Test float32 computation."""
        N, M, D = SMALL_N, SMALL_M, 3
        x_np, y_np, _, _ = get_test_data(N, M, D, dtype='float32')
        x, y = to_jax(x_np), to_jax(y_np)

        op = Genred("SqDist(x, y)", [f"x=Vi({D})", f"y=Vj({D})"], 'Sum', axis=1)
        result = op(x, y)

        self.assertEqual(result.dtype, jnp.float32)


# =============================================================================
# LazyTensor Tests
# =============================================================================

class TestLazyTensorBasics(unittest.TestCase):
    """Test basic LazyTensor functionality."""

    def setUp(self):
        self.N, self.M, self.D = SMALL_N, SMALL_M, 3
        self.x_np, self.y_np, self.b_np, _ = get_test_data(self.N, self.M, self.D)
        self.x = to_jax(self.x_np)
        self.y = to_jax(self.y_np)
        self.b = to_jax(self.b_np)

    def test_lazy_creation_vi(self):
        """Test LazyTensor creation with Vi pattern."""
        x_i = LazyTensor(self.x[:, None, :])
        self.assertEqual(x_i._shape, (self.N, 1, self.D))

    def test_lazy_creation_vj(self):
        """Test LazyTensor creation with Vj pattern."""
        y_j = LazyTensor(self.y[None, :, :])
        self.assertEqual(y_j._shape, (1, self.M, self.D))

    def test_lazy_subtraction(self):
        """Test LazyTensor subtraction."""
        x_i = LazyTensor(self.x[:, None, :])
        y_j = LazyTensor(self.y[None, :, :])
        diff = x_i - y_j
        self.assertIsInstance(diff, LazyTensor)

    def test_lazy_squared_distance(self):
        """Test LazyTensor squared distance computation."""
        x_i = LazyTensor(self.x[:, None, :])
        y_j = LazyTensor(self.y[None, :, :])

        D_ij = ((x_i - y_j) ** 2).sum(-1)
        result = D_ij.sum(1)

        # Reference
        if KEOPS_TORCH_AVAILABLE:
            x_t = to_torch(self.x_np)
            y_t = to_torch(self.y_np)
            x_i_t = LazyTensor_torch(x_t[:, None, :])
            y_j_t = LazyTensor_torch(y_t[None, :, :])
            D_ij_t = ((x_i_t - y_j_t) ** 2).sum(-1)
            ref = torch_to_numpy(D_ij_t.sum(1))
            rtol, atol = RTOL, ATOL
        else:
            diff = self.x[:, None, :] - self.y[None, :, :]
            ref = np.array(jnp.sum(jnp.sum(diff ** 2, axis=-1), axis=1, keepdims=True))
            rtol, atol = RTOL_ELEMWISE, ATOL_ELEMWISE

        self.assertEqual(result.shape, (self.N, 1))
        match, max_diff = compare_arrays(result, ref, rtol=rtol, atol=atol)
        self.assertTrue(match, f"Max diff: {max_diff}")

    def test_lazy_gaussian_kernel(self):
        """Test LazyTensor Gaussian kernel."""
        x_i = LazyTensor(self.x[:, None, :])
        y_j = LazyTensor(self.y[None, :, :])
        b_j = LazyTensor(self.b[None, :, :])

        D_ij = ((x_i - y_j) ** 2).sum(-1)
        result = ((-D_ij).exp() * b_j).sum(1)

        # Reference
        if KEOPS_TORCH_AVAILABLE:
            x_t = to_torch(self.x_np)
            y_t = to_torch(self.y_np)
            b_t = to_torch(self.b_np)
            x_i_t = LazyTensor_torch(x_t[:, None, :])
            y_j_t = LazyTensor_torch(y_t[None, :, :])
            b_j_t = LazyTensor_torch(b_t[None, :, :])
            D_ij_t = ((x_i_t - y_j_t) ** 2).sum(-1)
            ref = torch_to_numpy(((-D_ij_t).exp() * b_j_t).sum(1))
            rtol, atol = RTOL, ATOL
        else:
            # Element-wise to match KeOps reduction order
            diff = self.x[:, None, :] - self.y[None, :, :]
            sqdist = jnp.sum(diff ** 2, axis=-1)
            K = jnp.exp(-sqdist)
            ref = np.array(jnp.sum(K[:, :, None] * self.b[None, :, :], axis=1))
            rtol, atol = RTOL_ELEMWISE, ATOL_ELEMWISE

        self.assertEqual(result.shape, (self.N, self.D))
        match, max_diff = compare_arrays(result, ref, rtol=rtol, atol=atol)
        self.assertTrue(match, f"Max diff: {max_diff}")


class TestLazyTensorOperations(unittest.TestCase):
    """Test LazyTensor mathematical operations."""

    def setUp(self):
        self.N, self.M, self.D = SMALL_N, SMALL_M, 3
        self.x_np, self.y_np, self.b_np, _ = get_test_data(self.N, self.M, self.D)
        self.x = to_jax(self.x_np)
        self.y = to_jax(self.y_np)
        self.b = to_jax(self.b_np)

    def test_lazy_exp(self):
        """Test exp() operation."""
        x_i = LazyTensor(self.x[:, None, :])
        y_j = LazyTensor(self.y[None, :, :])

        result = (x_i - y_j).exp().sum(-1).sum(1)

        if KEOPS_TORCH_AVAILABLE:
            x_t, y_t = to_torch(self.x_np), to_torch(self.y_np)
            x_i_t = LazyTensor_torch(x_t[:, None, :])
            y_j_t = LazyTensor_torch(y_t[None, :, :])
            ref = torch_to_numpy((x_i_t - y_j_t).exp().sum(-1).sum(1))
            rtol, atol = RTOL, ATOL
        else:
            diff = self.x[:, None, :] - self.y[None, :, :]
            ref = np.array(jnp.sum(jnp.sum(jnp.exp(diff), axis=-1), axis=1, keepdims=True))
            rtol, atol = RTOL_ELEMWISE, ATOL_ELEMWISE

        match, max_diff = compare_arrays(result, ref, rtol=rtol, atol=atol)
        self.assertTrue(match, f"Max diff: {max_diff}")

    def test_lazy_log(self):
        """Test log() operation on positive values."""
        # Use absolute values to ensure positive
        pos_x = jnp.abs(self.x) + 0.1
        pos_y = jnp.abs(self.y) + 0.1

        x_i = LazyTensor(pos_x[:, None, :])
        y_j = LazyTensor(pos_y[None, :, :])

        result = (x_i + y_j).log().sum(-1).sum(1)

        if KEOPS_TORCH_AVAILABLE:
            pos_x_np = np.abs(self.x_np) + 0.1
            pos_y_np = np.abs(self.y_np) + 0.1
            x_t, y_t = to_torch(pos_x_np), to_torch(pos_y_np)
            x_i_t = LazyTensor_torch(x_t[:, None, :])
            y_j_t = LazyTensor_torch(y_t[None, :, :])
            ref = torch_to_numpy((x_i_t + y_j_t).log().sum(-1).sum(1))
            rtol, atol = RTOL, ATOL
        else:
            sum_xy = pos_x[:, None, :] + pos_y[None, :, :]
            ref = np.array(jnp.sum(jnp.sum(jnp.log(sum_xy), axis=-1), axis=1, keepdims=True))
            rtol, atol = RTOL_ELEMWISE, ATOL_ELEMWISE

        match, max_diff = compare_arrays(result, ref, rtol=rtol, atol=atol)
        self.assertTrue(match, f"Max diff: {max_diff}")

    def test_lazy_multiplication(self):
        """Test element-wise multiplication."""
        x_i = LazyTensor(self.x[:, None, :])
        y_j = LazyTensor(self.y[None, :, :])

        result = (x_i * y_j).sum(-1).sum(1)

        if KEOPS_TORCH_AVAILABLE:
            x_t, y_t = to_torch(self.x_np), to_torch(self.y_np)
            x_i_t = LazyTensor_torch(x_t[:, None, :])
            y_j_t = LazyTensor_torch(y_t[None, :, :])
            ref = torch_to_numpy((x_i_t * y_j_t).sum(-1).sum(1))
            rtol, atol = RTOL, ATOL
        else:
            ref = np.array(jnp.sum(jnp.sum(self.x[:, None, :] * self.y[None, :, :], axis=-1), axis=1, keepdims=True))
            rtol, atol = RTOL_ELEMWISE, ATOL_ELEMWISE

        match, max_diff = compare_arrays(result, ref, rtol=rtol, atol=atol)
        self.assertTrue(match, f"Max diff: {max_diff}")

    def test_lazy_scalar_multiplication(self):
        """Test scalar multiplication."""
        x_i = LazyTensor(self.x[:, None, :])
        y_j = LazyTensor(self.y[None, :, :])

        result = (2.0 * (x_i - y_j)).sum(-1).sum(1)

        if KEOPS_TORCH_AVAILABLE:
            x_t, y_t = to_torch(self.x_np), to_torch(self.y_np)
            x_i_t = LazyTensor_torch(x_t[:, None, :])
            y_j_t = LazyTensor_torch(y_t[None, :, :])
            ref = torch_to_numpy((2.0 * (x_i_t - y_j_t)).sum(-1).sum(1))
            rtol, atol = RTOL, ATOL
        else:
            diff = self.x[:, None, :] - self.y[None, :, :]
            ref = np.array(jnp.sum(jnp.sum(2.0 * diff, axis=-1), axis=1, keepdims=True))
            rtol, atol = RTOL_ELEMWISE, ATOL_ELEMWISE

        match, max_diff = compare_arrays(result, ref, rtol=rtol, atol=atol)
        self.assertTrue(match, f"Max diff: {max_diff}")


class TestLazyTensorReductions(unittest.TestCase):
    """Test LazyTensor reduction operations."""

    def setUp(self):
        self.N, self.M, self.D = SMALL_N, SMALL_M, 3
        self.x_np, self.y_np, _, _ = get_test_data(self.N, self.M, self.D)
        self.x = to_jax(self.x_np)
        self.y = to_jax(self.y_np)

    def test_lazy_sum_dim1(self):
        """Test sum over dimension 1 (j)."""
        x_i = LazyTensor(self.x[:, None, :])
        y_j = LazyTensor(self.y[None, :, :])

        result = ((x_i - y_j) ** 2).sum(-1).sum(1)

        if KEOPS_TORCH_AVAILABLE:
            x_t, y_t = to_torch(self.x_np), to_torch(self.y_np)
            x_i_t = LazyTensor_torch(x_t[:, None, :])
            y_j_t = LazyTensor_torch(y_t[None, :, :])
            ref = torch_to_numpy(((x_i_t - y_j_t) ** 2).sum(-1).sum(1))
            rtol, atol = RTOL, ATOL
        else:
            diff = self.x[:, None, :] - self.y[None, :, :]
            ref = np.array(jnp.sum(jnp.sum(diff ** 2, axis=-1), axis=1, keepdims=True))
            rtol, atol = RTOL_ELEMWISE, ATOL_ELEMWISE

        self.assertEqual(result.shape, (self.N, 1))
        match, max_diff = compare_arrays(result, ref, rtol=rtol, atol=atol)
        self.assertTrue(match, f"Max diff: {max_diff}")

    def test_lazy_sum_dim0(self):
        """Test sum over dimension 0 (i)."""
        x_i = LazyTensor(self.x[:, None, :])
        y_j = LazyTensor(self.y[None, :, :])

        result = ((x_i - y_j) ** 2).sum(-1).sum(0)

        if KEOPS_TORCH_AVAILABLE:
            x_t, y_t = to_torch(self.x_np), to_torch(self.y_np)
            x_i_t = LazyTensor_torch(x_t[:, None, :])
            y_j_t = LazyTensor_torch(y_t[None, :, :])
            ref = torch_to_numpy(((x_i_t - y_j_t) ** 2).sum(-1).sum(0))
            rtol, atol = RTOL, ATOL
        else:
            diff = self.x[:, None, :] - self.y[None, :, :]
            ref = np.array(jnp.sum(jnp.sum(diff ** 2, axis=-1), axis=0, keepdims=True).T)
            rtol, atol = RTOL_ELEMWISE, ATOL_ELEMWISE

        self.assertEqual(result.shape, (self.M, 1))
        match, max_diff = compare_arrays(result, ref, rtol=rtol, atol=atol)
        self.assertTrue(match, f"Max diff: {max_diff}")

    def test_lazy_min(self):
        """Test min reduction."""
        x_i = LazyTensor(self.x[:, None, :])
        y_j = LazyTensor(self.y[None, :, :])

        result = ((x_i - y_j) ** 2).sum(-1).min(1)

        if KEOPS_TORCH_AVAILABLE:
            x_t, y_t = to_torch(self.x_np), to_torch(self.y_np)
            x_i_t = LazyTensor_torch(x_t[:, None, :])
            y_j_t = LazyTensor_torch(y_t[None, :, :])
            ref = torch_to_numpy(((x_i_t - y_j_t) ** 2).sum(-1).min(1))
            rtol, atol = RTOL, ATOL
        else:
            diff = self.x[:, None, :] - self.y[None, :, :]
            ref = np.array(jnp.min(jnp.sum(diff ** 2, axis=-1), axis=1, keepdims=True))
            rtol, atol = RTOL_ELEMWISE, ATOL_ELEMWISE

        self.assertEqual(result.shape, (self.N, 1))
        match, max_diff = compare_arrays(result, ref, rtol=rtol, atol=atol)
        self.assertTrue(match, f"Max diff: {max_diff}")

    def test_lazy_max(self):
        """Test max reduction."""
        x_i = LazyTensor(self.x[:, None, :])
        y_j = LazyTensor(self.y[None, :, :])

        result = ((x_i - y_j) ** 2).sum(-1).max(1)

        if KEOPS_TORCH_AVAILABLE:
            x_t, y_t = to_torch(self.x_np), to_torch(self.y_np)
            x_i_t = LazyTensor_torch(x_t[:, None, :])
            y_j_t = LazyTensor_torch(y_t[None, :, :])
            ref = torch_to_numpy(((x_i_t - y_j_t) ** 2).sum(-1).max(1))
            rtol, atol = RTOL, ATOL
        else:
            diff = self.x[:, None, :] - self.y[None, :, :]
            ref = np.array(jnp.max(jnp.sum(diff ** 2, axis=-1), axis=1, keepdims=True))
            rtol, atol = RTOL_ELEMWISE, ATOL_ELEMWISE

        self.assertEqual(result.shape, (self.N, 1))
        match, max_diff = compare_arrays(result, ref, rtol=rtol, atol=atol)
        self.assertTrue(match, f"Max diff: {max_diff}")


# =============================================================================
# Vi/Vj/Pm Helper Tests
# =============================================================================

class TestViVjPmHelpers(unittest.TestCase):
    """Test Vi, Vj, Pm helper functions."""

    def setUp(self):
        self.N, self.M, self.D = SMALL_N, SMALL_M, 3
        self.x_np, self.y_np, _, _ = get_test_data(self.N, self.M, self.D)
        self.x = to_jax(self.x_np)
        self.y = to_jax(self.y_np)

    def test_vi_from_2d(self):
        """Test Vi() from 2D array."""
        x_i = Vi(self.x)
        self.assertEqual(x_i._shape, (self.N, 1, self.D))

    def test_vj_from_2d(self):
        """Test Vj() from 2D array."""
        y_j = Vj(self.y)
        self.assertEqual(y_j._shape, (1, self.M, self.D))

    def test_pm_scalar(self):
        """Test Pm() with scalar."""
        s = Pm(jnp.array([0.5]))
        self.assertEqual(s.ndim, 1)

    def test_vi_vj_computation(self):
        """Test computation with Vi and Vj helpers."""
        x_i = Vi(self.x)
        y_j = Vj(self.y)

        result = ((x_i - y_j) ** 2).sum(-1).sum(1)

        if KEOPS_TORCH_AVAILABLE:
            from pykeops.torch import Vi as Vi_torch, Vj as Vj_torch
            x_t, y_t = to_torch(self.x_np), to_torch(self.y_np)
            x_i_t = Vi_torch(x_t)
            y_j_t = Vj_torch(y_t)
            ref = torch_to_numpy(((x_i_t - y_j_t) ** 2).sum(-1).sum(1))
            rtol, atol = RTOL, ATOL
        else:
            diff = self.x[:, None, :] - self.y[None, :, :]
            ref = np.array(jnp.sum(jnp.sum(diff ** 2, axis=-1), axis=1, keepdims=True))
            rtol, atol = RTOL_ELEMWISE, ATOL_ELEMWISE

        match, max_diff = compare_arrays(result, ref, rtol=rtol, atol=atol)
        self.assertTrue(match, f"Max diff: {max_diff}")


# =============================================================================
# JIT Compilation Tests
# =============================================================================

class TestJITCompilation(unittest.TestCase):
    """Test JIT compilation."""

    def setUp(self):
        self.N, self.M, self.D = SMALL_N, SMALL_M, 3
        self.x_np, self.y_np, _, _ = get_test_data(self.N, self.M, self.D)
        self.x = to_jax(self.x_np)
        self.y = to_jax(self.y_np)

    def test_genred_jit(self):
        """Test JIT-compiled Genred."""
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Sum', axis=1)

        @jax.jit
        def compute(x, y):
            return op(x, y)

        result1 = compute(self.x, self.y)
        result2 = compute(self.x, self.y)  # Should use cached compilation

        match, max_diff = compare_arrays(result1, result2, rtol=1e-7)
        self.assertTrue(match, f"JIT results differ: {max_diff}")

    def test_lazy_jit(self):
        """Test JIT-compiled LazyTensor."""

        @jax.jit
        def compute(x, y):
            x_i = LazyTensor(x[:, None, :])
            y_j = LazyTensor(y[None, :, :])
            return ((x_i - y_j) ** 2).sum(-1).sum(1)

        result1 = compute(self.x, self.y)
        result2 = compute(self.x, self.y)

        match, max_diff = compare_arrays(result1, result2, rtol=1e-7)
        self.assertTrue(match, f"JIT results differ: {max_diff}")

    def test_jit_repeated_calls(self):
        """Test repeated JIT calls with same shapes."""
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Sum', axis=1)

        @jax.jit
        def compute(x, y):
            return op(x, y)

        results = [compute(self.x, self.y) for _ in range(5)]

        for i in range(1, len(results)):
            match, max_diff = compare_arrays(results[0], results[i], rtol=1e-7)
            self.assertTrue(match, f"Result {i} differs: {max_diff}")


# =============================================================================
# Gradient Tests
# =============================================================================

class TestGradients(unittest.TestCase):
    """Test gradient computation."""

    def setUp(self):
        self.N, self.M, self.D = 20, 15, 3  # Smaller for gradient tests
        self.x_np, self.y_np, self.b_np, _ = get_test_data(self.N, self.M, self.D)
        self.x = to_jax(self.x_np)
        self.y = to_jax(self.y_np)
        self.b = to_jax(self.b_np)

    def test_genred_grad_vi(self):
        """Test gradient w.r.t. Vi variable."""
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Sum', axis=1)

        def loss(x):
            return jnp.sum(op(x, self.y))

        grad = jax.grad(loss)(self.x)

        self.assertEqual(grad.shape, self.x.shape)
        self.assertFalse(jnp.any(jnp.isnan(grad)))

        # Compare with PyTorch if available
        if KEOPS_TORCH_AVAILABLE:
            x_t = to_torch(self.x_np).requires_grad_(True)
            y_t = to_torch(self.y_np)
            op_t = Genred_torch("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Sum', axis=1)
            loss_t = op_t(x_t, y_t).sum()
            loss_t.backward()
            ref_grad = torch_to_numpy(x_t.grad)

            match, max_diff = compare_arrays(grad, ref_grad, rtol=RTOL, atol=ATOL)
            self.assertTrue(match, f"Gradient diff: {max_diff}")

    def test_genred_grad_vj(self):
        """Test gradient w.r.t. Vj variable."""
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Sum', axis=1)

        def loss(y):
            return jnp.sum(op(self.x, y))

        grad = jax.grad(loss)(self.y)

        self.assertEqual(grad.shape, self.y.shape)
        self.assertFalse(jnp.any(jnp.isnan(grad)))

        if KEOPS_TORCH_AVAILABLE:
            x_t = to_torch(self.x_np)
            y_t = to_torch(self.y_np).requires_grad_(True)
            op_t = Genred_torch("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Sum', axis=1)
            loss_t = op_t(x_t, y_t).sum()
            loss_t.backward()
            ref_grad = torch_to_numpy(y_t.grad)

            match, max_diff = compare_arrays(grad, ref_grad, rtol=RTOL, atol=ATOL)
            self.assertTrue(match, f"Gradient diff: {max_diff}")

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
            return jnp.sum(((x_i - y_j) ** 2).sum(-1).sum(1))

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
        self.x_np, self.y_np, self.b_np, _ = get_batched_test_data(self.B, self.N, self.M, self.D)
        self.x = to_jax(self.x_np)
        self.y = to_jax(self.y_np)
        self.b = to_jax(self.b_np)

    def test_vmap_genred(self):
        """Test vmap over Genred."""
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Sum', axis=1)

        result = jax.vmap(op)(self.x, self.y)

        self.assertEqual(result.shape, (self.B, self.N, 1))

    def test_batched_3d_input(self):
        """Test 3D batched input directly."""
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Sum', axis=1)

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
# Large Matrix Tests
# =============================================================================

class TestLargeMatrices(unittest.TestCase):
    """Test with large matrices."""

    def test_large_sum_reduction(self):
        """Test Sum reduction with large matrices."""
        N, M, D = LARGE_N, LARGE_M, 3
        x_np, y_np, _, _ = get_test_data(N, M, D)
        x, y = to_jax(x_np), to_jax(y_np)

        formula = "SqDist(x, y)"
        aliases = [f"x=Vi({D})", f"y=Vj({D})"]

        op = Genred(formula, aliases, 'Sum', axis=1)
        result = op(x, y)

        self.assertEqual(result.shape, (N, 1))

        # Compare with PyTorch if available
        ref = pytorch_genred_reference(formula, aliases, 'Sum', 1, x_np, y_np)
        if ref is not None:
            match, max_diff = compare_arrays(result, ref, rtol=RTOL, atol=ATOL)
            self.assertTrue(match, f"Max diff: {max_diff}")

    def test_large_gaussian_kernel(self):
        """Test Gaussian kernel with large matrices."""
        N, M, D = LARGE_N, LARGE_M, 3
        x_np, y_np, _, _ = get_test_data(N, M, D)
        x, y = to_jax(x_np), to_jax(y_np)

        formula = "Exp(-SqNorm2(x-y) * s)"
        aliases = [f"x=Vi({D})", f"y=Vj({D})", "s=Pm(1)"]
        s = jnp.array([1.0])
        s_np = np.array(s)

        op = Genred(formula, aliases, 'Sum', axis=1)
        result = op(x, y, s)

        self.assertEqual(result.shape, (N, 1))

        ref = pytorch_genred_reference(formula, aliases, 'Sum', 1, x_np, y_np, s_np)
        if ref is not None:
            match, max_diff = compare_arrays(result, ref, rtol=RTOL, atol=ATOL)
            self.assertTrue(match, f"Max diff: {max_diff}")

    def test_large_kernel_weighted_sum(self):
        """Test kernel-weighted sum with large matrices."""
        N, M, D = LARGE_N, LARGE_M, 3
        x_np, y_np, b_np, _ = get_test_data(N, M, D)
        x, y, b = to_jax(x_np), to_jax(y_np), to_jax(b_np)

        formula = "Exp(-SqNorm2(x-y) * s) * b"
        aliases = [f"x=Vi({D})", f"y=Vj({D})", f"b=Vj({D})", "s=Pm(1)"]
        s = jnp.array([1.0])
        s_np = np.array(s)

        op = Genred(formula, aliases, 'Sum', axis=1)
        result = op(x, y, b, s)

        self.assertEqual(result.shape, (N, D))

        ref = pytorch_genred_reference(formula, aliases, 'Sum', 1, x_np, y_np, b_np, s_np)
        if ref is not None:
            match, max_diff = compare_arrays(result, ref, rtol=RTOL, atol=ATOL)
            self.assertTrue(match, f"Max diff: {max_diff}")

    def test_large_lazy_tensor(self):
        """Test LazyTensor with large matrices."""
        N, M, D = LARGE_N, LARGE_M, 3
        x_np, y_np, _, _ = get_test_data(N, M, D)
        x, y = to_jax(x_np), to_jax(y_np)

        x_i = LazyTensor(x[:, None, :])
        y_j = LazyTensor(y[None, :, :])

        result = ((x_i - y_j) ** 2).sum(-1).sum(1)

        self.assertEqual(result.shape, (N, 1))

        if KEOPS_TORCH_AVAILABLE:
            x_t, y_t = to_torch(x_np), to_torch(y_np)
            x_i_t = LazyTensor_torch(x_t[:, None, :])
            y_j_t = LazyTensor_torch(y_t[None, :, :])
            ref = torch_to_numpy(((x_i_t - y_j_t) ** 2).sum(-1).sum(1))

            match, max_diff = compare_arrays(result, ref, rtol=RTOL, atol=ATOL)
            self.assertTrue(match, f"Max diff: {max_diff}")

    def test_large_gradient(self):
        """Test gradient with large matrices."""
        N, M, D = 1000, 800, 3  # Moderate size for gradient test
        x_np, y_np, _, _ = get_test_data(N, M, D)
        x, y = to_jax(x_np), to_jax(y_np)

        op = Genred("SqDist(x, y)", [f"x=Vi({D})", f"y=Vj({D})"], 'Sum', axis=1)

        def loss(x):
            return jnp.sum(op(x, y))

        grad = jax.grad(loss)(x)

        self.assertEqual(grad.shape, x.shape)
        self.assertFalse(jnp.any(jnp.isnan(grad)))

        if KEOPS_TORCH_AVAILABLE:
            x_t = to_torch(x_np).requires_grad_(True)
            y_t = to_torch(y_np)
            op_t = Genred_torch("SqDist(x, y)", [f"x=Vi({D})", f"y=Vj({D})"], 'Sum', axis=1)
            loss_t = op_t(x_t, y_t).sum()
            loss_t.backward()
            ref_grad = torch_to_numpy(x_t.grad)

            match, max_diff = compare_arrays(grad, ref_grad, rtol=RTOL, atol=ATOL)
            self.assertTrue(match, f"Gradient diff: {max_diff}")


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

        x_np, y_np, _, _ = get_test_data(N, M, D)
        x, y = to_jax(x_np), to_jax(y_np)

        formula = "SqDist(x, y)"
        aliases = [f"x=Vi({D})", f"y=Vj({D})"]

        op = Genred(formula, aliases, 'Sum', axis=1)
        result = op(x, y)

        self.assertEqual(result.shape, (N, 1))

        ref = pytorch_genred_reference(formula, aliases, 'Sum', 1, x_np, y_np)
        if ref is not None:
            match, max_diff = compare_arrays(result, ref, rtol=RTOL, atol=ATOL)
            self.assertTrue(match, f"Max diff: {max_diff}")


# =============================================================================
# Test Runner
# =============================================================================

def run_api_tests():
    """Run all API tests with nice output."""
    if not KEOPS_JAX_AVAILABLE:
        print(f"{Colors.RED}KeOps JAX not available. Cannot run tests.{Colors.RESET}")
        return False

    print_header("KeOps JAX API Tests")

    if KEOPS_TORCH_AVAILABLE:
        print(f"  {Colors.GREEN}Ground truth: PyTorch KeOps{Colors.RESET}")
        print(f"  Tolerances: rtol={RTOL}, atol={ATOL}")
    else:
        print(f"  {Colors.YELLOW}Ground truth: Element-wise JAX (PyTorch not available){Colors.RESET}")
        print(f"  Tolerances: rtol={RTOL_ELEMWISE}, atol={ATOL_ELEMWISE}")
    print()

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
        TestLargeMatrices,
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
