#!/usr/bin/env python3
"""
KeOps JAX Advanced Features Tests
=================================
Tests for advanced KeOps features not covered in the basic API tests.

Tests cover:
- Advanced Reductions (LogSumExp, KMin, ArgKMin)
- Exotic Math Operations (Trig, Step, Abs, Sign)
- Complex Broadcasting (mixing batch dims)
- Higher-Order Gradients (Hessians)
- SoftMax stability

These tests ensure the specific CUDA kernels for these operations 
are correctly bound to JAX.
"""

import sys
import unittest
import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy.special

from test_utils import (
    TestSuite, Status, print_header, print_subheader, print_info,
    compare_arrays, run_test, print_environment_info, RICH_AVAILABLE
)

# =============================================================================
# Import KeOps
# =============================================================================

try:
    from pykeops.jax import Genred, LazyTensor
    KEOPS_AVAILABLE = True
except ImportError as e:
    print(f"Error: pykeops.jax not found: {e}")
    sys.exit(1)

# Optional PyTorch for ground truth
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
RTOL_LOOSE = 1e-3  # For operations with different accumulation order
ATOL_LOOSE = 1e-3


# =============================================================================
# Test Helpers
# =============================================================================

def generate_data(n, m, d, batch=None, seed=SEED):
    """Generate random test data, optionally batched."""
    np.random.seed(seed)
    
    if batch:
        shape_x = (batch, n, d)
        shape_y = (batch, m, d)
    else:
        shape_x = (n, d)
        shape_y = (m, d)
        
    return {
        'x': jnp.array(np.random.randn(*shape_x).astype(np.float32)),
        'y': jnp.array(np.random.randn(*shape_y).astype(np.float32)),
        'sigma': jnp.array([0.5], dtype=jnp.float32),
    }


# =============================================================================
# Advanced Reductions
# =============================================================================

class TestAdvancedReductions(unittest.TestCase):
    """Tests for specialized reductions like LogSumExp and KMin."""
    
    def setUp(self):
        self.data = generate_data(100, 80, 3)
        self.D = 3
    
    def test_logsumexp(self):
        """
        Test LogSumExp reduction.
        Critical for stable SoftMax implementations.
        Formula: LogSumExp(-|x-y|^2)
        """
        formula = "-SqDist(x, y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        
        op = Genred(formula, aliases, reduction_op='LogSumExp', axis=1)
        result = op(self.data['x'], self.data['y'])
        
        # Pure JAX Reference
        diff = self.data['x'][:, None, :] - self.data['y'][None, :, :]
        sqdist = jnp.sum(diff ** 2, axis=-1)
        expected = jax.scipy.special.logsumexp(-sqdist, axis=1, keepdims=True)
        
        match, max_diff = compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)
        self.assertTrue(match, f"Max diff: {max_diff}")

    def test_logsumexp_stability(self):
        """
        Test LogSumExp numerical stability on large values.
        Should not overflow/underflow due to max-subtraction trick.
        """
        # Create large values that would overflow naive exp()
        np.random.seed(SEED)
        x_large = jnp.array(np.random.randn(50, 3).astype(np.float32)) * 50
        y_large = jnp.array(np.random.randn(40, 3).astype(np.float32)) * 50
        
        formula = "-SqDist(x, y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        op = Genred(formula, aliases, reduction_op='LogSumExp', axis=1)
        result = op(x_large, y_large)
        
        # Check result is finite (no overflow/underflow)
        self.assertTrue(jnp.all(jnp.isfinite(result)), "LogSumExp produced non-finite values")
        
        # Reference
        diff = x_large[:, None, :] - y_large[None, :, :]
        sqdist = jnp.sum(diff ** 2, axis=-1)
        expected = jax.scipy.special.logsumexp(-sqdist, axis=1, keepdims=True)
        
        match, max_diff = compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)
        self.assertTrue(match, f"Stability test max diff: {max_diff}")

    def test_kmin(self):
        """Test KMin reduction (K-smallest values, for KNN)."""
        K = 5
        formula = "SqDist(x,y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        
        op = Genred(formula, aliases, reduction_op='KMin', axis=1, opt_arg=K)
        result = op(self.data['x'], self.data['y'])
        
        # Pure JAX Reference
        diff = self.data['x'][:, None, :] - self.data['y'][None, :, :]
        sqdist = jnp.sum(diff ** 2, axis=-1)
        expected = jnp.sort(sqdist, axis=1)[:, :K]
        
        self.assertEqual(result.shape, (100, K))
        match, max_diff = compare_arrays(result, expected, rtol=RTOL, atol=ATOL)
        self.assertTrue(match, f"Max diff: {max_diff}")

    def test_argkmin(self):
        """Test ArgKMin reduction (K-nearest neighbor indices)."""
        K = 5
        formula = "SqDist(x,y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        
        op = Genred(formula, aliases, reduction_op='ArgKMin', axis=1, opt_arg=K)
        result = op(self.data['x'], self.data['y'])
        
        # Pure JAX Reference
        diff = self.data['x'][:, None, :] - self.data['y'][None, :, :]
        sqdist = jnp.sum(diff ** 2, axis=-1)
        expected = jnp.argsort(sqdist, axis=1)[:, :K]
        
        self.assertEqual(result.shape, (100, K))
        # Indices should match exactly (convert to int for comparison)
        match, max_diff = compare_arrays(result.astype(jnp.int32), expected.astype(jnp.int32), rtol=0, atol=0)
        self.assertTrue(match, f"Indices mismatch")

    def test_kmin_vs_pytorch(self):
        """Test KMin against PyTorch ground truth."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        
        K = 5
        np.random.seed(SEED)
        x_np = np.random.randn(100, 3).astype(np.float32)
        y_np = np.random.randn(80, 3).astype(np.float32)
        
        formula = "SqDist(x,y)"
        aliases = ["x=Vi(3)", "y=Vj(3)"]
        
        # JAX
        op_jax = Genred(formula, aliases, reduction_op='KMin', axis=1, opt_arg=K)
        result_jax = op_jax(jnp.array(x_np), jnp.array(y_np))
        
        # PyTorch
        op_torch = Genred_torch(formula, aliases, reduction_op='KMin', axis=1, opt_arg=K)
        result_torch = op_torch(torch.tensor(x_np, device='cuda'), 
                                torch.tensor(y_np, device='cuda'))
        
        match, max_diff = compare_arrays(result_jax, result_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)
        self.assertTrue(match, f"KMin JAX vs PyTorch diff: {max_diff}")


# =============================================================================
# Exotic Math Operations
# =============================================================================

class TestExoticMath(unittest.TestCase):
    """Test mathematical operations beyond simple arithmetic."""
    
    def test_trigonometry(self):
        """Test Sin, Cos operations via LazyTensor."""
        x = jnp.linspace(-np.pi, np.pi, 100)[:, None]  # (100, 1)
        y = jnp.array([[0.0]])  # (1, 1)
        
        x_i = LazyTensor(x[:, None, :])  # (100, 1, 1)
        y_j = LazyTensor(y[None, :, :])  # (1, 1, 1)
        
        # Formula: sin(x - y) + cos(x)
        result = ((x_i - y_j).sin() + x_i.cos()).sum(axis=1)
        
        # Reference
        expected = jnp.sin(x - y) + jnp.cos(x)
        
        match, max_diff = compare_arrays(result, expected, rtol=RTOL, atol=ATOL)
        self.assertTrue(match, f"Trig max diff: {max_diff}")

    def test_abs_sign(self):
        """Test Abs and Sign functions."""
        x = jnp.array([[-2.0], [-0.5], [0.5], [2.0]])
        y = jnp.array([[0.0]])
        
        x_i = LazyTensor(x[:, None, :])
        y_j = LazyTensor(y[None, :, :])
        
        # Formula: abs(x) * sign(x) should equal x
        result = (x_i.abs() * x_i.sign()).sum(axis=1)
        
        # Reference: abs(x) * sign(x) = x (for x != 0)
        expected = x
        
        match, max_diff = compare_arrays(result, expected, rtol=RTOL, atol=ATOL)
        self.assertTrue(match, f"Abs*Sign max diff: {max_diff}")

    def test_step_function(self):
        """Test Step (Heaviside) function."""
        x = jnp.array([[-1.0], [-0.1], [0.1], [1.0]])
        
        x_i = LazyTensor(x[:, None, :])
        y_j = LazyTensor(jnp.array([[0.0]])[None, :, :])
        
        # KeOps Step(x) = 1 if x >= 0, else 0
        result = x_i.step().sum(axis=1)
        
        # Reference
        expected = jnp.where(x >= 0, 1.0, 0.0)
        
        match, max_diff = compare_arrays(result, expected, rtol=RTOL, atol=ATOL)
        self.assertTrue(match, f"Step max diff: {max_diff}")

    def test_clamp(self):
        """Test Clamp (clipping) operation."""
        x = jnp.array([[-2.0], [-0.5], [0.5], [2.0]])
        
        x_i = LazyTensor(x[:, None, :])
        y_j = LazyTensor(jnp.array([[0.0]])[None, :, :])
        
        # Clamp to [-1, 1]
        result = x_i.clamp(-1.0, 1.0).sum(axis=1)
        
        # Reference
        expected = jnp.clip(x, -1.0, 1.0)
        
        match, max_diff = compare_arrays(result, expected, rtol=RTOL, atol=ATOL)
        self.assertTrue(match, f"Clamp max diff: {max_diff}")

    def test_power_operations(self):
        """Test power and sqrt operations."""
        x = jnp.array([[0.5], [1.0], [2.0], [4.0]])
        
        x_i = LazyTensor(x[:, None, :])
        y_j = LazyTensor(jnp.array([[0.0]])[None, :, :])
        
        # sqrt(x) and x^2
        result_sqrt = x_i.sqrt().sum(axis=1)
        result_sq = (x_i ** 2).sum(axis=1)
        
        expected_sqrt = jnp.sqrt(x)
        expected_sq = x ** 2
        
        match1, diff1 = compare_arrays(result_sqrt, expected_sqrt, rtol=RTOL, atol=ATOL)
        match2, diff2 = compare_arrays(result_sq, expected_sq, rtol=RTOL, atol=ATOL)
        
        self.assertTrue(match1, f"Sqrt max diff: {diff1}")
        self.assertTrue(match2, f"Square max diff: {diff2}")


# =============================================================================
# Complex Broadcasting
# =============================================================================

class TestBroadcasting(unittest.TestCase):
    """Test broadcasting scenarios."""
    
    def test_batch_vs_single(self):
        """
        Test batched x (B, N, D) against single y (M, D).
        Common use case: compare batch of queries against single database.
        """
        B, N, M, D = 4, 50, 40, 3
        np.random.seed(SEED)
        
        x = jnp.array(np.random.randn(B, N, D).astype(np.float32))
        y = jnp.array(np.random.randn(M, D).astype(np.float32))  # No batch dim
        
        # LazyTensor approach
        x_i = LazyTensor(x[:, :, None, :])  # (B, N, 1, D)
        y_j = LazyTensor(y[None, None, :, :])  # (1, 1, M, D) - broadcast batch
        
        result = ((x_i - y_j) ** 2).sum(-1).sum(axis=2)
        
        # Reference: need to broadcast y to match x's batch
        diff = x[:, :, None, :] - y[None, None, :, :]  # (B, N, M, D)
        sqdist = jnp.sum(diff ** 2, axis=-1)  # (B, N, M)
        expected = jnp.sum(sqdist, axis=2, keepdims=True)  # (B, N, 1)
        
        self.assertEqual(result.shape, (B, N, 1))
        match, max_diff = compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)
        self.assertTrue(match, f"Batch vs single max diff: {max_diff}")

    def test_asymmetric_batch(self):
        """
        Test asymmetric batching: (B, N, D) vs (B, M, D) with different N, M.
        """
        B, N, M, D = 3, 100, 80, 3
        data = generate_data(N, M, D, batch=B)
        
        x_i = LazyTensor(data['x'][:, :, None, :])
        y_j = LazyTensor(data['y'][:, None, :, :])
        
        result = ((x_i - y_j) ** 2).sum(-1).sum(axis=2)
        
        # Reference
        diff = data['x'][:, :, None, :] - data['y'][:, None, :, :]
        sqdist = jnp.sum(diff ** 2, axis=-1)
        expected = jnp.sum(sqdist, axis=2, keepdims=True)
        
        self.assertEqual(result.shape, (B, N, 1))
        match, max_diff = compare_arrays(result, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)
        self.assertTrue(match, f"Asymmetric batch max diff: {max_diff}")


# =============================================================================
# Higher Order Derivatives
# =============================================================================

class TestHigherOrderGrads(unittest.TestCase):
    """Test second-order derivatives (Hessian-vector products)."""
    
    def setUp(self):
        self.data = generate_data(50, 40, 3)
        self.D = 3
        
    def test_hessian_vector_product(self):
        """
        Test 2nd order gradient: ∇²(Loss) @ v
        Verifies the backward pass graph is differentiable.
        """
        x = self.data['x']
        y = self.data['y']
        v = jnp.ones_like(x)
        
        formula = "SqDist(x, y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)
        
        def loss_fn(x_in):
            return jnp.sum(op(x_in, y))
            
        def grad_fn(x_in):
            return jax.grad(loss_fn)(x_in)
            
        # Compute Hessian-vector product via JVP of gradient
        _, hvp_result = jax.jvp(grad_fn, (x,), (v,))
        
        # Pure JAX reference
        def jax_loss(x_in):
            diff = x_in[:, None, :] - y[None, :, :]
            sqdist = jnp.sum(diff ** 2, axis=-1)
            return jnp.sum(sqdist)
            
        def jax_grad(x_in):
            return jax.grad(jax_loss)(x_in)
            
        _, hvp_expected = jax.jvp(jax_grad, (x,), (v,))
        
        match, max_diff = compare_arrays(hvp_result, hvp_expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)
        self.assertTrue(match, f"Hessian-vector product max diff: {max_diff}")

    def test_hessian_vector_product_gaussian(self):
        """Test HVP on Gaussian kernel (more complex gradient graph)."""
        x = self.data['x']
        y = self.data['y']
        sigma = self.data['sigma']
        v = jnp.ones_like(x)
        
        formula = "Exp(-SqNorm2(x-y) * s)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})", "s=Pm(1)"]
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)
        
        def loss_fn(x_in):
            return jnp.sum(op(x_in, y, sigma))
            
        def grad_fn(x_in):
            return jax.grad(loss_fn)(x_in)
            
        # Compute HVP
        _, hvp_result = jax.jvp(grad_fn, (x,), (v,))
        
        # Just verify it's finite and non-zero
        self.assertTrue(jnp.all(jnp.isfinite(hvp_result)), "HVP produced non-finite values")
        self.assertTrue(jnp.any(hvp_result != 0), "HVP is all zeros")

    def test_double_grad(self):
        """Test grad(grad(f)) directly."""
        x = self.data['x'][:10]  # Smaller for speed
        y = self.data['y'][:8]
        
        formula = "SqDist(x, y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)
        
        def loss_fn(x_in):
            return jnp.sum(op(x_in, y))
        
        # First derivative
        grad1 = jax.grad(loss_fn)(x)
        
        # Second derivative (gradient of sum of gradients)
        def grad_sum(x_in):
            return jnp.sum(jax.grad(loss_fn)(x_in))
        
        grad2 = jax.grad(grad_sum)(x)
        
        # For SqDist, the Hessian is constant (2*I per pair)
        # So grad2 should be uniform
        self.assertTrue(jnp.all(jnp.isfinite(grad2)), "Double grad produced non-finite values")


# =============================================================================
# Main Test Runner
# =============================================================================

def run_tests():
    """Run all advanced tests."""
    print_header("KeOps JAX Advanced Features", "LogSumExp, KMin, Broadcasting, Hessians")
    print_environment_info()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestAdvancedReductions,
        TestExoticMath,
        TestBroadcasting,
        TestHigherOrderGrads,
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
        print(f"{'ALL ADVANCED TESTS PASSED':^70}")
        print(f"{'='*70}\n")
        return 0
    else:
        print(f"\n{'='*70}")
        print(f"{'SOME ADVANCED TESTS FAILED':^70}")
        print(f"{'='*70}\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
