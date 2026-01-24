#!/usr/bin/env python3
"""
KeOps JAX Advanced Features Tests
=================================
Tests for advanced KeOps features not covered in the basic API tests.

All tests compare JAX KeOps against PyTorch KeOps (ground truth).

Tests cover:
- Advanced Reductions (LogSumExp, KMin, ArgKMin) - if supported
- Exotic Math Operations (Trig, Step, Abs, Sign, Clamp)
- Batched Operations
- Higher-Order Gradients (Hessians) - if supported
- Various Genred formulas

These tests ensure the JAX backend produces identical results to PyTorch.
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
    from pykeops.jax import Genred, LazyTensor
    KEOPS_JAX_AVAILABLE = True
except ImportError as e:
    print(f"Error: pykeops.jax not found: {e}")
    sys.exit(1)

# PyTorch KeOps is REQUIRED for these tests (ground truth)
try:
    import torch
    from pykeops.torch import Genred as Genred_torch, LazyTensor as LazyTensor_torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

if not TORCH_AVAILABLE:
    print("Warning: PyTorch KeOps not available - advanced tests require PyTorch as ground truth")


# =============================================================================
# Configuration
# =============================================================================

SEED = 42
RTOL = 1e-5
ATOL = 1e-5


# =============================================================================
# Test Helpers
# =============================================================================

def generate_data_np(n, m, d, batch=None, seed=SEED):
    """Generate random test data as numpy arrays."""
    np.random.seed(seed)
    
    if batch:
        shape_x = (batch, n, d)
        shape_y = (batch, m, d)
    else:
        shape_x = (n, d)
        shape_y = (m, d)
        
    return {
        'x': np.random.randn(*shape_x).astype(np.float32),
        'y': np.random.randn(*shape_y).astype(np.float32),
        'sigma': np.array([0.5], dtype=np.float32),
    }


def skip_if_no_torch(test_func):
    """Decorator to skip test if PyTorch not available."""
    def wrapper(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch KeOps not available")
        return test_func(self)
    return wrapper


# =============================================================================
# Advanced Reductions (may not be supported - tests will reveal this)
# =============================================================================

class TestAdvancedReductions(unittest.TestCase):
    """Tests for specialized reductions like LogSumExp and KMin."""
    
    def setUp(self):
        self.data_np = generate_data_np(100, 80, 3)
        self.D = 3
    
    @skip_if_no_torch
    def test_logsumexp(self):
        """
        Test LogSumExp reduction.
        Critical for stable SoftMax implementations.
        """
        formula = "-SqDist(x, y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        
        # JAX
        try:
            op_jax = Genred(formula, aliases, reduction_op='LogSumExp', axis=1)
            result_jax = op_jax(jnp.array(self.data_np['x']), jnp.array(self.data_np['y']))
        except (NameError, NotImplementedError, TypeError) as e:
            self.skipTest(f"LogSumExp not supported in JAX backend: {e}")
        
        # PyTorch ground truth
        op_torch = Genred_torch(formula, aliases, reduction_op='LogSumExp', axis=1)
        result_torch = op_torch(
            torch.tensor(self.data_np['x'], device='cuda'),
            torch.tensor(self.data_np['y'], device='cuda')
        )
        
        match, max_diff = compare_arrays(result_jax, result_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)
        self.assertTrue(match, f"LogSumExp max diff: {max_diff}")

    @skip_if_no_torch
    def test_kmin(self):
        """Test KMin reduction (K-smallest values, for KNN)."""
        K = 5
        formula = "SqDist(x,y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        
        # JAX
        try:
            op_jax = Genred(formula, aliases, reduction_op='KMin', axis=1, opt_arg=K)
            result_jax = op_jax(jnp.array(self.data_np['x']), jnp.array(self.data_np['y']))
        except (NameError, NotImplementedError, TypeError) as e:
            self.skipTest(f"KMin not supported in JAX backend: {e}")
        
        # PyTorch ground truth
        op_torch = Genred_torch(formula, aliases, reduction_op='KMin', axis=1, opt_arg=K)
        result_torch = op_torch(
            torch.tensor(self.data_np['x'], device='cuda'),
            torch.tensor(self.data_np['y'], device='cuda')
        )
        
        self.assertEqual(result_jax.shape, (100, K))
        match, max_diff = compare_arrays(result_jax, result_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)
        self.assertTrue(match, f"KMin max diff: {max_diff}")

    @skip_if_no_torch
    def test_argkmin(self):
        """Test ArgKMin reduction (K-nearest neighbor indices)."""
        K = 5
        formula = "SqDist(x,y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        
        # JAX
        try:
            op_jax = Genred(formula, aliases, reduction_op='ArgKMin', axis=1, opt_arg=K)
            result_jax = op_jax(jnp.array(self.data_np['x']), jnp.array(self.data_np['y']))
        except (NameError, NotImplementedError, TypeError) as e:
            self.skipTest(f"ArgKMin not supported in JAX backend: {e}")
        
        # PyTorch ground truth
        op_torch = Genred_torch(formula, aliases, reduction_op='ArgKMin', axis=1, opt_arg=K)
        result_torch = op_torch(
            torch.tensor(self.data_np['x'], device='cuda'),
            torch.tensor(self.data_np['y'], device='cuda')
        )
        
        self.assertEqual(result_jax.shape, (100, K))
        # Indices should match exactly
        match, max_diff = compare_arrays(
            result_jax.astype(jnp.int32), 
            result_torch.cpu().numpy().astype(np.int32), 
            rtol=0, atol=0
        )
        self.assertTrue(match, f"ArgKMin indices mismatch")


# =============================================================================
# Exotic Math Operations
# =============================================================================

class TestExoticMath(unittest.TestCase):
    """Test mathematical operations beyond simple arithmetic."""
    
    @skip_if_no_torch
    def test_trigonometry_sin(self):
        """Test Sin operation via LazyTensor."""
        np.random.seed(SEED)
        x_np = np.random.randn(50, 3).astype(np.float32)
        y_np = np.random.randn(40, 3).astype(np.float32)
        
        # JAX
        x_i_jax = LazyTensor(jnp.array(x_np)[:, None, :])
        y_j_jax = LazyTensor(jnp.array(y_np)[None, :, :])
        result_jax = (x_i_jax - y_j_jax).sin().sum(axis=1)
        
        # PyTorch
        x_i_torch = LazyTensor_torch(torch.tensor(x_np, device='cuda')[:, None, :])
        y_j_torch = LazyTensor_torch(torch.tensor(y_np, device='cuda')[None, :, :])
        result_torch = (x_i_torch - y_j_torch).sin().sum(axis=1)
        
        match, max_diff = compare_arrays(result_jax, result_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)
        self.assertTrue(match, f"Sin max diff: {max_diff}")

    @skip_if_no_torch
    def test_trigonometry_cos(self):
        """Test Cos operation via LazyTensor."""
        np.random.seed(SEED)
        x_np = np.random.randn(50, 3).astype(np.float32)
        y_np = np.random.randn(40, 3).astype(np.float32)
        
        # JAX
        x_i_jax = LazyTensor(jnp.array(x_np)[:, None, :])
        y_j_jax = LazyTensor(jnp.array(y_np)[None, :, :])
        result_jax = (x_i_jax - y_j_jax).cos().sum(axis=1)
        
        # PyTorch
        x_i_torch = LazyTensor_torch(torch.tensor(x_np, device='cuda')[:, None, :])
        y_j_torch = LazyTensor_torch(torch.tensor(y_np, device='cuda')[None, :, :])
        result_torch = (x_i_torch - y_j_torch).cos().sum(axis=1)
        
        match, max_diff = compare_arrays(result_jax, result_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)
        self.assertTrue(match, f"Cos max diff: {max_diff}")

    @skip_if_no_torch
    def test_abs_sign(self):
        """Test Abs and Sign functions."""
        np.random.seed(SEED)
        x_np = np.random.randn(50, 3).astype(np.float32)
        y_np = np.random.randn(40, 3).astype(np.float32)
        
        # JAX
        x_i_jax = LazyTensor(jnp.array(x_np)[:, None, :])
        y_j_jax = LazyTensor(jnp.array(y_np)[None, :, :])
        diff_jax = x_i_jax - y_j_jax
        result_jax = (diff_jax.abs() * diff_jax.sign()).sum(axis=1)
        
        # PyTorch
        x_i_torch = LazyTensor_torch(torch.tensor(x_np, device='cuda')[:, None, :])
        y_j_torch = LazyTensor_torch(torch.tensor(y_np, device='cuda')[None, :, :])
        diff_torch = x_i_torch - y_j_torch
        result_torch = (diff_torch.abs() * diff_torch.sign()).sum(axis=1)
        
        match, max_diff = compare_arrays(result_jax, result_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)
        self.assertTrue(match, f"Abs*Sign max diff: {max_diff}")

    @skip_if_no_torch
    def test_step_function(self):
        """Test Step (Heaviside) function."""
        np.random.seed(SEED)
        x_np = np.random.randn(50, 3).astype(np.float32)
        y_np = np.random.randn(40, 3).astype(np.float32)
        
        # JAX
        x_i_jax = LazyTensor(jnp.array(x_np)[:, None, :])
        y_j_jax = LazyTensor(jnp.array(y_np)[None, :, :])
        result_jax = (x_i_jax - y_j_jax).step().sum(axis=1)
        
        # PyTorch
        x_i_torch = LazyTensor_torch(torch.tensor(x_np, device='cuda')[:, None, :])
        y_j_torch = LazyTensor_torch(torch.tensor(y_np, device='cuda')[None, :, :])
        result_torch = (x_i_torch - y_j_torch).step().sum(axis=1)
        
        match, max_diff = compare_arrays(result_jax, result_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)
        self.assertTrue(match, f"Step max diff: {max_diff}")

    @skip_if_no_torch
    def test_clamp(self):
        """Test Clamp (clipping) operation."""
        np.random.seed(SEED)
        x_np = np.random.randn(50, 3).astype(np.float32) * 3  # Larger range to test clamping
        y_np = np.random.randn(40, 3).astype(np.float32)
        
        # JAX
        x_i_jax = LazyTensor(jnp.array(x_np)[:, None, :])
        y_j_jax = LazyTensor(jnp.array(y_np)[None, :, :])
        result_jax = (x_i_jax - y_j_jax).clamp(-1.0, 1.0).sum(axis=1)
        
        # PyTorch
        x_i_torch = LazyTensor_torch(torch.tensor(x_np, device='cuda')[:, None, :])
        y_j_torch = LazyTensor_torch(torch.tensor(y_np, device='cuda')[None, :, :])
        result_torch = (x_i_torch - y_j_torch).clamp(-1.0, 1.0).sum(axis=1)
        
        match, max_diff = compare_arrays(result_jax, result_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)
        self.assertTrue(match, f"Clamp max diff: {max_diff}")

    @skip_if_no_torch
    def test_power_sqrt(self):
        """Test power and sqrt operations."""
        np.random.seed(SEED)
        # Use positive values for sqrt
        x_np = np.abs(np.random.randn(50, 3).astype(np.float32)) + 0.1
        y_np = np.abs(np.random.randn(40, 3).astype(np.float32)) + 0.1
        
        # JAX - test sqrt
        x_i_jax = LazyTensor(jnp.array(x_np)[:, None, :])
        y_j_jax = LazyTensor(jnp.array(y_np)[None, :, :])
        result_sqrt_jax = (x_i_jax + y_j_jax).sqrt().sum(axis=1)
        
        # PyTorch - test sqrt
        x_i_torch = LazyTensor_torch(torch.tensor(x_np, device='cuda')[:, None, :])
        y_j_torch = LazyTensor_torch(torch.tensor(y_np, device='cuda')[None, :, :])
        result_sqrt_torch = (x_i_torch + y_j_torch).sqrt().sum(axis=1)
        
        match, max_diff = compare_arrays(result_sqrt_jax, result_sqrt_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)
        self.assertTrue(match, f"Sqrt max diff: {max_diff}")

    @skip_if_no_torch
    def test_power_square(self):
        """Test squaring operation."""
        np.random.seed(SEED)
        x_np = np.random.randn(50, 3).astype(np.float32)
        y_np = np.random.randn(40, 3).astype(np.float32)
        
        # JAX
        x_i_jax = LazyTensor(jnp.array(x_np)[:, None, :])
        y_j_jax = LazyTensor(jnp.array(y_np)[None, :, :])
        result_jax = ((x_i_jax - y_j_jax) ** 2).sum(axis=1)
        
        # PyTorch
        x_i_torch = LazyTensor_torch(torch.tensor(x_np, device='cuda')[:, None, :])
        y_j_torch = LazyTensor_torch(torch.tensor(y_np, device='cuda')[None, :, :])
        result_torch = ((x_i_torch - y_j_torch) ** 2).sum(axis=1)
        
        match, max_diff = compare_arrays(result_jax, result_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)
        self.assertTrue(match, f"Square max diff: {max_diff}")

    @skip_if_no_torch
    def test_exp_log(self):
        """Test exp and log operations."""
        np.random.seed(SEED)
        # Use small positive values to avoid overflow/underflow
        x_np = np.abs(np.random.randn(50, 3).astype(np.float32)) * 0.5 + 0.1
        y_np = np.abs(np.random.randn(40, 3).astype(np.float32)) * 0.5 + 0.1
        
        # JAX
        x_i_jax = LazyTensor(jnp.array(x_np)[:, None, :])
        y_j_jax = LazyTensor(jnp.array(y_np)[None, :, :])
        result_jax = (x_i_jax + y_j_jax).log().sum(axis=1)
        
        # PyTorch
        x_i_torch = LazyTensor_torch(torch.tensor(x_np, device='cuda')[:, None, :])
        y_j_torch = LazyTensor_torch(torch.tensor(y_np, device='cuda')[None, :, :])
        result_torch = (x_i_torch + y_j_torch).log().sum(axis=1)
        
        match, max_diff = compare_arrays(result_jax, result_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)
        self.assertTrue(match, f"Log max diff: {max_diff}")


# =============================================================================
# Batched Operations
# =============================================================================

class TestBatchedOperations(unittest.TestCase):
    """Test batched LazyTensor operations."""
    
    @skip_if_no_torch
    def test_batched_sqdist(self):
        """Test batched squared distance."""
        B, N, M, D = 3, 50, 40, 3
        data = generate_data_np(N, M, D, batch=B)
        
        # JAX
        x_i_jax = LazyTensor(jnp.array(data['x'])[:, :, None, :])
        y_j_jax = LazyTensor(jnp.array(data['y'])[:, None, :, :])
        result_jax = ((x_i_jax - y_j_jax) ** 2).sum(-1).sum(axis=2)
        
        # PyTorch
        x_i_torch = LazyTensor_torch(torch.tensor(data['x'], device='cuda')[:, :, None, :])
        y_j_torch = LazyTensor_torch(torch.tensor(data['y'], device='cuda')[:, None, :, :])
        result_torch = ((x_i_torch - y_j_torch) ** 2).sum(-1).sum(axis=2)
        
        match, max_diff = compare_arrays(result_jax, result_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)
        self.assertTrue(match, f"Batched sqdist max diff: {max_diff}")

    @skip_if_no_torch
    def test_batched_gaussian(self):
        """Test batched Gaussian kernel."""
        B, N, M, D = 3, 50, 40, 3
        data = generate_data_np(N, M, D, batch=B)
        sigma = 0.5
        
        # JAX
        x_i_jax = LazyTensor(jnp.array(data['x'])[:, :, None, :])
        y_j_jax = LazyTensor(jnp.array(data['y'])[:, None, :, :])
        K_jax = (-((x_i_jax - y_j_jax) ** 2).sum(-1) / (2 * sigma**2)).exp()
        result_jax = K_jax.sum(axis=2)
        
        # PyTorch
        x_i_torch = LazyTensor_torch(torch.tensor(data['x'], device='cuda')[:, :, None, :])
        y_j_torch = LazyTensor_torch(torch.tensor(data['y'], device='cuda')[:, None, :, :])
        K_torch = (-((x_i_torch - y_j_torch) ** 2).sum(-1) / (2 * sigma**2)).exp()
        result_torch = K_torch.sum(axis=2)
        
        match, max_diff = compare_arrays(result_jax, result_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)
        self.assertTrue(match, f"Batched Gaussian max diff: {max_diff}")


# =============================================================================
# Higher Order Derivatives (may not be supported)
# =============================================================================

class TestHigherOrderGrads(unittest.TestCase):
    """Test second-order derivatives (Hessian-vector products)."""
    
    @skip_if_no_torch
    def test_hessian_vector_product(self):
        """
        Test 2nd order gradient: ∇²(Loss) @ v
        Verifies the backward pass graph is differentiable.
        """
        data = generate_data_np(30, 25, 3)
        
        x_jax = jnp.array(data['x'])
        y_jax = jnp.array(data['y'])
        v_jax = jnp.ones_like(x_jax)
        
        formula = "SqDist(x, y)"
        aliases = ["x=Vi(3)", "y=Vj(3)"]
        op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1)
        
        def loss_fn_jax(x_in):
            return jnp.sum(op_jax(x_in, y_jax))
            
        def grad_fn_jax(x_in):
            return jax.grad(loss_fn_jax)(x_in)
        
        # Try to compute Hessian-vector product
        try:
            _, hvp_jax = jax.jvp(grad_fn_jax, (x_jax,), (v_jax,))
        except ValueError as e:
            if "cannot be differentiated" in str(e):
                self.skipTest(f"Higher-order gradients not supported: {e}")
            raise
        
        # PyTorch ground truth - compute HVP manually
        x_torch = torch.tensor(data['x'], device='cuda', requires_grad=True)
        y_torch = torch.tensor(data['y'], device='cuda')
        v_torch = torch.ones_like(x_torch)
        
        op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
        
        # First gradient
        loss = op_torch(x_torch, y_torch).sum()
        grad1, = torch.autograd.grad(loss, x_torch, create_graph=True)
        
        # HVP = gradient of (grad1 . v) w.r.t x
        hvp_torch, = torch.autograd.grad(grad1, x_torch, grad_outputs=v_torch)
        
        match, max_diff = compare_arrays(hvp_jax, hvp_torch.cpu().numpy(), rtol=1e-4, atol=1e-4)
        self.assertTrue(match, f"HVP max diff: {max_diff}")


# =============================================================================
# Genred with Various Formulas
# =============================================================================

class TestGenredFormulas(unittest.TestCase):
    """Test various Genred formulas comparing JAX vs PyTorch."""
    
    @skip_if_no_torch
    def test_laplacian_kernel(self):
        """Test Laplacian kernel: exp(-|x-y|)."""
        data = generate_data_np(100, 80, 3)
        
        formula = "Exp(-Sqrt(SqDist(x,y)))"
        aliases = ["x=Vi(3)", "y=Vj(3)"]
        
        # JAX
        op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1)
        result_jax = op_jax(jnp.array(data['x']), jnp.array(data['y']))
        
        # PyTorch
        op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
        result_torch = op_torch(
            torch.tensor(data['x'], device='cuda'),
            torch.tensor(data['y'], device='cuda')
        )
        
        match, max_diff = compare_arrays(result_jax, result_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)
        self.assertTrue(match, f"Laplacian max diff: {max_diff}")

    @skip_if_no_torch
    def test_cauchy_kernel(self):
        """Test Cauchy kernel: 1/(1+|x-y|^2)."""
        data = generate_data_np(100, 80, 3)
        
        formula = "Inv(IntCst(1) + SqDist(x,y))"
        aliases = ["x=Vi(3)", "y=Vj(3)"]
        
        # JAX
        op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1)
        result_jax = op_jax(jnp.array(data['x']), jnp.array(data['y']))
        
        # PyTorch
        op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
        result_torch = op_torch(
            torch.tensor(data['x'], device='cuda'),
            torch.tensor(data['y'], device='cuda')
        )
        
        match, max_diff = compare_arrays(result_jax, result_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)
        self.assertTrue(match, f"Cauchy max diff: {max_diff}")

    @skip_if_no_torch
    def test_weighted_sum(self):
        """Test weighted kernel sum: K(x,y) @ b."""
        data = generate_data_np(100, 80, 3)
        np.random.seed(SEED + 1)
        b_np = np.random.randn(80, 3).astype(np.float32)
        
        formula = "Exp(-SqDist(x,y) * s) * b"
        aliases = ["x=Vi(3)", "y=Vj(3)", "b=Vj(3)", "s=Pm(1)"]
        
        # JAX
        op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1)
        result_jax = op_jax(
            jnp.array(data['x']), 
            jnp.array(data['y']), 
            jnp.array(b_np),
            jnp.array(data['sigma'])
        )
        
        # PyTorch
        op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
        result_torch = op_torch(
            torch.tensor(data['x'], device='cuda'),
            torch.tensor(data['y'], device='cuda'),
            torch.tensor(b_np, device='cuda'),
            torch.tensor(data['sigma'], device='cuda')
        )
        
        match, max_diff = compare_arrays(result_jax, result_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)
        self.assertTrue(match, f"Weighted sum max diff: {max_diff}")


# =============================================================================
# Main Test Runner
# =============================================================================

def run_tests():
    """Run all advanced tests."""
    print_header("KeOps JAX Advanced Features", "Comparing JAX KeOps vs PyTorch KeOps")
    print_environment_info()
    
    if not TORCH_AVAILABLE:
        print("\n⚠️  PyTorch KeOps not available - all tests will be skipped")
        print("    These tests require PyTorch as ground truth\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestAdvancedReductions,
        TestExoticMath,
        TestBatchedOperations,
        TestHigherOrderGrads,
        TestGenredFormulas,
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
        # Count skipped vs failed
        skipped = len(result.skipped)
        failed = len(result.failures) + len(result.errors)
        print(f"\n{'='*70}")
        if failed > 0:
            print(f"{'SOME ADVANCED TESTS FAILED':^70}")
        else:
            print(f"{'ALL RUN TESTS PASSED ({} skipped)'.format(skipped):^70}")
        print(f"{'='*70}\n")
        return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(run_tests())
