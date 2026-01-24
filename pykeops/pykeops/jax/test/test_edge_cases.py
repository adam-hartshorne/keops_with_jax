#!/usr/bin/env python3
"""
KeOps JAX Edge Case Tests
=========================
Tests for edge cases discovered during development.

These tests specifically target bugs that were found and fixed:
1. Scalar multiplication variable ordering (LazyTensor._var_ids fix)
2. Batched vs non-batched operations
3. Complex multi-variable varifold kernels with lengthscales
4. Pm parameter gradient handling
5. High-dimensional chunking
6. LazyTensor shape handling (trailing dimensions)

All tests compare JAX KeOps against PyTorch KeOps when available.
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

# Import PyTorch KeOps for ground truth
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


# =============================================================================
# Test 1: Scalar Multiplication Variable Ordering
# =============================================================================
# This was a critical bug where `2.0 * (x_i - y_j)` would give wrong results
# because the _var_ids weren't being tracked properly through operations.

def test_scalar_multiplication():
    """Test scalar multiplication with LazyTensor (variable ordering bug fix)."""
    np.random.seed(SEED)
    N, M, D = 50, 40, 3
    
    x_np = np.random.randn(N, D).astype(np.float32)
    y_np = np.random.randn(M, D).astype(np.float32)
    
    x = jnp.array(x_np)
    y = jnp.array(y_np)
    
    # Create LazyTensors
    x_i = LazyTensor(x[:, None, :])
    y_j = LazyTensor(y[None, :, :])
    
    # Compute: 2.0 * (x - y) with scalar on LEFT (was buggy)
    diff = x_i - y_j
    result_left = (2.0 * diff).sum(-1).sum(axis=1)
    
    # Compute: (x - y) * 2.0 with scalar on RIGHT
    result_right = (diff * 2.0).sum(-1).sum(axis=1)
    
    # Expected result (pure JAX)
    expected = jnp.sum(2.0 * (x[:, None, :] - y[None, :, :]), axis=(1, 2), keepdims=False)
    expected = expected[:, None]  # Add trailing dim to match KeOps
    
    # Compare
    jax_left = np.squeeze(np.array(result_left))
    jax_right = np.squeeze(np.array(result_right))
    expected_np = np.squeeze(np.array(expected))
    
    match_left, diff_left = compare_arrays(jax_left, expected_np, rtol=RTOL, atol=ATOL)
    match_right, diff_right = compare_arrays(jax_right, expected_np, rtol=RTOL, atol=ATOL)
    match_lr, diff_lr = compare_arrays(jax_left, jax_right, rtol=RTOL, atol=ATOL)
    
    passed = match_left and match_right and match_lr
    max_diff = max(diff_left, diff_right, diff_lr)
    
    return passed, max_diff


def test_scalar_multiplication_pytorch():
    """Test scalar multiplication against PyTorch ground truth."""
    if not TORCH_AVAILABLE:
        return None  # Skip
    
    np.random.seed(SEED)
    N, M, D = 50, 40, 3
    
    x_np = np.random.randn(N, D).astype(np.float32)
    y_np = np.random.randn(M, D).astype(np.float32)
    
    # JAX
    x_jax = jnp.array(x_np)
    y_jax = jnp.array(y_np)
    x_i = LazyTensor(x_jax[:, None, :])
    y_j = LazyTensor(y_jax[None, :, :])
    result_jax = (2.0 * (x_i - y_j)).sum(-1).sum(axis=1)
    
    # PyTorch
    x_torch = torch.tensor(x_np, device='cuda')
    y_torch = torch.tensor(y_np, device='cuda')
    x_i_t = LazyTensor_torch(x_torch[:, None, :])
    y_j_t = LazyTensor_torch(y_torch[None, :, :])
    result_torch = (2.0 * (x_i_t - y_j_t)).sum(-1).sum(axis=1)
    
    jax_np = np.squeeze(np.array(result_jax))
    torch_np = np.squeeze(result_torch.cpu().numpy())
    
    return compare_arrays(jax_np, torch_np, rtol=RTOL, atol=ATOL)


# =============================================================================
# Test 2: Batched vs Non-Batched Operations
# =============================================================================

def test_non_batched_operations():
    """Test non-batched (2D input) LazyTensor operations."""
    np.random.seed(SEED)
    N, M, D = 100, 80, 3
    
    x_np = np.random.randn(N, D).astype(np.float32)
    y_np = np.random.randn(M, D).astype(np.float32)
    
    x = jnp.array(x_np)
    y = jnp.array(y_np)
    
    # Pure JAX reference
    jax_dist = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    jax_result = jax_dist.sum(axis=1)
    
    # KeOps
    x_i = LazyTensor(x[:, None, :])
    y_j = LazyTensor(y[None, :, :])
    D_ij = ((x_i - y_j) ** 2).sum(-1)
    keops_result = D_ij.sum(axis=1)
    
    # Compare (squeeze KeOps result as it has trailing dim)
    return compare_arrays(jax_result, keops_result, rtol=RTOL, atol=ATOL, squeeze=True)


def test_batched_operations():
    """Test batched (3D input) LazyTensor operations."""
    np.random.seed(SEED)
    B, N, M, D = 4, 100, 80, 3
    
    x_np = np.random.randn(B, N, D).astype(np.float32)
    y_np = np.random.randn(B, M, D).astype(np.float32)
    
    x = jnp.array(x_np)
    y = jnp.array(y_np)
    
    # Pure JAX reference
    jax_dist = jnp.sum((x[:, :, None, :] - y[:, None, :, :]) ** 2, axis=-1)
    jax_result = jax_dist.sum(axis=2)
    
    # KeOps
    x_i = LazyTensor(x[:, :, None, :])
    y_j = LazyTensor(y[:, None, :, :])
    D_ij = ((x_i - y_j) ** 2).sum(-1)
    keops_result = D_ij.sum(axis=2)
    
    # Compare
    return compare_arrays(jax_result, keops_result, rtol=RTOL, atol=ATOL, squeeze=True)


def test_batched_vs_pytorch():
    """Test batched operations against PyTorch ground truth."""
    if not TORCH_AVAILABLE:
        return None
    
    np.random.seed(SEED)
    B, N, M, D = 4, 100, 80, 3
    
    x_np = np.random.randn(B, N, D).astype(np.float32)
    y_np = np.random.randn(B, M, D).astype(np.float32)
    
    # JAX
    x_jax = jnp.array(x_np)
    y_jax = jnp.array(y_np)
    x_i = LazyTensor(x_jax[:, :, None, :])
    y_j = LazyTensor(y_jax[:, None, :, :])
    result_jax = ((x_i - y_j) ** 2).sum(-1).sum(axis=2)
    
    # PyTorch
    x_torch = torch.tensor(x_np, device='cuda')
    y_torch = torch.tensor(y_np, device='cuda')
    x_i_t = LazyTensor_torch(x_torch[:, :, None, :])
    y_j_t = LazyTensor_torch(y_torch[:, None, :, :])
    result_torch = ((x_i_t - y_j_t) ** 2).sum(-1).sum(axis=2)
    
    return compare_arrays(result_jax, result_torch.cpu().numpy(), rtol=RTOL, atol=ATOL, squeeze=True)


# =============================================================================
# Test 3: Complex Varifold Kernel with Lengthscales
# =============================================================================
# This tests the multi-variable case with 6 variables and scalar divisions

def test_varifold_forward():
    """Test varifold kernel forward pass with multiple lengthscales."""
    np.random.seed(SEED)
    batch_size, n_i, n_j = 2, 10, 8
    
    # Generate data
    x_np = np.random.randn(batch_size, n_i, 3).astype(np.float32)
    y_np = np.random.randn(batch_size, n_j, 3).astype(np.float32)
    nx_np = np.random.randn(batch_size, n_i, 3).astype(np.float32)
    ny_np = np.random.randn(batch_size, n_j, 3).astype(np.float32)
    wi_np = np.random.randn(batch_size, n_i, 1).astype(np.float32)
    wj_np = np.random.randn(batch_size, n_j, 1).astype(np.float32)
    
    # Normalize normals
    nx_np = nx_np / (np.linalg.norm(nx_np, axis=-1, keepdims=True) + 1e-8)
    ny_np = ny_np / (np.linalg.norm(ny_np, axis=-1, keepdims=True) + 1e-8)
    
    sigma_pos, sigma_norm = 1.5, 0.8
    
    # Convert to JAX
    x, y = jnp.array(x_np), jnp.array(y_np)
    nx, ny = jnp.array(nx_np), jnp.array(ny_np)
    wi, wj = jnp.array(wi_np), jnp.array(wj_np)
    
    # LazyTensor version
    x_i = LazyTensor(x.reshape(batch_size, n_i, 1, 3))
    y_j = LazyTensor(y.reshape(batch_size, 1, n_j, 3))
    nx_i = LazyTensor(nx.reshape(batch_size, n_i, 1, 3))
    ny_j = LazyTensor(ny.reshape(batch_size, 1, n_j, 3))
    wi_i = LazyTensor(wi.reshape(batch_size, n_i, 1, 1))
    wj_j = LazyTensor(wj.reshape(batch_size, 1, n_j, 1))
    
    sq_dist_pos = ((x_i - y_j) ** 2).sum(-1)
    sq_dist_norm = ((nx_i - ny_j) ** 2).sum(-1)
    K_ij = wi_i * wj_j * (-(sq_dist_pos / (sigma_pos ** 2))).exp() * (-(sq_dist_norm / (sigma_norm ** 2))).exp()
    result_lazy = K_ij.sum(dim=2)
    
    # Pure JAX reference
    sq_dist_pos_ref = jnp.sum((x[:, :, None, :] - y[:, None, :, :]) ** 2, axis=-1)
    sq_dist_norm_ref = jnp.sum((nx[:, :, None, :] - ny[:, None, :, :]) ** 2, axis=-1)
    K_ref = wi[:, :, None, :] * wj[:, None, :, :] * jnp.exp(-sq_dist_pos_ref[:, :, :, None] / (sigma_pos ** 2)) * jnp.exp(-sq_dist_norm_ref[:, :, :, None] / (sigma_norm ** 2))
    result_ref = K_ref.sum(axis=2)
    
    return compare_arrays(result_lazy, result_ref, rtol=RTOL, atol=ATOL, squeeze=True)


def test_varifold_gradient():
    """Test varifold kernel gradient computation."""
    np.random.seed(SEED)
    batch_size, n_i, n_j = 2, 10, 8
    
    # Generate data
    x_np = np.random.randn(batch_size, n_i, 3).astype(np.float32)
    y_np = np.random.randn(batch_size, n_j, 3).astype(np.float32)
    nx_np = np.random.randn(batch_size, n_i, 3).astype(np.float32)
    ny_np = np.random.randn(batch_size, n_j, 3).astype(np.float32)
    wi_np = np.random.randn(batch_size, n_i, 1).astype(np.float32)
    wj_np = np.random.randn(batch_size, n_j, 1).astype(np.float32)
    
    nx_np = nx_np / (np.linalg.norm(nx_np, axis=-1, keepdims=True) + 1e-8)
    ny_np = ny_np / (np.linalg.norm(ny_np, axis=-1, keepdims=True) + 1e-8)
    
    sigma_pos, sigma_norm = 1.5, 0.8
    
    x, y = jnp.array(x_np), jnp.array(y_np)
    nx, ny = jnp.array(nx_np), jnp.array(ny_np)
    wi, wj = jnp.array(wi_np), jnp.array(wj_np)
    
    def forward_lazy(x_var):
        x_i = LazyTensor(x_var.reshape(batch_size, n_i, 1, 3))
        y_j = LazyTensor(y.reshape(batch_size, 1, n_j, 3))
        nx_i = LazyTensor(nx.reshape(batch_size, n_i, 1, 3))
        ny_j = LazyTensor(ny.reshape(batch_size, 1, n_j, 3))
        wi_i = LazyTensor(wi.reshape(batch_size, n_i, 1, 1))
        wj_j = LazyTensor(wj.reshape(batch_size, 1, n_j, 1))
        
        sq_dist_pos = ((x_i - y_j) ** 2).sum(-1)
        sq_dist_norm = ((nx_i - ny_j) ** 2).sum(-1)
        K_ij = wi_i * wj_j * (-(sq_dist_pos / (sigma_pos ** 2))).exp() * (-(sq_dist_norm / (sigma_norm ** 2))).exp()
        return K_ij.sum(dim=2).sum()
    
    def forward_ref(x_var):
        sq_dist_pos = jnp.sum((x_var[:, :, None, :] - y[:, None, :, :]) ** 2, axis=-1)
        sq_dist_norm = jnp.sum((nx[:, :, None, :] - ny[:, None, :, :]) ** 2, axis=-1)
        K = wi[:, :, None, :] * wj[:, None, :, :] * jnp.exp(-sq_dist_pos[:, :, :, None] / (sigma_pos ** 2)) * jnp.exp(-sq_dist_norm[:, :, :, None] / (sigma_norm ** 2))
        return K.sum()
    
    grad_lazy = jax.grad(forward_lazy)(x)
    grad_ref = jax.grad(forward_ref)(x)
    
    return compare_arrays(grad_lazy, grad_ref, rtol=1e-3, atol=1e-3)


def test_varifold_vs_pytorch():
    """Test varifold kernel against PyTorch ground truth."""
    if not TORCH_AVAILABLE:
        return None
    
    np.random.seed(SEED)
    batch_size, n_i, n_j = 2, 10, 8
    
    x_np = np.random.randn(batch_size, n_i, 3).astype(np.float32)
    y_np = np.random.randn(batch_size, n_j, 3).astype(np.float32)
    nx_np = np.random.randn(batch_size, n_i, 3).astype(np.float32)
    ny_np = np.random.randn(batch_size, n_j, 3).astype(np.float32)
    wi_np = np.random.randn(batch_size, n_i, 1).astype(np.float32)
    wj_np = np.random.randn(batch_size, n_j, 1).astype(np.float32)
    
    nx_np = nx_np / (np.linalg.norm(nx_np, axis=-1, keepdims=True) + 1e-8)
    ny_np = ny_np / (np.linalg.norm(ny_np, axis=-1, keepdims=True) + 1e-8)
    
    sigma_pos, sigma_norm = 1.5, 0.8
    
    # JAX
    x, y = jnp.array(x_np), jnp.array(y_np)
    nx, ny = jnp.array(nx_np), jnp.array(ny_np)
    wi, wj = jnp.array(wi_np), jnp.array(wj_np)
    
    x_i = LazyTensor(x.reshape(batch_size, n_i, 1, 3))
    y_j = LazyTensor(y.reshape(batch_size, 1, n_j, 3))
    nx_i = LazyTensor(nx.reshape(batch_size, n_i, 1, 3))
    ny_j = LazyTensor(ny.reshape(batch_size, 1, n_j, 3))
    wi_i = LazyTensor(wi.reshape(batch_size, n_i, 1, 1))
    wj_j = LazyTensor(wj.reshape(batch_size, 1, n_j, 1))
    
    sq_dist_pos = ((x_i - y_j) ** 2).sum(-1)
    sq_dist_norm = ((nx_i - ny_j) ** 2).sum(-1)
    K_ij = wi_i * wj_j * (-(sq_dist_pos / (sigma_pos ** 2))).exp() * (-(sq_dist_norm / (sigma_norm ** 2))).exp()
    result_jax = K_ij.sum(dim=2)
    
    # PyTorch
    x_t = torch.tensor(x_np, device='cuda')
    y_t = torch.tensor(y_np, device='cuda')
    nx_t = torch.tensor(nx_np, device='cuda')
    ny_t = torch.tensor(ny_np, device='cuda')
    wi_t = torch.tensor(wi_np, device='cuda')
    wj_t = torch.tensor(wj_np, device='cuda')
    
    x_i_t = LazyTensor_torch(x_t.view(batch_size, n_i, 1, 3))
    y_j_t = LazyTensor_torch(y_t.view(batch_size, 1, n_j, 3))
    nx_i_t = LazyTensor_torch(nx_t.view(batch_size, n_i, 1, 3))
    ny_j_t = LazyTensor_torch(ny_t.view(batch_size, 1, n_j, 3))
    wi_i_t = LazyTensor_torch(wi_t.view(batch_size, n_i, 1, 1))
    wj_j_t = LazyTensor_torch(wj_t.view(batch_size, 1, n_j, 1))
    
    sq_dist_pos_t = ((x_i_t - y_j_t) ** 2).sum(-1)
    sq_dist_norm_t = ((nx_i_t - ny_j_t) ** 2).sum(-1)
    K_ij_t = wi_i_t * wj_j_t * (-(sq_dist_pos_t / (sigma_pos ** 2))).exp() * (-(sq_dist_norm_t / (sigma_norm ** 2))).exp()
    result_torch = K_ij_t.sum(dim=2)
    
    return compare_arrays(result_jax, result_torch.cpu().numpy(), rtol=RTOL, atol=ATOL, squeeze=True)


# =============================================================================
# Test 4: Pm Parameter Gradient Handling
# =============================================================================

def test_pm_gradient_simple():
    """Test gradient computation with respect to Pm parameters."""
    np.random.seed(SEED)
    N, M, D = 50, 40, 3
    
    x_np = np.random.randn(N, D).astype(np.float32)
    y_np = np.random.randn(M, D).astype(np.float32)
    
    x = jnp.array(x_np)
    y = jnp.array(y_np)
    sigma = jnp.array([0.5])
    
    formula = "Exp(-SqNorm2(x-y) * s)"
    aliases = [f"x=Vi({D})", f"y=Vj({D})", "s=Pm(1)"]
    
    op = Genred(formula, aliases, reduction_op='Sum', axis=1)
    
    def forward(s_var):
        return op(x, y, s_var).sum()
    
    # Compute gradient w.r.t. sigma
    grad_sigma = jax.grad(forward)(sigma)
    
    # Check that gradient is non-zero and finite
    grad_val = float(grad_sigma[0])
    passed = np.isfinite(grad_val) and abs(grad_val) > 1e-10
    
    return passed, abs(grad_val)


def test_pm_gradient_vs_pytorch():
    """Test Pm parameter gradient against PyTorch ground truth."""
    if not TORCH_AVAILABLE:
        return None
    
    np.random.seed(SEED)
    N, M, D = 50, 40, 3
    
    x_np = np.random.randn(N, D).astype(np.float32)
    y_np = np.random.randn(M, D).astype(np.float32)
    
    # JAX
    x_jax = jnp.array(x_np)
    y_jax = jnp.array(y_np)
    sigma_jax = jnp.array([0.5])
    
    formula = "Exp(-SqNorm2(x-y) * s)"
    aliases = [f"x=Vi({D})", f"y=Vj({D})", "s=Pm(1)"]
    
    op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1)
    grad_jax = jax.grad(lambda s: op_jax(x_jax, y_jax, s).sum())(sigma_jax)
    
    # PyTorch
    x_torch = torch.tensor(x_np, device='cuda')
    y_torch = torch.tensor(y_np, device='cuda')
    sigma_torch = torch.tensor([0.5], device='cuda', requires_grad=True)
    
    op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
    result_torch = op_torch(x_torch, y_torch, sigma_torch).sum()
    result_torch.backward()
    grad_torch = sigma_torch.grad
    
    return compare_arrays(grad_jax, grad_torch.cpu().numpy(), rtol=1e-3, atol=1e-4)


# =============================================================================
# Test 5: High-Dimensional Operations (Chunking)
# =============================================================================

def test_high_dim_gaussian():
    """Test high-dimensional Gaussian kernel (requires chunking)."""
    np.random.seed(SEED)
    N, M, D = 1000, 800, 128  # High dimension triggers chunking
    
    x_np = np.random.randn(N, D).astype(np.float32)
    y_np = np.random.randn(M, D).astype(np.float32)
    
    x = jnp.array(x_np)
    y = jnp.array(y_np)
    sigma = jnp.array([0.1])
    
    formula = "Exp(-SqNorm2(x-y) * s)"
    aliases = [f"x=Vi({D})", f"y=Vj({D})", "s=Pm(1)"]
    
    op = Genred(formula, aliases, reduction_op='Sum', axis=1)
    result = op(x, y, sigma)
    
    # Check output shape and finite values
    passed = result.shape == (N, 1) and np.all(np.isfinite(np.array(result)))
    
    return passed, 0.0


def test_high_dim_vs_pytorch():
    """Test high-dimensional operations against PyTorch."""
    if not TORCH_AVAILABLE:
        return None
    
    np.random.seed(SEED)
    N, M, D = 500, 400, 64
    
    x_np = np.random.randn(N, D).astype(np.float32)
    y_np = np.random.randn(M, D).astype(np.float32)
    
    formula = "SqNorm2(x-y)"
    aliases = [f"x=Vi({D})", f"y=Vj({D})"]
    
    # JAX
    x_jax = jnp.array(x_np)
    y_jax = jnp.array(y_np)
    op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1)
    result_jax = op_jax(x_jax, y_jax)
    
    # PyTorch
    x_torch = torch.tensor(x_np, device='cuda')
    y_torch = torch.tensor(y_np, device='cuda')
    op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
    result_torch = op_torch(x_torch, y_torch)
    
    return compare_arrays(result_jax, result_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)


# =============================================================================
# Test 6: Genred with Multiple Variable Types
# =============================================================================

def test_mixed_vi_vj_pm():
    """Test Genred with mixed Vi, Vj, and Pm variables."""
    if not TORCH_AVAILABLE:
        return None  # Skip without PyTorch ground truth
    
    np.random.seed(SEED)
    N, M, D = 50, 40, 3
    
    x_np = np.random.randn(N, D).astype(np.float32)
    y_np = np.random.randn(M, D).astype(np.float32)
    b_np = np.random.randn(M, D).astype(np.float32)
    sigma_np = np.array([0.5], dtype=np.float32)
    
    # Formula: exp(-|x-y|^2 * sigma) * b_j (weighted kernel sum)
    formula = "Exp(-SqNorm2(x-y) * s) * b"
    aliases = [f"x=Vi({D})", f"y=Vj({D})", f"b=Vj({D})", "s=Pm(1)"]
    
    # JAX
    x_jax = jnp.array(x_np)
    y_jax = jnp.array(y_np)
    b_jax = jnp.array(b_np)
    sigma_jax = jnp.array(sigma_np)
    
    op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1)
    result_jax = op_jax(x_jax, y_jax, b_jax, sigma_jax)
    
    # PyTorch ground truth
    x_torch = torch.tensor(x_np, device='cuda')
    y_torch = torch.tensor(y_np, device='cuda')
    b_torch = torch.tensor(b_np, device='cuda')
    sigma_torch = torch.tensor(sigma_np, device='cuda')
    
    from pykeops.torch import Genred as Genred_torch
    op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
    result_torch = op_torch(x_torch, y_torch, b_torch, sigma_torch)
    
    return compare_arrays(result_jax, result_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)


# =============================================================================
# Test 7: JIT Compilation
# =============================================================================

def test_jit_compilation():
    """Test that JIT compilation works correctly."""
    np.random.seed(SEED)
    N, M, D = 100, 80, 3
    
    x_np = np.random.randn(N, D).astype(np.float32)
    y_np = np.random.randn(M, D).astype(np.float32)
    
    x = jnp.array(x_np)
    y = jnp.array(y_np)
    
    formula = "SqNorm2(x-y)"
    aliases = [f"x=Vi({D})", f"y=Vj({D})"]
    op = Genred(formula, aliases, reduction_op='Sum', axis=1)
    
    @jax.jit
    def compute(x, y):
        return op(x, y)
    
    # First call (compilation)
    result1 = compute(x, y)
    
    # Second call (cached)
    result2 = compute(x, y)
    
    # Results should be identical
    return compare_arrays(result1, result2, rtol=0, atol=0)


def test_jit_gradient():
    """Test that JIT compilation works with gradients."""
    np.random.seed(SEED)
    N, M, D = 100, 80, 3
    
    x_np = np.random.randn(N, D).astype(np.float32)
    y_np = np.random.randn(M, D).astype(np.float32)
    
    x = jnp.array(x_np)
    y = jnp.array(y_np)
    
    formula = "SqNorm2(x-y)"
    aliases = [f"x=Vi({D})", f"y=Vj({D})"]
    op = Genred(formula, aliases, reduction_op='Sum', axis=1)
    
    @jax.jit
    def forward_and_grad(x, y):
        def fwd(x):
            return op(x, y).sum()
        return jax.grad(fwd)(x)
    
    # First call
    grad1 = forward_and_grad(x, y)
    
    # Second call
    grad2 = forward_and_grad(x, y)
    
    # Gradients should be identical
    return compare_arrays(grad1, grad2, rtol=0, atol=0)


# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    print_header("KeOps JAX Edge Case Tests", 
                "Testing edge cases discovered during development")
    
    print_environment_info()
    
    suite = TestSuite("Edge Case Tests", "Bugs that were found and fixed")
    
    # Test 1: Scalar Multiplication
    print_subheader("Scalar Multiplication (Variable Ordering)")
    run_test("Left scalar multiplication: 2.0 * (x - y)", test_scalar_multiplication, suite)
    if TORCH_AVAILABLE:
        run_test("Scalar multiplication vs PyTorch", test_scalar_multiplication_pytorch, suite)
    
    # Test 2: Batched vs Non-Batched
    print_subheader("Batched vs Non-Batched Operations")
    run_test("Non-batched (2D inputs)", test_non_batched_operations, suite)
    run_test("Batched (3D inputs)", test_batched_operations, suite)
    if TORCH_AVAILABLE:
        run_test("Batched vs PyTorch", test_batched_vs_pytorch, suite)
    
    # Test 3: Varifold Kernel
    print_subheader("Complex Varifold Kernel (6 Variables)")
    run_test("Varifold forward pass", test_varifold_forward, suite)
    run_test("Varifold gradient", test_varifold_gradient, suite)
    if TORCH_AVAILABLE:
        run_test("Varifold vs PyTorch", test_varifold_vs_pytorch, suite)
    
    # Test 4: Pm Parameter Gradients
    print_subheader("Pm Parameter Gradient Handling")
    run_test("Pm gradient (simple)", test_pm_gradient_simple, suite)
    if TORCH_AVAILABLE:
        run_test("Pm gradient vs PyTorch", test_pm_gradient_vs_pytorch, suite)
    
    # Test 5: High-Dimensional
    print_subheader("High-Dimensional Operations (Chunking)")
    run_test("High-dim Gaussian (D=128)", test_high_dim_gaussian, suite)
    if TORCH_AVAILABLE:
        run_test("High-dim vs PyTorch (D=64)", test_high_dim_vs_pytorch, suite)
    
    # Test 6: Mixed Variables
    print_subheader("Mixed Variable Types")
    run_test("Mixed Vi, Vj, Pm formula", test_mixed_vi_vj_pm, suite)
    
    # Test 7: JIT
    print_subheader("JIT Compilation")
    run_test("JIT forward pass", test_jit_compilation, suite)
    run_test("JIT gradient computation", test_jit_gradient, suite)
    
    # Print summary
    suite.print_summary()
    
    return 0 if suite.all_passed() else 1


if __name__ == "__main__":
    sys.exit(main())
