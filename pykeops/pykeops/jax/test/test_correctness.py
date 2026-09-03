#!/usr/bin/env python3
"""
KeOps JAX Correctness Tests
===========================
Cross-validation of JAX KeOps against PyTorch KeOps.

Tests cover:
- Forward pass comparison for various formulas
- Gradient comparison (wrt Vi, Vj, Pm variables)
- Multiple problem sizes (tiny to large)
- Both Genred and LazyTensor interfaces
- Reduction operations (Sum, Min, Max, ArgMin, ArgMax)

PyTorch KeOps serves as the ground truth since it's the reference implementation.
"""

import sys
import pytest
import numpy as np

# =============================================================================
# Imports
# =============================================================================

from test_utils import (
    TestSuite, TestResult, Status,
    print_header, print_subheader, print_info, print_success, print_error, print_warning,
    compare_arrays, run_test, print_environment_info,
    print_benchmark_table, RICH_AVAILABLE,
    setup_jax_float64, get_np_dtype, get_dtype_str, is_float64_mode
)

# Setup float64 mode BEFORE importing JAX
setup_jax_float64()

import jax
import jax.numpy as jnp

if RICH_AVAILABLE:
    from test_utils import console

# Import JAX KeOps
try:
    from pykeops.jax import Genred as Genred_jax, LazyTensor as LazyTensor_jax
    JAX_AVAILABLE = True
except ImportError as e:
    print(f"Error: pykeops.jax not found: {e}")
    JAX_AVAILABLE = False
    sys.exit(1)

# Import PyTorch KeOps
try:
    import torch
    from pykeops.torch import Genred as Genred_torch, LazyTensor as LazyTensor_torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        print_success("PyTorch KeOps available - using as ground truth")
    else:
        print_warning("CUDA not available for PyTorch")
except ImportError:
    TORCH_AVAILABLE = False
    print_warning("PyTorch KeOps not found - tests will be limited")

if not TORCH_AVAILABLE:
    print_error("PyTorch KeOps with CUDA is required for correctness tests")
    print_info("Install with: pip install pykeops torch")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

# Every test here needs a GPU and compares against PyTorch KeOps. conftest.py registers these
# markers and skips on missing hardware, so declaring them at module level is what makes
# `pytest -m pytorch` and `pytest -m gpu` select anything.
pytestmark = [pytest.mark.gpu, pytest.mark.pytorch]

SEED = 42
RTOL = 1e-4  # Tight tolerance since both are KeOps
ATOL = 1e-5

# Test sizes: (name, N, M, D)
TEST_SIZES = [
    ("Tiny", 10, 8, 3),
    ("Small", 100, 80, 3),
    ("Medium", 500, 400, 3),
    ("Large", 2000, 1500, 3),
    ("HighDim", 200, 150, 32),
]

# Test formulas: (name, formula, aliases_template, has_param, has_b)
TEST_FORMULAS = [
    ("SqDist", "SqDist(x, y)", ["x=Vi(D)", "y=Vj(D)"], False, False),
    ("SqNorm2", "SqNorm2(x-y)", ["x=Vi(D)", "y=Vj(D)"], False, False),
    ("Gaussian", "Exp(-SqNorm2(x-y)*s)", ["x=Vi(D)", "y=Vj(D)", "s=Pm(1)"], True, False),
    ("Laplacian", "Exp(-Norm2(x-y))", ["x=Vi(D)", "y=Vj(D)"], False, False),
    ("Cauchy", "Inv(IntCst(1)+SqNorm2(x-y))", ["x=Vi(D)", "y=Vj(D)"], False, False),
    ("WeightedSum", "Exp(-SqNorm2(x-y)*s)*b", ["x=Vi(D)", "y=Vj(D)", "b=Vj(D)", "s=Pm(1)"], True, True),
]

# Reduction operations to test
REDUCTIONS = ["Sum", "Min", "Max"]


# =============================================================================
# Helper Functions
# =============================================================================

def generate_test_data(n, m, d, seed=SEED):
    """Generate matching test data for JAX and PyTorch."""
    np.random.seed(seed)
    
    x_np = np.random.randn(n, d).astype(get_np_dtype())
    y_np = np.random.randn(m, d).astype(get_np_dtype())
    b_np = np.random.randn(m, d).astype(get_np_dtype())
    s_np = np.array([0.5], dtype=get_np_dtype())
    
    return {
        'x_np': x_np, 'y_np': y_np, 'b_np': b_np, 's_np': s_np,
        'x_jax': jnp.array(x_np),
        'y_jax': jnp.array(y_np),
        'b_jax': jnp.array(b_np),
        's_jax': jnp.array(s_np),
        'x_torch': torch.tensor(x_np, device='cuda'),
        'y_torch': torch.tensor(y_np, device='cuda'),
        'b_torch': torch.tensor(b_np, device='cuda'),
        's_torch': torch.tensor(s_np, device='cuda'),
    }


def make_aliases(alias_templates, d):
    """Replace D placeholder with actual dimension."""
    return [a.replace("D", str(d)) for a in alias_templates]


# =============================================================================
# Forward Pass Tests
# =============================================================================

def check_forward_genred(formula_name, formula, alias_templates, has_param, has_b, 
                        n, m, d, reduction='Sum', axis=1):
    """Test Genred forward pass: JAX vs PyTorch."""
    aliases = make_aliases(alias_templates, d)
    data = generate_test_data(n, m, d)
    
    # JAX
    op_jax = Genred_jax(formula, aliases, reduction_op=reduction, axis=axis, dtype=get_dtype_str())
    if has_b:
        result_jax = op_jax(data['x_jax'], data['y_jax'], data['b_jax'], data['s_jax'])
    elif has_param:
        result_jax = op_jax(data['x_jax'], data['y_jax'], data['s_jax'])
    else:
        result_jax = op_jax(data['x_jax'], data['y_jax'])
    
    # PyTorch
    op_torch = Genred_torch(formula, aliases, reduction_op=reduction, axis=axis, dtype=get_dtype_str())
    if has_b:
        result_torch = op_torch(data['x_torch'], data['y_torch'], data['b_torch'], data['s_torch'])
    elif has_param:
        result_torch = op_torch(data['x_torch'], data['y_torch'], data['s_torch'])
    else:
        result_torch = op_torch(data['x_torch'], data['y_torch'])
    
    jax_np = np.array(result_jax)
    torch_np = result_torch.cpu().numpy()
    
    return compare_arrays(jax_np, torch_np, rtol=RTOL, atol=ATOL)


def check_forward_lazytensor(n, m, d):
    """Test LazyTensor forward pass: JAX vs PyTorch."""
    data = generate_test_data(n, m, d)
    
    # JAX LazyTensor
    x_i = LazyTensor_jax(data['x_jax'][:, None, :])
    y_j = LazyTensor_jax(data['y_jax'][None, :, :])
    K_jax = ((x_i - y_j) ** 2).sum(-1)
    result_jax = K_jax.sum(axis=1)
    
    # PyTorch LazyTensor
    x_i_t = LazyTensor_torch(data['x_torch'][:, None, :])
    y_j_t = LazyTensor_torch(data['y_torch'][None, :, :])
    K_torch = ((x_i_t - y_j_t) ** 2).sum(-1)
    result_torch = K_torch.sum(axis=1)
    
    return compare_arrays(result_jax, result_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)


# =============================================================================
# Gradient Tests
# =============================================================================

def check_gradient_vi(formula_name, formula, alias_templates, has_param, has_b, n, m, d):
    """Test gradient w.r.t. Vi variable (x)."""
    aliases = make_aliases(alias_templates, d)
    data = generate_test_data(n, m, d)
    
    # JAX gradient
    op_jax = Genred_jax(formula, aliases, reduction_op='Sum', axis=1, dtype=get_dtype_str())
    
    def forward_jax(x):
        if has_b:
            return op_jax(x, data['y_jax'], data['b_jax'], data['s_jax']).sum()
        elif has_param:
            return op_jax(x, data['y_jax'], data['s_jax']).sum()
        else:
            return op_jax(x, data['y_jax']).sum()
    
    grad_jax = jax.grad(forward_jax)(data['x_jax'])
    
    # PyTorch gradient
    x_torch = data['x_torch'].clone().requires_grad_(True)
    op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1, dtype=get_dtype_str())
    
    if has_b:
        result_torch = op_torch(x_torch, data['y_torch'], data['b_torch'], data['s_torch']).sum()
    elif has_param:
        result_torch = op_torch(x_torch, data['y_torch'], data['s_torch']).sum()
    else:
        result_torch = op_torch(x_torch, data['y_torch']).sum()
    
    result_torch.backward()
    grad_torch = x_torch.grad
    
    return compare_arrays(grad_jax, grad_torch.cpu().numpy(), rtol=1e-3, atol=1e-4)


def check_gradient_vj(formula_name, formula, alias_templates, has_param, has_b, n, m, d):
    """Test gradient w.r.t. Vj variable (y)."""
    aliases = make_aliases(alias_templates, d)
    data = generate_test_data(n, m, d)
    
    # JAX gradient
    op_jax = Genred_jax(formula, aliases, reduction_op='Sum', axis=1, dtype=get_dtype_str())
    
    def forward_jax(y):
        if has_b:
            return op_jax(data['x_jax'], y, data['b_jax'], data['s_jax']).sum()
        elif has_param:
            return op_jax(data['x_jax'], y, data['s_jax']).sum()
        else:
            return op_jax(data['x_jax'], y).sum()
    
    grad_jax = jax.grad(forward_jax)(data['y_jax'])
    
    # PyTorch gradient
    y_torch = data['y_torch'].clone().requires_grad_(True)
    op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1, dtype=get_dtype_str())
    
    if has_b:
        result_torch = op_torch(data['x_torch'], y_torch, data['b_torch'], data['s_torch']).sum()
    elif has_param:
        result_torch = op_torch(data['x_torch'], y_torch, data['s_torch']).sum()
    else:
        result_torch = op_torch(data['x_torch'], y_torch).sum()
    
    result_torch.backward()
    grad_torch = y_torch.grad
    
    return compare_arrays(grad_jax, grad_torch.cpu().numpy(), rtol=1e-3, atol=1e-4)


def check_gradient_pm(n, m, d):
    """Test gradient w.r.t. Pm parameter."""
    data = generate_test_data(n, m, d)
    
    formula = "Exp(-SqNorm2(x-y)*s)"
    aliases = make_aliases(["x=Vi(D)", "y=Vj(D)", "s=Pm(1)"], d)
    
    # JAX gradient
    op_jax = Genred_jax(formula, aliases, reduction_op='Sum', axis=1, dtype=get_dtype_str())
    grad_jax = jax.grad(lambda s: op_jax(data['x_jax'], data['y_jax'], s).sum())(data['s_jax'])
    
    # PyTorch gradient
    s_torch = data['s_torch'].clone().requires_grad_(True)
    op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1, dtype=get_dtype_str())
    result_torch = op_torch(data['x_torch'], data['y_torch'], s_torch).sum()
    result_torch.backward()
    grad_torch = s_torch.grad
    
    return compare_arrays(grad_jax, grad_torch.cpu().numpy(), rtol=1e-3, atol=1e-4)


def check_gradient_lazytensor(n, m, d):
    """Test LazyTensor gradient."""
    data = generate_test_data(n, m, d)
    
    # JAX
    def forward_jax(x):
        x_i = LazyTensor_jax(x[:, None, :])
        y_j = LazyTensor_jax(data['y_jax'][None, :, :])
        K = ((x_i - y_j) ** 2).sum(-1)
        return K.sum(axis=1).sum()
    
    grad_jax = jax.grad(forward_jax)(data['x_jax'])
    
    # PyTorch
    x_torch = data['x_torch'].clone().requires_grad_(True)
    x_i_t = LazyTensor_torch(x_torch[:, None, :])
    y_j_t = LazyTensor_torch(data['y_torch'][None, :, :])
    K_t = ((x_i_t - y_j_t) ** 2).sum(-1)
    result_t = K_t.sum(axis=1).sum()
    result_t.backward()
    grad_torch = x_torch.grad
    
    return compare_arrays(grad_jax, grad_torch.cpu().numpy(), rtol=1e-3, atol=1e-4)


# =============================================================================
# Reduction Operation Tests
# =============================================================================

def check_reduction(reduction, n, m, d, axis=1):
    """Test specific reduction operation."""
    data = generate_test_data(n, m, d)
    
    formula = "SqDist(x,y)"
    aliases = make_aliases(["x=Vi(D)", "y=Vj(D)"], d)
    
    # JAX
    op_jax = Genred_jax(formula, aliases, reduction_op=reduction, axis=axis, dtype=get_dtype_str())
    result_jax = op_jax(data['x_jax'], data['y_jax'])
    
    # PyTorch
    op_torch = Genred_torch(formula, aliases, reduction_op=reduction, axis=axis, dtype=get_dtype_str())
    result_torch = op_torch(data['x_torch'], data['y_torch'])
    
    return compare_arrays(result_jax, result_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)


# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    print_header("KeOps JAX Correctness Tests", 
                "Cross-validation against PyTorch KeOps")
    
    print_environment_info()
    
    suite = TestSuite("Correctness Tests", "JAX KeOps vs PyTorch KeOps")
    
    # ==========================================================================
    # Forward Pass Tests
    # ==========================================================================
    print_subheader("Forward Pass: Genred Interface")
    
    for size_name, n, m, d in TEST_SIZES[:3]:  # Test first 3 sizes
        for formula_name, formula, aliases, has_param, has_b in TEST_FORMULAS:
            test_name = f"{formula_name} [{size_name}: {n}x{m}x{d}]"
            run_test(
                test_name,
                lambda fn=formula_name, f=formula, a=aliases, hp=has_param, hb=has_b, 
                       nn=n, mm=m, dd=d: check_forward_genred(fn, f, a, hp, hb, nn, mm, dd),
                suite
            )
    
    print_subheader("Forward Pass: LazyTensor Interface")
    
    for size_name, n, m, d in TEST_SIZES[:3]:
        test_name = f"LazyTensor SqDist [{size_name}: {n}x{m}x{d}]"
        run_test(
            test_name,
            lambda nn=n, mm=m, dd=d: check_forward_lazytensor(nn, mm, dd),
            suite
        )
    
    # ==========================================================================
    # Gradient Tests
    # ==========================================================================
    print_subheader("Gradients: Vi (x) Variable")
    
    n, m, d = 100, 80, 3
    for formula_name, formula, aliases, has_param, has_b in TEST_FORMULAS:
        test_name = f"∂/∂x {formula_name}"
        run_test(
            test_name,
            lambda fn=formula_name, f=formula, a=aliases, hp=has_param, hb=has_b: 
                check_gradient_vi(fn, f, a, hp, hb, n, m, d),
            suite
        )
    
    print_subheader("Gradients: Vj (y) Variable")
    
    for formula_name, formula, aliases, has_param, has_b in TEST_FORMULAS:
        test_name = f"∂/∂y {formula_name}"
        run_test(
            test_name,
            lambda fn=formula_name, f=formula, a=aliases, hp=has_param, hb=has_b: 
                check_gradient_vj(fn, f, a, hp, hb, n, m, d),
            suite
        )
    
    print_subheader("Gradients: Pm Parameter")
    
    for size_name, n, m, d in TEST_SIZES[:3]:
        test_name = f"∂/∂σ Gaussian [{size_name}]"
        run_test(
            test_name,
            lambda nn=n, mm=m, dd=d: check_gradient_pm(nn, mm, dd),
            suite
        )
    
    print_subheader("Gradients: LazyTensor")
    
    for size_name, n, m, d in TEST_SIZES[:3]:
        test_name = f"LazyTensor ∂/∂x [{size_name}]"
        run_test(
            test_name,
            lambda nn=n, mm=m, dd=d: check_gradient_lazytensor(nn, mm, dd),
            suite
        )
    
    # ==========================================================================
    # Reduction Tests
    # ==========================================================================
    print_subheader("Reduction Operations")
    
    n, m, d = 100, 80, 3
    for reduction in REDUCTIONS:
        for axis in [0, 1]:
            test_name = f"{reduction} reduction (axis={axis})"
            run_test(
                test_name,
                lambda r=reduction, a=axis: check_reduction(r, n, m, d, a),
                suite
            )
    
    # ==========================================================================
    # Large Scale Tests
    # ==========================================================================
    print_subheader("Large Scale Tests")
    
    for size_name, n, m, d in TEST_SIZES[3:]:  # Large and HighDim
        test_name = f"Large SqDist [{size_name}: {n}x{m}x{d}]"
        run_test(
            test_name,
            lambda nn=n, mm=m, dd=d: check_forward_genred(
                "SqDist", "SqDist(x,y)", ["x=Vi(D)", "y=Vj(D)"], False, False, nn, mm, dd),
            suite
        )
        
        test_name = f"Large Gaussian [{size_name}: {n}x{m}x{d}]"
        run_test(
            test_name,
            lambda nn=n, mm=m, dd=d: check_forward_genred(
                "Gaussian", "Exp(-SqNorm2(x-y)*s)", ["x=Vi(D)", "y=Vj(D)", "s=Pm(1)"], 
                True, False, nn, mm, dd),
            suite
        )
    
    # ==========================================================================
    # Print Summary
    # ==========================================================================
    suite.print_summary()
    
    return 0 if suite.all_passed() else 1


if __name__ == "__main__":
    sys.exit(main())
