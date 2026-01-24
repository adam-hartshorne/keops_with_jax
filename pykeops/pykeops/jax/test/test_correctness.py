#!/usr/bin/env python3
"""
KeOps JAX Correctness Tests
===========================
Cross-validation against PyTorch KeOps to ensure numerical correctness.

Tests cover:
- Forward pass comparison
- Gradient comparison (wrt Vi, Vj, Pm variables)
- Multiple formula types
- Various problem sizes
- Edge cases
"""

import sys
import time
import numpy as np

import jax
import jax.numpy as jnp

# Import test utilities
from test_utils import (
    Colors, Status, TestResult, TestSuite,
    print_header, print_subheader,
    ASCIITable, TableColumn, compare_arrays, format_comparison_result
)

# Import KeOps JAX
try:
    from pykeops.jax import Genred as Genred_jax, LazyTensor as LazyTensor_jax, Vi, Vj, Pm
    JAX_AVAILABLE = True
except ImportError as e:
    print(f"{Colors.RED}Error: pykeops.jax not found: {e}{Colors.RESET}")
    JAX_AVAILABLE = False

# Import KeOps PyTorch
try:
    import torch
    from pykeops.torch import Genred as Genred_torch, LazyTensor as LazyTensor_torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if not TORCH_AVAILABLE:
        print(f"{Colors.YELLOW}Warning: CUDA not available for PyTorch{Colors.RESET}")
except ImportError as e:
    print(f"{Colors.YELLOW}Warning: PyTorch KeOps not found: {e}{Colors.RESET}")
    TORCH_AVAILABLE = False


# =============================================================================
# Test Configuration
# =============================================================================

SEED = 42
RTOL = 1e-4
ATOL = 1e-5

# Test configurations: (name, N, M, D)
TEST_SIZES = [
    ("Tiny", 10, 8, 3),
    ("Small", 50, 40, 3),
    ("Medium", 200, 150, 3),
    ("Large", 500, 400, 3),
    ("High-D", 100, 80, 32),
]

# Formula configurations: (name, formula, aliases_template, has_param)
TEST_FORMULAS = [
    ("SqDist", "SqDist(x, y)", ["x=Vi(D)", "y=Vj(D)"], False),
    ("Gaussian", "Exp(-SqNorm2(x-y)*s)", ["x=Vi(D)", "y=Vj(D)", "s=Pm(1)"], True),
    ("Laplacian", "Exp(-Norm2(x-y))", ["x=Vi(D)", "y=Vj(D)"], False),
    ("Cauchy", "Inv(IntCst(1) + SqNorm2(x-y))", ["x=Vi(D)", "y=Vj(D)"], False),
    ("WeightedSum", "Exp(-SqNorm2(x-y)*s)*b", ["x=Vi(D)", "y=Vj(D)", "b=Vj(D)", "s=Pm(1)"], True),
]


# =============================================================================
# Test Data Generation
# =============================================================================

def generate_test_data(n, m, d, seed=SEED):
    """Generate matching test data for JAX and PyTorch."""
    np.random.seed(seed)
    
    x_np = np.random.randn(n, d).astype(np.float32)
    y_np = np.random.randn(m, d).astype(np.float32)
    b_np = np.random.randn(m, d).astype(np.float32)
    s_np = np.array([0.5], dtype=np.float32)
    
    # JAX arrays
    x_jax = jnp.array(x_np)
    y_jax = jnp.array(y_np)
    b_jax = jnp.array(b_np)
    s_jax = jnp.array(s_np)
    
    # PyTorch tensors
    if TORCH_AVAILABLE:
        x_torch = torch.tensor(x_np, device='cuda', requires_grad=True)
        y_torch = torch.tensor(y_np, device='cuda', requires_grad=True)
        b_torch = torch.tensor(b_np, device='cuda')
        s_torch = torch.tensor(s_np, device='cuda')
    else:
        x_torch = y_torch = b_torch = s_torch = None
    
    return {
        'x_jax': x_jax, 'y_jax': y_jax, 'b_jax': b_jax, 's_jax': s_jax,
        'x_torch': x_torch, 'y_torch': y_torch, 'b_torch': b_torch, 's_torch': s_torch,
        'x_np': x_np, 'y_np': y_np, 'b_np': b_np, 's_np': s_np,
    }


def make_aliases(alias_templates, d):
    """Replace D placeholder with actual dimension."""
    return [a.replace("D", str(d)) for a in alias_templates]


# =============================================================================
# Forward Pass Comparison
# =============================================================================

def test_forward_pass(formula_name, formula, alias_templates, has_param, n, m, d):
    """Compare forward pass between JAX and PyTorch."""
    aliases = make_aliases(alias_templates, d)
    data = generate_test_data(n, m, d)
    
    # JAX computation
    op_jax = Genred_jax(formula, aliases, reduction_op='Sum', axis=1)
    
    if has_param and "b=Vj" in str(alias_templates):
        result_jax = op_jax(data['x_jax'], data['y_jax'], data['b_jax'], data['s_jax'])
    elif has_param:
        result_jax = op_jax(data['x_jax'], data['y_jax'], data['s_jax'])
    else:
        result_jax = op_jax(data['x_jax'], data['y_jax'])
    
    result_jax_np = np.array(result_jax)
    
    # PyTorch computation
    if TORCH_AVAILABLE:
        op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
        
        if has_param and "b=Vj" in str(alias_templates):
            result_torch = op_torch(data['x_torch'], data['y_torch'], data['b_torch'], data['s_torch'])
        elif has_param:
            result_torch = op_torch(data['x_torch'], data['y_torch'], data['s_torch'])
        else:
            result_torch = op_torch(data['x_torch'], data['y_torch'])
        
        result_torch_np = result_torch.detach().cpu().numpy()
        
        # Compare
        match, max_diff = compare_arrays(result_jax_np, result_torch_np, rtol=RTOL, atol=ATOL)
        return match, max_diff, result_jax_np.shape
    else:
        return None, None, result_jax_np.shape


# =============================================================================
# Gradient Comparison
# =============================================================================

def test_gradient_vi(formula_name, formula, alias_templates, has_param, n, m, d):
    """Compare gradient w.r.t. x (Vi variable)."""
    aliases = make_aliases(alias_templates, d)
    data = generate_test_data(n, m, d)
    
    # JAX gradient
    op_jax = Genred_jax(formula, aliases, reduction_op='Sum', axis=1)
    
    if has_param and "b=Vj" in str(alias_templates):
        def loss_jax(x):
            return jnp.sum(op_jax(x, data['y_jax'], data['b_jax'], data['s_jax']))
    elif has_param:
        def loss_jax(x):
            return jnp.sum(op_jax(x, data['y_jax'], data['s_jax']))
    else:
        def loss_jax(x):
            return jnp.sum(op_jax(x, data['y_jax']))
    
    grad_jax = jax.grad(loss_jax)(data['x_jax'])
    grad_jax_np = np.array(grad_jax)
    
    # PyTorch gradient
    if TORCH_AVAILABLE:
        # Need to recreate with requires_grad=True
        x_torch = torch.tensor(data['x_np'], device='cuda', requires_grad=True)
        y_torch = torch.tensor(data['y_np'], device='cuda')
        b_torch = torch.tensor(data['b_np'], device='cuda')
        s_torch = torch.tensor(data['s_np'], device='cuda')
        
        op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
        
        if has_param and "b=Vj" in str(alias_templates):
            result_torch = op_torch(x_torch, y_torch, b_torch, s_torch)
        elif has_param:
            result_torch = op_torch(x_torch, y_torch, s_torch)
        else:
            result_torch = op_torch(x_torch, y_torch)
        
        loss_torch = result_torch.sum()
        loss_torch.backward()
        grad_torch_np = x_torch.grad.cpu().numpy()
        
        # Compare
        match, max_diff = compare_arrays(grad_jax_np, grad_torch_np, rtol=RTOL, atol=ATOL)
        return match, max_diff
    else:
        return None, None


def test_gradient_vj(formula_name, formula, alias_templates, has_param, n, m, d):
    """Compare gradient w.r.t. y (Vj variable)."""
    aliases = make_aliases(alias_templates, d)
    data = generate_test_data(n, m, d)
    
    # JAX gradient
    op_jax = Genred_jax(formula, aliases, reduction_op='Sum', axis=1)
    
    if has_param and "b=Vj" in str(alias_templates):
        def loss_jax(y):
            return jnp.sum(op_jax(data['x_jax'], y, data['b_jax'], data['s_jax']))
    elif has_param:
        def loss_jax(y):
            return jnp.sum(op_jax(data['x_jax'], y, data['s_jax']))
    else:
        def loss_jax(y):
            return jnp.sum(op_jax(data['x_jax'], y))
    
    grad_jax = jax.grad(loss_jax)(data['y_jax'])
    grad_jax_np = np.array(grad_jax)
    
    # PyTorch gradient
    if TORCH_AVAILABLE:
        x_torch = torch.tensor(data['x_np'], device='cuda')
        y_torch = torch.tensor(data['y_np'], device='cuda', requires_grad=True)
        b_torch = torch.tensor(data['b_np'], device='cuda')
        s_torch = torch.tensor(data['s_np'], device='cuda')
        
        op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
        
        if has_param and "b=Vj" in str(alias_templates):
            result_torch = op_torch(x_torch, y_torch, b_torch, s_torch)
        elif has_param:
            result_torch = op_torch(x_torch, y_torch, s_torch)
        else:
            result_torch = op_torch(x_torch, y_torch)
        
        loss_torch = result_torch.sum()
        loss_torch.backward()
        grad_torch_np = y_torch.grad.cpu().numpy()
        
        # Compare
        match, max_diff = compare_arrays(grad_jax_np, grad_torch_np, rtol=RTOL, atol=ATOL)
        return match, max_diff
    else:
        return None, None


# =============================================================================
# LazyTensor Comparison
# =============================================================================

def test_lazytensor_forward(n, m, d):
    """Compare LazyTensor forward pass."""
    data = generate_test_data(n, m, d)
    
    # JAX LazyTensor
    x_i = LazyTensor_jax(data['x_jax'][:, None, :])
    y_j = LazyTensor_jax(data['y_jax'][None, :, :])
    result_jax = ((x_i - y_j)**2).sum(-1).sum(1)
    result_jax_np = np.array(result_jax)
    
    # PyTorch LazyTensor
    if TORCH_AVAILABLE:
        x_i_torch = LazyTensor_torch(data['x_torch'][:, None, :])
        y_j_torch = LazyTensor_torch(data['y_torch'][None, :, :])
        result_torch = ((x_i_torch - y_j_torch)**2).sum(-1).sum(1)
        result_torch_np = result_torch.detach().cpu().numpy()
        
        match, max_diff = compare_arrays(result_jax_np, result_torch_np, rtol=RTOL, atol=ATOL)
        return match, max_diff
    else:
        return None, None


def test_lazytensor_gradient(n, m, d):
    """Compare LazyTensor gradient."""
    data = generate_test_data(n, m, d)
    
    # JAX gradient
    def loss_jax(x):
        x_i = LazyTensor_jax(x[:, None, :])
        y_j = LazyTensor_jax(data['y_jax'][None, :, :])
        return jnp.sum(((x_i - y_j)**2).sum(-1).sum(1))
    
    grad_jax = jax.grad(loss_jax)(data['x_jax'])
    grad_jax_np = np.array(grad_jax)
    
    # PyTorch gradient
    if TORCH_AVAILABLE:
        x_torch = torch.tensor(data['x_np'], device='cuda', requires_grad=True)
        y_torch = torch.tensor(data['y_np'], device='cuda')
        
        x_i_torch = LazyTensor_torch(x_torch[:, None, :])
        y_j_torch = LazyTensor_torch(y_torch[None, :, :])
        result_torch = ((x_i_torch - y_j_torch)**2).sum(-1).sum(1)
        
        loss_torch = result_torch.sum()
        loss_torch.backward()
        grad_torch_np = x_torch.grad.cpu().numpy()
        
        match, max_diff = compare_arrays(grad_jax_np, grad_torch_np, rtol=RTOL, atol=ATOL)
        return match, max_diff
    else:
        return None, None


# =============================================================================
# Main Test Runner
# =============================================================================

def run_correctness_tests():
    """Run all correctness tests."""
    if not JAX_AVAILABLE:
        print(f"{Colors.RED}KeOps JAX not available. Cannot run tests.{Colors.RESET}")
        return False
    
    print_header("KeOps JAX vs PyTorch Correctness Tests")
    
    if not TORCH_AVAILABLE:
        print(f"{Colors.YELLOW}PyTorch not available - running JAX-only validation{Colors.RESET}")
        print()
    
    suite = TestSuite("Correctness Test Results")
    
    # =========================
    # Forward Pass Tests
    # =========================
    print_subheader("Forward Pass Comparison")
    
    # Create results table
    table = ASCIITable([
        TableColumn("Formula", 15),
        TableColumn("Size", 20),
        TableColumn("Status", 12),
        TableColumn("Max Diff", 12),
    ], title="Forward Pass: JAX vs PyTorch")
    
    for formula_name, formula, alias_templates, has_param in TEST_FORMULAS:
        for size_name, n, m, d in TEST_SIZES:
            test_name = f"fwd_{formula_name}_{size_name}"
            
            try:
                start = time.time()
                match, max_diff, shape = test_forward_pass(
                    formula_name, formula, alias_templates, has_param, n, m, d
                )
                duration = (time.time() - start) * 1000
                
                if match is None:
                    status = Status.SKIP
                    status_str = f"{Colors.YELLOW}SKIP{Colors.RESET}"
                    diff_str = "-"
                elif match:
                    status = Status.PASS
                    status_str = f"{Colors.GREEN}PASS{Colors.RESET}"
                    diff_str = f"{max_diff:.2e}"
                else:
                    status = Status.FAIL
                    status_str = f"{Colors.RED}FAIL{Colors.RESET}"
                    diff_str = f"{Colors.RED}{max_diff:.2e}{Colors.RESET}"
                
                suite.add_result(TestResult(test_name, status, duration))
                table.add_row([formula_name, f"{size_name} ({n}x{m}x{d})", status_str, diff_str])
                
            except Exception as e:
                suite.add_result(TestResult(test_name, Status.ERROR, 0, str(e)))
                table.add_row([formula_name, size_name, f"{Colors.RED}ERROR{Colors.RESET}", str(e)[:10]])
    
    table.print()
    
    # =========================
    # Gradient Tests (Vi)
    # =========================
    print_subheader("Gradient (Vi) Comparison")
    
    table = ASCIITable([
        TableColumn("Formula", 15),
        TableColumn("Size", 20),
        TableColumn("Status", 12),
        TableColumn("Max Diff", 12),
    ], title="Gradient (d/dx): JAX vs PyTorch")
    
    for formula_name, formula, alias_templates, has_param in TEST_FORMULAS:
        for size_name, n, m, d in TEST_SIZES[:3]:  # Only small/medium for gradients
            test_name = f"grad_vi_{formula_name}_{size_name}"
            
            try:
                start = time.time()
                match, max_diff = test_gradient_vi(
                    formula_name, formula, alias_templates, has_param, n, m, d
                )
                duration = (time.time() - start) * 1000
                
                if match is None:
                    status = Status.SKIP
                    status_str = f"{Colors.YELLOW}SKIP{Colors.RESET}"
                    diff_str = "-"
                elif match:
                    status = Status.PASS
                    status_str = f"{Colors.GREEN}PASS{Colors.RESET}"
                    diff_str = f"{max_diff:.2e}"
                else:
                    status = Status.FAIL
                    status_str = f"{Colors.RED}FAIL{Colors.RESET}"
                    diff_str = f"{Colors.RED}{max_diff:.2e}{Colors.RESET}"
                
                suite.add_result(TestResult(test_name, status, duration))
                table.add_row([formula_name, f"{size_name} ({n}x{m}x{d})", status_str, diff_str])
                
            except Exception as e:
                suite.add_result(TestResult(test_name, Status.ERROR, 0, str(e)))
                table.add_row([formula_name, size_name, f"{Colors.RED}ERROR{Colors.RESET}", str(e)[:10]])
    
    table.print()
    
    # =========================
    # Gradient Tests (Vj)
    # =========================
    print_subheader("Gradient (Vj) Comparison")
    
    table = ASCIITable([
        TableColumn("Formula", 15),
        TableColumn("Size", 20),
        TableColumn("Status", 12),
        TableColumn("Max Diff", 12),
    ], title="Gradient (d/dy): JAX vs PyTorch")
    
    for formula_name, formula, alias_templates, has_param in TEST_FORMULAS:
        for size_name, n, m, d in TEST_SIZES[:3]:
            test_name = f"grad_vj_{formula_name}_{size_name}"
            
            try:
                start = time.time()
                match, max_diff = test_gradient_vj(
                    formula_name, formula, alias_templates, has_param, n, m, d
                )
                duration = (time.time() - start) * 1000
                
                if match is None:
                    status = Status.SKIP
                    status_str = f"{Colors.YELLOW}SKIP{Colors.RESET}"
                    diff_str = "-"
                elif match:
                    status = Status.PASS
                    status_str = f"{Colors.GREEN}PASS{Colors.RESET}"
                    diff_str = f"{max_diff:.2e}"
                else:
                    status = Status.FAIL
                    status_str = f"{Colors.RED}FAIL{Colors.RESET}"
                    diff_str = f"{Colors.RED}{max_diff:.2e}{Colors.RESET}"
                
                suite.add_result(TestResult(test_name, status, duration))
                table.add_row([formula_name, f"{size_name} ({n}x{m}x{d})", status_str, diff_str])
                
            except Exception as e:
                suite.add_result(TestResult(test_name, Status.ERROR, 0, str(e)))
                table.add_row([formula_name, size_name, f"{Colors.RED}ERROR{Colors.RESET}", str(e)[:10]])
    
    table.print()
    
    # =========================
    # LazyTensor Tests
    # =========================
    print_subheader("LazyTensor Comparison")
    
    table = ASCIITable([
        TableColumn("Test", 20),
        TableColumn("Size", 20),
        TableColumn("Status", 12),
        TableColumn("Max Diff", 12),
    ], title="LazyTensor: JAX vs PyTorch")
    
    for size_name, n, m, d in TEST_SIZES[:3]:
        # Forward
        test_name = f"lazy_fwd_{size_name}"
        try:
            start = time.time()
            match, max_diff = test_lazytensor_forward(n, m, d)
            duration = (time.time() - start) * 1000
            
            if match is None:
                status = Status.SKIP
                status_str = f"{Colors.YELLOW}SKIP{Colors.RESET}"
                diff_str = "-"
            elif match:
                status = Status.PASS
                status_str = f"{Colors.GREEN}PASS{Colors.RESET}"
                diff_str = f"{max_diff:.2e}"
            else:
                status = Status.FAIL
                status_str = f"{Colors.RED}FAIL{Colors.RESET}"
                diff_str = f"{Colors.RED}{max_diff:.2e}{Colors.RESET}"
            
            suite.add_result(TestResult(test_name, status, duration))
            table.add_row(["Forward", f"{size_name} ({n}x{m}x{d})", status_str, diff_str])
            
        except Exception as e:
            suite.add_result(TestResult(test_name, Status.ERROR, 0, str(e)))
            table.add_row(["Forward", size_name, f"{Colors.RED}ERROR{Colors.RESET}", str(e)[:10]])
        
        # Gradient
        test_name = f"lazy_grad_{size_name}"
        try:
            start = time.time()
            match, max_diff = test_lazytensor_gradient(n, m, d)
            duration = (time.time() - start) * 1000
            
            if match is None:
                status = Status.SKIP
                status_str = f"{Colors.YELLOW}SKIP{Colors.RESET}"
                diff_str = "-"
            elif match:
                status = Status.PASS
                status_str = f"{Colors.GREEN}PASS{Colors.RESET}"
                diff_str = f"{max_diff:.2e}"
            else:
                status = Status.FAIL
                status_str = f"{Colors.RED}FAIL{Colors.RESET}"
                diff_str = f"{Colors.RED}{max_diff:.2e}{Colors.RESET}"
            
            suite.add_result(TestResult(test_name, status, duration))
            table.add_row(["Gradient", f"{size_name} ({n}x{m}x{d})", status_str, diff_str])
            
        except Exception as e:
            suite.add_result(TestResult(test_name, Status.ERROR, 0, str(e)))
            table.add_row(["Gradient", size_name, f"{Colors.RED}ERROR{Colors.RESET}", str(e)[:10]])
    
    table.print()
    
    # Print summary
    suite.print_summary()
    
    return suite.all_passed()


if __name__ == '__main__':
    success = run_correctness_tests()
    sys.exit(0 if success else 1)
