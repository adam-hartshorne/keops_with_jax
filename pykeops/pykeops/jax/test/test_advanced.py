#!/usr/bin/env python3
"""
KeOps JAX Advanced Features Tests
=================================
Tests for advanced KeOps features not covered in the basic API tests.

All tests compare JAX KeOps against PyTorch KeOps (ground truth).

Tests cover:
- All Reduction Types (Min, Max, ArgMin, KMin, LogSumExp, etc.)
- Exotic Math Operations (Trig, Step, Abs, Sign, Clamp)
- Batched Operations
- Higher-Order Gradients
- Various Genred formulas
"""

import sys
import numpy as np

import jax
import jax.numpy as jnp

from test_utils import (
    TestSuite, print_header, print_subheader, print_warning,
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


# =============================================================================
# 1. Advanced Reductions
# =============================================================================

def test_reduction(reduction_op, n, m, d, opt_arg=None, extra_args=False):
    """Generic test for reductions."""
    data = generate_data_np(n, m, d)

    # Setup Formula
    if extra_args:  # For weighted reductions
        formula = "-SqDist(x, y)"  # Weights handled via aliases usually or formula2
        # Special case for weighted logsumexp/softmax
        if "Weight" in reduction_op:
            formula = "-SqDist(x, y)"
            aliases = [f"x=Vi({d})", f"y=Vj({d})", f"b=Vj({d})"]
            formula2 = "b"
        else:
            formula = "SqDist(x,y)"
            aliases = [f"x=Vi({d})", f"y=Vj({d})"]
            formula2 = None
    else:
        formula = "SqDist(x,y)"
        aliases = [f"x=Vi({d})", f"y=Vj({d})"]
        formula2 = None

    # Additional B array for weights if needed
    if extra_args:
        np.random.seed(SEED + 100)
        b_np = np.random.randn(m, d).astype(np.float32)

    # --- JAX ---
    op_jax = Genred(formula, aliases, reduction_op=reduction_op, axis=1,
                    opt_arg=opt_arg, formula2=formula2)

    if extra_args:
        result_jax = op_jax(jnp.array(data['x']), jnp.array(data['y']), jnp.array(b_np))
    else:
        result_jax = op_jax(jnp.array(data['x']), jnp.array(data['y']))

    # --- PyTorch ---
    op_torch = Genred_torch(formula, aliases, reduction_op=reduction_op, axis=1,
                            opt_arg=opt_arg, formula2=formula2)

    if extra_args:
        result_torch = op_torch(
            torch.tensor(data['x'], device='cuda'),
            torch.tensor(data['y'], device='cuda'),
            torch.tensor(b_np, device='cuda')
        )
    else:
        result_torch = op_torch(
            torch.tensor(data['x'], device='cuda'),
            torch.tensor(data['y'], device='cuda')
        )

    # Comparison Logic
    if isinstance(result_jax, tuple):
        # Combined reductions (e.g. Min_ArgMin) return tuple (values, indices)
        val_match, val_diff = compare_arrays(result_jax[0], result_torch[0].cpu().numpy(), rtol=RTOL, atol=ATOL)

        # Check indices (exact integer match)
        idx_jax = np.array(result_jax[1]).astype(np.int32)
        idx_torch = result_torch[1].cpu().numpy().astype(np.int32)
        idx_match = np.all(idx_jax == idx_torch)

        if not val_match: return False, f"Values mismatch: {val_diff}"
        if not idx_match: return False, "Indices mismatch"
        return True, 0.0

    elif "Arg" in reduction_op and "Min_" not in reduction_op and "Max_" not in reduction_op:
        # Pure index reductions (ArgMin, ArgMax)
        idx_jax = np.array(result_jax).astype(np.int32)
        idx_torch = result_torch.cpu().numpy().astype(np.int32)
        if np.all(idx_jax == idx_torch):
            return True, 0.0
        return False, "Indices mismatch"

    else:
        # Standard value reductions
        return compare_arrays(result_jax, result_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)


# =============================================================================
# 2. Exotic Math
# =============================================================================

def test_lazy_math(op_name, n=50, m=40, d=3):
    """Test LazyTensor math operations."""
    np.random.seed(SEED)

    # Generate data
    x_np = np.random.randn(n, d).astype(np.float32)
    y_np = np.random.randn(m, d).astype(np.float32)

    # Adjust domains for specific ops to avoid NaNs
    if op_name in ["Sqrt", "Log"]:
        x_np = np.abs(x_np) + 0.1
        y_np = np.abs(y_np) + 0.1
    if op_name == "Clamp":
        x_np *= 3  # Wider range

    # --- JAX ---
    x_i_jax = LazyTensor(jnp.array(x_np)[:, None, :])
    y_j_jax = LazyTensor(jnp.array(y_np)[None, :, :])

    if op_name == "Sin":
        res_jax = (x_i_jax - y_j_jax).sin().sum(1)
    elif op_name == "Cos":
        res_jax = (x_i_jax - y_j_jax).cos().sum(1)
    elif op_name == "SinCos":
        res_jax = ((x_i_jax - y_j_jax).sin() + x_i_jax.cos()).sum(1)
    elif op_name == "AbsSign":
        diff = x_i_jax - y_j_jax
        res_jax = (diff.abs() * diff.sign()).sum(1)
    elif op_name == "Step":
        res_jax = (x_i_jax - y_j_jax).step().sum(1)
    elif op_name == "Clamp":
        res_jax = (x_i_jax - y_j_jax).clamp(-1.0, 1.0).sum(1)
    elif op_name == "Sqrt":
        res_jax = (x_i_jax + y_j_jax).sqrt().sum(1)
    elif op_name == "Square":
        res_jax = ((x_i_jax - y_j_jax) ** 2).sum(1)
    elif op_name == "Log":
        res_jax = (x_i_jax + y_j_jax).log().sum(1)

    # --- PyTorch ---
    x_i_torch = LazyTensor_torch(torch.tensor(x_np, device='cuda')[:, None, :])
    y_j_torch = LazyTensor_torch(torch.tensor(y_np, device='cuda')[None, :, :])

    if op_name == "Sin":
        res_torch = (x_i_torch - y_j_torch).sin().sum(1)
    elif op_name == "Cos":
        res_torch = (x_i_torch - y_j_torch).cos().sum(1)
    elif op_name == "SinCos":
        res_torch = ((x_i_torch - y_j_torch).sin() + x_i_torch.cos()).sum(1)
    elif op_name == "AbsSign":
        diff = x_i_torch - y_j_torch
        res_torch = (diff.abs() * diff.sign()).sum(1)
    elif op_name == "Step":
        res_torch = (x_i_torch - y_j_torch).step().sum(1)
    elif op_name == "Clamp":
        res_torch = (x_i_torch - y_j_torch).clamp(-1.0, 1.0).sum(1)
    elif op_name == "Sqrt":
        res_torch = (x_i_torch + y_j_torch).sqrt().sum(1)
    elif op_name == "Square":
        res_torch = ((x_i_torch - y_j_torch) ** 2).sum(1)
    elif op_name == "Log":
        res_torch = (x_i_torch + y_j_torch).log().sum(1)

    return compare_arrays(res_jax, res_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)


# =============================================================================
# 3. Batched Ops
# =============================================================================

def test_batched(mode, n=50, m=40, d=3, batch=3):
    data = generate_data_np(n, m, d, batch=batch)

    # JAX
    x_i_jax = LazyTensor(jnp.array(data['x'])[:, :, None, :])
    y_j_jax = LazyTensor(jnp.array(data['y'])[:, None, :, :])

    # PyTorch
    x_i_torch = LazyTensor_torch(torch.tensor(data['x'], device='cuda')[:, :, None, :])
    y_j_torch = LazyTensor_torch(torch.tensor(data['y'], device='cuda')[:, None, :, :])

    if mode == "SqDist":
        res_jax = ((x_i_jax - y_j_jax) ** 2).sum(-1).sum(2)
        res_torch = ((x_i_torch - y_j_torch) ** 2).sum(-1).sum(2)
    elif mode == "Gaussian":
        sigma = 0.5
        K_jax = (-((x_i_jax - y_j_jax) ** 2).sum(-1) / (2 * sigma ** 2)).exp()
        res_jax = K_jax.sum(2)

        K_torch = (-((x_i_torch - y_j_torch) ** 2).sum(-1) / (2 * sigma ** 2)).exp()
        res_torch = K_torch.sum(2)

    return compare_arrays(res_jax, res_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)


# =============================================================================
# 4. Higher Order Grads
# =============================================================================

def test_hessian():
    """Documents limitation: FFI calls cannot be differentiated twice."""
    data = generate_data_np(30, 25, 3)
    x_jax = jnp.array(data['x'])
    y_jax = jnp.array(data['y'])
    v_jax = jnp.ones_like(x_jax)

    op_jax = Genred("SqDist(x, y)", ["x=Vi(3)", "y=Vj(3)"], reduction_op='Sum', axis=1)

    def grad_fn(x):
        return jax.grad(lambda x_in: jnp.sum(op_jax(x_in, y_jax)))(x)

    try:
        # Try second order derivative
        _, hvp_jax = jax.jvp(grad_fn, (x_jax,), (v_jax,))
        return False, "Unexpected: Higher order gradients shouldn't work yet"
    except ValueError as e:
        if "cannot be differentiated" in str(e):
            print_warning("Confirmed: Higher-order gradients trigger expected error (FFI limitation)")
            return True, 0.0
        return False, f"Unexpected error type: {e}"


# =============================================================================
# 5. Formulas
# =============================================================================

def test_formula(name, n=100, m=80, d=3):
    data = generate_data_np(n, m, d)

    if name == "Laplacian":
        formula = "Exp(-Sqrt(SqDist(x,y)))"
        aliases = ["x=Vi(3)", "y=Vj(3)"]
        extra_args_jax = []
        extra_args_torch = []
    elif name == "Cauchy":
        formula = "Inv(IntCst(1) + SqDist(x,y))"
        aliases = ["x=Vi(3)", "y=Vj(3)"]
        extra_args_jax = []
        extra_args_torch = []
    elif name == "WeightedSum":
        formula = "Exp(-SqDist(x,y) * s) * b"
        aliases = ["x=Vi(3)", "y=Vj(3)", "b=Vj(3)", "s=Pm(1)"]
        # Make dummy b
        np.random.seed(SEED + 1)
        b = np.random.randn(m, d).astype(np.float32)
        extra_args_jax = [jnp.array(b), jnp.array(data['sigma'])]
        extra_args_torch = [torch.tensor(b, device='cuda'), torch.tensor(data['sigma'], device='cuda')]

    # JAX
    op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1)
    res_jax = op_jax(jnp.array(data['x']), jnp.array(data['y']), *extra_args_jax)

    # PyTorch
    op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
    res_torch = op_torch(
        torch.tensor(data['x'], device='cuda'),
        torch.tensor(data['y'], device='cuda'),
        *extra_args_torch
    )

    return compare_arrays(res_jax, res_torch.cpu().numpy(), rtol=RTOL, atol=ATOL)


# =============================================================================
# Main Runner
# =============================================================================

def main():
    print_header("KeOps JAX Advanced Features", "Comparing JAX vs PyTorch Ground Truth")
    print_environment_info()

    if not TORCH_AVAILABLE:
        print("⚠️  PyTorch KeOps not available. Skipping advanced tests.")
        return 0

    suite = TestSuite("Advanced Tests", "Feature parity check")
    N, M, D = 100, 80, 3

    # 1. Reductions
    print_subheader("1. Specialized Reductions")

    reductions = [
        ("Min", None, False),
        ("Max", None, False),
        ("ArgMin", None, False),
        ("ArgMax", None, False),
        ("Min_ArgMin", None, False),
        ("Max_ArgMax", None, False),
        ("KMin", 5, False),
        ("ArgKMin", 5, False),
        ("KMin_ArgKMin", 5, False),
        ("LogSumExp", None, False),
        ("LogSumExpWeight", None, True),
        ("SumSoftMaxWeight", None, True),
    ]

    for op, arg, extra in reductions:
        name = f"{op}" + (f"(K={arg})" if arg else "")
        run_test(
            name,
            lambda o=op, k=arg, e=extra: test_reduction(o, N, M, D, k, e),
            suite
        )

    # 2. Exotic Math
    print_subheader("2. Exotic Math Operations")
    math_ops = ["Sin", "Cos", "SinCos", "AbsSign", "Step", "Clamp", "Sqrt", "Square", "Log"]
    for op in math_ops:
        run_test(f"Math: {op}", lambda o=op: test_lazy_math(o), suite)

    # 3. Batched
    print_subheader("3. Batched Operations")
    run_test("Batched SqDist", lambda: test_batched("SqDist"), suite)
    run_test("Batched Gaussian", lambda: test_batched("Gaussian"), suite)

    # 4. Higher Order
    print_subheader("4. Higher Order Gradients")
    run_test("Hessian-Vector (Limitation Check)", test_hessian, suite)

    # 5. Formulas
    print_subheader("5. Complex Formulas")
    run_test("Kernel: Laplacian", lambda: test_formula("Laplacian"), suite)
    run_test("Kernel: Cauchy", lambda: test_formula("Cauchy"), suite)
    run_test("Kernel: WeightedSum", lambda: test_formula("WeightedSum"), suite)

    # Summary
    suite.print_summary()
    return 0 if suite.all_passed() else 1


if __name__ == "__main__":
    sys.exit(main())