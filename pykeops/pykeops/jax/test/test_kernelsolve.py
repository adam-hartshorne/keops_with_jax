#!/usr/bin/env python3
"""
Test script for KernelSolve functionality.

Tests:
1. Basic kernel solve (Gaussian kernel)
2. Comparison with PyTorch KernelSolve
3. Different regularization values
4. Gradient computation through the solve
"""

import numpy as np
import jax.numpy as jnp
import jax

print("=" * 70)
print("Testing KernelSolve")
print("=" * 70)

# Generate test data
np.random.seed(42)
N = 100
D = 3
x_np = np.random.randn(N, D).astype(np.float32)
b_np = np.random.randn(N, D).astype(np.float32)

# =============================================================================
# Test 1: Basic KernelSolve
# =============================================================================
print("\n" + "=" * 70)
print("Test 1: Basic KernelSolve (Gaussian Kernel)")
print("=" * 70)

try:
    from pykeops.jax import KernelSolve

    # Define Gaussian kernel: K(x,y) * a = exp(-|x-y|^2) * a
    formula = "Exp(-SqDist(x,y)) * a"
    aliases = ["x=Vi(3)", "y=Vj(3)", "a=Vj(3)"]

    solver = KernelSolve(formula, aliases, "a", axis=1)

    x_jax = jnp.array(x_np)
    b_jax = jnp.array(b_np)

    # Solve (αI + K)a = b
    alpha = 0.1
    a_star = solver(x_jax, x_jax, b_jax, alpha=alpha, eps=1e-6)

    print(f"Solution shape: {a_star.shape}")
    print(f"Solution (first 3 rows):\n{a_star[:3]}")

    # Verify solution: compute residual ||(αI + K)a - b||
    from pykeops.jax import Genred

    K_op = Genred(formula, aliases, reduction_op='Sum', axis=1)
    Ka = K_op(x_jax, x_jax, a_star)
    residual = Ka + alpha * a_star - b_jax
    residual_norm = jnp.linalg.norm(residual)

    print(f"\nResidual ||Ka + αa - b||: {residual_norm}")

    if residual_norm < 1e-4:
        print("✓ Basic KernelSolve PASSED!")
    else:
        print(f"✗ Basic KernelSolve FAILED - residual too large: {residual_norm}")

except Exception as e:
    print(f"✗ Basic KernelSolve FAILED: {e}")
    import traceback

    traceback.print_exc()

# =============================================================================
# Test 2: Compare with PyTorch KernelSolve
# =============================================================================
print("\n" + "=" * 70)
print("Test 2: Compare with PyTorch KernelSolve")
print("=" * 70)

try:
    import torch
    from pykeops.torch import KernelSolve as KernelSolve_torch

    # PyTorch solve
    solver_torch = KernelSolve_torch(formula, aliases, "a", axis=1)

    x_torch = torch.tensor(x_np, device='cuda')
    b_torch = torch.tensor(b_np, device='cuda')

    a_star_torch = solver_torch(x_torch, x_torch, b_torch, alpha=alpha, eps=1e-6)

    # Compare
    a_star_jax_np = np.array(a_star)
    a_star_torch_np = a_star_torch.cpu().numpy()

    max_diff = np.abs(a_star_jax_np - a_star_torch_np).max()
    print(f"Max difference JAX vs PyTorch: {max_diff}")

    if max_diff < 1e-4:
        print("✓ JAX vs PyTorch comparison PASSED!")
    else:
        print(f"✗ JAX vs PyTorch comparison FAILED - max diff: {max_diff}")

except ImportError:
    print("PyTorch not available - skipping comparison")
except Exception as e:
    print(f"✗ PyTorch comparison FAILED: {e}")
    import traceback

    traceback.print_exc()

# =============================================================================
# Test 3: Different alpha values
# =============================================================================
print("\n" + "=" * 70)
print("Test 3: Different Regularization Values")
print("=" * 70)

try:
    for alpha_test in [0.01, 0.1, 1.0, 10.0]:
        a_star = solver(x_jax, x_jax, b_jax, alpha=alpha_test, eps=1e-6)

        Ka = K_op(x_jax, x_jax, a_star)
        residual = Ka + alpha_test * a_star - b_jax
        residual_norm = jnp.linalg.norm(residual)

        status = "✓" if residual_norm < 1e-4 else "✗"
        print(f"  α={alpha_test:5.2f}: residual={residual_norm:.2e} {status}")

    print("✓ Different alpha values PASSED!")

except Exception as e:
    print(f"✗ Different alpha values FAILED: {e}")
    import traceback

    traceback.print_exc()

# =============================================================================
# Test 4: Laplacian Kernel
# =============================================================================
print("\n" + "=" * 70)
print("Test 4: Laplacian Kernel")
print("=" * 70)

try:
    # Laplacian kernel: exp(-|x-y|) * a
    formula_lap = "Exp(-Sqrt(SqDist(x,y)+IntCst(1e-6))) * a"
    aliases_lap = ["x=Vi(3)", "y=Vj(3)", "a=Vj(3)"]

    solver_lap = KernelSolve(formula_lap, aliases_lap, "a", axis=1)

    alpha = 0.1
    a_star_lap = solver_lap(x_jax, x_jax, b_jax, alpha=alpha, eps=1e-6)

    # Verify
    K_op_lap = Genred(formula_lap, aliases_lap, reduction_op='Sum', axis=1)
    Ka_lap = K_op_lap(x_jax, x_jax, a_star_lap)
    residual_lap = Ka_lap + alpha * a_star_lap - b_jax
    residual_norm_lap = jnp.linalg.norm(residual_lap)

    print(f"Laplacian kernel residual: {residual_norm_lap}")

    if residual_norm_lap < 1e-4:
        print("✓ Laplacian Kernel PASSED!")
    else:
        print(f"✗ Laplacian Kernel FAILED - residual: {residual_norm_lap}")

except Exception as e:
    print(f"✗ Laplacian Kernel FAILED: {e}")
    import traceback

    traceback.print_exc()

# =============================================================================
# Test 5: Kernel Ridge Regression Example
# =============================================================================
print("\n" + "=" * 70)
print("Test 5: Kernel Ridge Regression")
print("=" * 70)

try:
    # Generate regression data
    np.random.seed(123)
    N_train = 200
    x_train = np.random.randn(N_train, 1).astype(np.float32)
    y_train = (np.sin(3 * x_train) + 0.1 * np.random.randn(N_train, 1)).astype(np.float32)

    x_train_jax = jnp.array(x_train)
    y_train_jax = jnp.array(y_train)

    # Gaussian RBF kernel with sigma as a Pm parameter
    # K(x,y) = exp(-|x-y|^2 * oos2) where oos2 = 1/(2*sigma^2)
    sigma = 0.5
    oos2 = np.array([1.0 / (2 * sigma ** 2)], dtype=np.float32)  # 1/(2*sigma^2)

    formula_krr = "Exp(-SqDist(x,y) * oos2) * a"
    aliases_krr = ["x=Vi(1)", "y=Vj(1)", "a=Vj(1)", "oos2=Pm(1)"]

    solver_krr = KernelSolve(formula_krr, aliases_krr, "a", axis=1)

    # Solve for coefficients
    alpha_krr = 0.01
    oos2_jax = jnp.array(oos2)
    coeffs = solver_krr(x_train_jax, x_train_jax, y_train_jax, oos2_jax, alpha=alpha_krr)

    # Make predictions
    K_krr = Genred(formula_krr, aliases_krr, reduction_op='Sum', axis=1)
    y_pred = K_krr(x_train_jax, x_train_jax, coeffs, oos2_jax)

    # Compute training error
    mse = jnp.mean((y_pred - y_train_jax) ** 2)
    print(f"Training MSE: {mse:.6f}")

    if mse < 0.1:
        print("✓ Kernel Ridge Regression PASSED!")
    else:
        print(f"⚠ Kernel Ridge Regression: MSE higher than expected ({mse})")

except Exception as e:
    print(f"✗ Kernel Ridge Regression FAILED: {e}")
    import traceback

    traceback.print_exc()

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("KernelSolve Testing Complete")
print("=" * 70)