#!/usr/bin/env python3
"""
Debug: Is the JAX expected value calculation wrong?
"""
import os
os.environ['PYKEOPS_JAX_MODE'] = '1'

import jax.numpy as jnp
import numpy as np
from pykeops.jax import Genred

# Use same seed
np.random.seed(42)
N, M, D, E = 100, 50, 3, 2

# NumPy data
x_np = np.random.randn(N, D).astype(np.float32)
y_np = np.random.randn(M, D).astype(np.float32)
b_np = np.random.randn(M, E).astype(np.float32)
s_val = 0.5

# JAX data (from NumPy)
x_jax = jnp.array(x_np)
y_jax = jnp.array(y_np)
b_jax = jnp.array(b_np)
s_jax = jnp.array([s_val])

print("="*80)
print("DEBUGGING JAX EXPECTED VALUE CALCULATION")
print("="*80)

# NumPy expected (known good from PyTorch test)
print("\n1. NumPy expected value:")
print("-"*80)
K_np = np.exp(-np.sum((x_np[:, None, :] - y_np[None, :, :]) ** 2, axis=-1) * s_val)
expected_np = K_np @ b_np
print(f"K shape: {K_np.shape}")
print(f"Expected shape: {expected_np.shape}")
print(f"Expected[0]: {expected_np[0]}")
print(f"Expected range: [{expected_np.min():.6f}, {expected_np.max():.6f}]")

# JAX expected (as in test)
print("\n2. JAX expected value:")
print("-"*80)
K_jax = jnp.exp(-jnp.sum((x_jax[:, None, :] - y_jax[None, :, :]) ** 2, axis=-1) * s_jax[0])
expected_jax = K_jax @ b_jax
print(f"K shape: {K_jax.shape}")
print(f"Expected shape: {expected_jax.shape}")
print(f"Expected[0]: {expected_jax[0]}")
print(f"Expected range: [{expected_jax.min():.6f}, {expected_jax.max():.6f}]")

# Compare
print("\n3. NumPy vs JAX expected:")
print("-"*80)
diff_expected = np.abs(np.array(expected_jax) - expected_np)
print(f"Max difference: {diff_expected.max():.6e}")
print(f"Identical: {np.allclose(expected_jax, expected_np)}")

# KeOps JAX
print("\n4. KeOps JAX result:")
print("-"*80)
formula = "Exp(-SqDist(x, y) * s) * b"
aliases = ["x=Vi(3)", "y=Vj(3)", "b=Vj(2)", "s=Pm(1)"]
op = Genred(formula, aliases, 'Sum', axis=1)
res_jax = op(x_jax, y_jax, b_jax, s_jax)
print(f"Result shape: {res_jax.shape}")
print(f"Result[0]: {res_jax[0]}")
print(f"Result range: [{res_jax.min():.6f}, {res_jax.max():.6f}]")

# Compare all
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

diff_keops_vs_jax_expected = np.abs(np.array(res_jax) - np.array(expected_jax))
diff_keops_vs_numpy_expected = np.abs(np.array(res_jax) - expected_np)

rel_jax = diff_keops_vs_jax_expected / (np.abs(np.array(expected_jax)) + 1e-10)
rel_numpy = diff_keops_vs_numpy_expected / (np.abs(expected_np) + 1e-10)

print(f"KeOps JAX vs JAX expected:")
print(f"  Max absolute error: {diff_keops_vs_jax_expected.max():.6e}")
print(f"  Max relative error: {rel_jax.max():.6e}")

print(f"\nKeOps JAX vs NumPy expected:")
print(f"  Max absolute error: {diff_keops_vs_numpy_expected.max():.6e}")
print(f"  Max relative error: {rel_numpy.max():.6e}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if diff_expected.max() > 1e-6:
    print("⚠️  JAX and NumPy compute different expected values!")
    print("    → Bug in expected value calculation")
elif rel_jax.max() > 1e-3:
    print("⚠️  KeOps JAX differs significantly from both expected values")
    print("    → Bug in KeOps JAX implementation")
else:
    print("✓  Everything matches - no bugs found")

print("="*80)