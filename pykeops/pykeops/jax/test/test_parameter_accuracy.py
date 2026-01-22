#!/usr/bin/env python3
"""
Save JAX KeOps results to file for comparison
"""
import os
os.environ['PYKEOPS_JAX_MODE'] = '1'

import jax.numpy as jnp
import numpy as np
from pykeops.jax import Genred
import pickle

print("="*80)
print("JAX KEOPS: Saving results to file")
print("="*80)

# Use SAME random seed as PyTorch
np.random.seed(42)
N, M, D, E = 100, 50, 3, 2

# Generate data (SAME as PyTorch)
x_np = np.random.randn(N, D).astype(np.float32)
y_np = np.random.randn(M, D).astype(np.float32)
b_np = np.random.randn(M, E).astype(np.float32)
s_val = 0.5

# Convert to JAX
x_jax = jnp.array(x_np)
y_jax = jnp.array(y_np)
b_jax = jnp.array(b_np)
s_jax = jnp.array([s_val])

print(f"\nData shapes: x={x_jax.shape}, y={y_jax.shape}, b={b_jax.shape}")
print(f"Parameter s: {s_val}")

# Compute NumPy expected (ground truth)
print("\nComputing NumPy expected value...")
K_np = np.exp(-np.sum((x_np[:, None, :] - y_np[None, :, :]) ** 2, axis=-1) * s_val)
expected_np = K_np @ b_np

# Compute JAX expected (to show it differs from NumPy)
print("Computing JAX expected value...")
K_jax = jnp.exp(-jnp.sum((x_jax[:, None, :] - y_jax[None, :, :]) ** 2, axis=-1) * s_jax[0])
expected_jax = np.array(K_jax @ b_jax)

# JAX KeOps with parameters
print("Computing JAX KeOps result...")
formula = "Exp(-SqDist(x, y) * s) * b"
aliases = ["x=Vi(3)", "y=Vj(3)", "b=Vj(2)", "s=Pm(1)"]
op = Genred(formula, aliases, 'Sum', axis=1)
res_jax = np.array(op(x_jax, y_jax, b_jax, s_jax))

# Save results
output = {
    'platform': 'jax',
    'x': x_np,
    'y': y_np,
    'b': b_np,
    's': s_val,
    'numpy_expected': expected_np,
    'jax_expected': expected_jax,  # Show JAX differs from NumPy
    'keops_result': res_jax,
    'formula': formula,
    'N': N,
    'M': M,
    'D': D,
    'E': E
}

filename = 'jax_results.pkl'
with open(filename, 'wb') as f:
    pickle.dump(output, f)

print(f"\nSaved results to {filename}")
print(f"  NumPy expected shape: {expected_np.shape}")
print(f"  JAX expected shape:   {expected_jax.shape}")
print(f"  JAX KeOps shape:      {res_jax.shape}")
print(f"  NumPy expected[0]: {expected_np[0]}")
print(f"  JAX expected[0]:   {expected_jax[0]}")
print(f"  JAX KeOps[0]:      {res_jax[0]}")

# Quick verification
diff_vs_numpy = np.abs(res_jax - expected_np)
diff_vs_jax = np.abs(res_jax - expected_jax)
rel_numpy = diff_vs_numpy / (np.abs(expected_np) + 1e-10)
rel_jax = diff_vs_jax / (np.abs(expected_jax) + 1e-10)

print(f"\nJAX KeOps vs NumPy expected:")
print(f"  Max absolute error: {diff_vs_numpy.max():.6e}")
print(f"  Max relative error: {rel_numpy.max():.6e}")

print(f"\nJAX KeOps vs JAX expected:")
print(f"  Max absolute error: {diff_vs_jax.max():.6e}")
print(f"  Max relative error: {rel_jax.max():.6e}")

print(f"\nJAX expected vs NumPy expected:")
diff_expected = np.abs(expected_jax - expected_np)
print(f"  Max absolute error: {diff_expected.max():.6e}")

print("="*80)