"""
Sanity check: Manually verify the JAX reference calculation is correct
"""
import numpy as np
import jax.numpy as jnp

import os


# Very simple test case
x = jnp.array([[1.0, 0.0], [2.0, 3.0]], dtype=jnp.float32)  # 2 points in 2D
y = jnp.array([[0.0, 1.0]], dtype=jnp.float32)              # 1 point in 2D

print("x =")
print(x)
print("\ny =")
print(y)

# Manual calculation
print("\n" + "="*60)
print("Manual calculation:")
print("="*60)

# Point 0: x[0] = [1, 0]
# Distance to y[0] = [0, 0]: sqrt((1-0)^2 + (0-0)^2) = sqrt(1) = 1
# Squared distance: 1
# exp(-1) ≈ 0.368
print("x[0]=[1,0] to y[0]=[0,0]:")
print(f"  dist^2 = (1-0)^2 + (0-0)^2 = 1")
print(f"  exp(-1) = {np.exp(-1):.6f}")
print(f"  sum (only 1 y point) = {np.exp(-1):.6f}")

# Point 1: x[1] = [0, 0]
# Distance to y[0] = [0, 0]: 0
# exp(0) = 1
print("\nx[1]=[0,0] to y[0]=[0,0]:")
print(f"  dist^2 = (0-0)^2 + (0-0)^2 = 0")
print(f"  exp(0) = {np.exp(0):.6f}")
print(f"  sum (only 1 y point) = {np.exp(0):.6f}")

print("\nExpected result: [[0.367879], [1.000000]]")

# JAX calculation
print("\n" + "="*60)
print("JAX calculation:")
print("="*60)

diffs = x[:, None, :] - y[None, :, :]  # (2, 1, 2)
print("diffs shape:", diffs.shape)
print("diffs =")
print(diffs)

sq_dists = jnp.sum(diffs**2, axis=-1)  # (2, 1)
print("\nsq_dists shape:", sq_dists.shape)
print("sq_dists =")
print(sq_dists)

kernel = jnp.exp(-sq_dists)  # (2, 1)
print("\nkernel =")
print(kernel)

result = jnp.sum(kernel, axis=1, keepdims=True)  # (2, 1)
print("\nresult (sum over axis=1) =")
print(result)

if jnp.allclose(result, jnp.array([[np.exp(-1)], [1.0]])):
    print("\n✓ JAX reference calculation is CORRECT")
else:
    print("\n✗ JAX reference calculation is WRONG!")

# Now test with KeOps
print("\n" + "="*60)
print("KeOps calculation:")
print("="*60)

import sys
sys.path.insert(0, '/home/claude/pykeops')

from pykeops.jax import Genred

formula = "Exp(-SqDist(x, y))"
aliases = ["x = Vi(2)", "y = Vj(2)"]
op = Genred(formula, aliases, reduction_op='Sum', axis=1)
result_keops = op(x, y)

print("KeOps result =")
print(result_keops)

error = float(jnp.max(jnp.abs(result - result_keops)))
print(f"\nError: {error:.2e}")

if error < 1e-5:
    print("✓ KeOps matches JAX!")
else:
    print(f"✗ KeOps MISMATCH!")
    print(f"Expected: {result.ravel()}")
    print(f"Got:      {result_keops.ravel()}")