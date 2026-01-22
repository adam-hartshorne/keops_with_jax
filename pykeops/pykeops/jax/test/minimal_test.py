#!/usr/bin/env python3
"""
Test with 2 i-points to see where gradient breaks
"""
import os
os.environ['PYKEOPS_JAX_MODE'] = '1'

import jax
import jax.numpy as jnp
import numpy as np
from pykeops.jax import Genred

# Test with 2 i-points, 1 j-point
x = jnp.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # (1, 2, 3)
y = jnp.array([[[7.0, 8.0, 9.0]]])  # (1, 1, 3)

print("x shape:", x.shape)
print("y shape:", y.shape)
print("x:", x.squeeze())
print("y:", y.squeeze())

# Manual gradient calculation
print("\n" + "="*60)
print("Manual Calculation")
print("="*60)

expected_grad = []
for i in range(2):
    grad_i = np.zeros(3)
    for j in range(1):
        diff = x[0, i, :] - y[0, j, :]
        grad_i += 2 * diff
        print(f"x[{i}] - y[{j}] = {diff}")
        print(f"  contribution: 2*diff = {2*diff}")
    expected_grad.append(grad_i)
    print(f"Total gradient for i={i}: {grad_i}")

expected_grad = np.array(expected_grad).reshape(1, 2, 3)
print(f"\nExpected gradient array:\n{expected_grad}")

# KeOps
op = Genred("Sum(SqDist(a, b))", ["a=Vi(3)", "b=Vj(3)"], 'Sum', 1)

print("\n" + "="*60)
print("KeOps Computation")
print("="*60)

# Forward
res = op(x, y)
print(f"Forward result: {res}")

# Gradient
res_fwd, vjp_fn = jax.vjp(lambda x_in: op(x_in, y), x)
cotangent = jnp.ones_like(res_fwd)
print(f"Cotangent shape: {cotangent.shape}")
grad = vjp_fn(cotangent)[0]

print(f"\nKeOps gradient:\n{grad}")
print(f"Expected gradient:\n{expected_grad}")
print(f"\nMatch: {np.allclose(grad, expected_grad)}")

if not np.allclose(grad, expected_grad):
    print(f"\nDifference:\n{grad - expected_grad}")
    print(f"\nRatio (KeOps/Expected):\n{grad / expected_grad}")