#!/usr/bin/env python3
"""
Diagnose why Vi/Vj still produces wrong output dimension
"""
import os
os.environ['PYKEOPS_JAX_MODE'] = '1'

import jax.numpy as jnp
from pykeops.jax import LazyTensor, Vi, Vj

print("=" * 80)
print("Diagnosing Vi/Vj ndim tracking")
print("=" * 80)
print()

# Test data
b_vector = jnp.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]])  # (3, 2) - TWO dimensional output

print(f"b_vector.shape: {b_vector.shape}")
print()

# Method 1: Direct reshaping (works)
print("Method 1: Direct reshaping")
print("-" * 80)
b_j1 = LazyTensor(b_vector[None, :, :])
print(f"b_j1.variables[0].shape: {b_j1.variables[0].shape}")
print(f"b_j1.ndim: {b_j1.ndim}")
print(f"b_j1.axis: {b_j1.axis}")
print()

# Method 2: Using Vj (broken)
print("Method 2: Using Vj")
print("-" * 80)
b_j2 = Vj(b_vector)
print(f"b_j2.variables[0].shape: {b_j2.variables[0].shape}")
print(f"b_j2.ndim: {b_j2.ndim} <- Should be 2!")
print(f"b_j2.axis: {b_j2.axis}")
print()

# Check if they're the same
print("Comparison:")
print("-" * 80)
print(f"Same variable shape? {b_j1.variables[0].shape == b_j2.variables[0].shape}")
print(f"Same ndim? {b_j1.ndim == b_j2.ndim}")
print(f"Same axis? {b_j1.axis == b_j2.axis}")
print()

if b_j1.ndim != b_j2.ndim:
    print("❌ PROBLEM FOUND: ndim mismatch!")
    print(f"   Direct reshaping: ndim = {b_j1.ndim}")
    print(f"   Vj helper: ndim = {b_j2.ndim}")
    print()
    print("The Vj helper is not preserving the output dimension correctly.")
else:
    print("✓ ndim is the same, problem must be elsewhere")

print()
print("=" * 80)
print("Checking formula construction")
print("=" * 80)
print()

print("b_j1.formula:", b_j1.formula)
print("b_j2.formula:", b_j2.formula)
print()

# Check if ndim gets lost during operations
x = jnp.array([[1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0]])  # (2, 3)
y = jnp.array([[0.0, 0.0, 0.0]])  # (1, 3)

x_i1 = LazyTensor(x[:, None, :])
y_j1 = LazyTensor(y[None, :, :])

x_i2 = Vi(x)
y_j2 = Vj(y)

print("Testing operation: (x_i - y_j)")
print("-" * 80)

D1 = ((x_i1 - y_j1) ** 2).sum(-1)
print(f"Method 1 - D1.ndim: {D1.ndim}")

D2 = ((x_i2 - y_j2) ** 2).sum(-1)
print(f"Method 2 - D2.ndim: {D2.ndim}")
print()

K1 = (-D1 * 0.5).exp()
K2 = (-D2 * 0.5).exp()

print(f"Method 1 - K1.ndim: {K1.ndim}")
print(f"Method 2 - K2.ndim: {K2.ndim}")
print()

print("Testing operation: K * b_j")
print("-" * 80)

Kb1 = K1 * b_j1
Kb2 = K2 * b_j2

print(f"Method 1 - (K * b_j).ndim: {Kb1.ndim}")
print(f"Method 2 - (K * b_j).ndim: {Kb2.ndim} <- This should be 2!")
print()

if Kb1.ndim != Kb2.ndim:
    print("❌ ndim gets lost during K * b_j operation!")
    print()
    print("The problem might be in how LazyTensor.__mul__ handles ndim.")