#!/usr/bin/env python3
"""
Understand KeOps reduction semantics
"""
import os
os.environ['PYKEOPS_JAX_MODE'] = '1'

import jax.numpy as jnp
from pykeops.jax import Genred

print("="*80)
print("Understanding KeOps Reduction Semantics")
print("="*80)

x = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)  # shape (1, 3)
s = jnp.array([0.5], dtype=jnp.float32)

# What does axis=1 mean with only Vi?
print("\nTest: x * s with Vi, axis=1")
print("-"*80)
print(f"x shape: {x.shape}")
print(f"s shape: {s.shape}")

formula = "x * s"
aliases = ["x=Vi(3)", "s=Pm(1)"]
op = Genred(formula, aliases, 'Sum', axis=1)

result = op(x, s)
print(f"\nKeOps result: {result}")
print(f"Result shape: {result.shape}")

# What should it be?
print("\nPossible interpretations:")
print("-"*80)

# Option 1: No reduction (just x * s)
opt1 = x * s[0]
print(f"1. No reduction (x * s):           {opt1} [shape {opt1.shape}]")
print(f"   Matches KeOps: {jnp.allclose(result, opt1)}")

# Option 2: Sum over features
opt2 = jnp.sum(x * s[0], axis=-1, keepdims=True)
print(f"2. Sum over features (axis=-1):    {opt2} [shape {opt2.shape}]")
print(f"   Matches KeOps: {jnp.allclose(result, opt2)}")

# Option 3: Sum over batch
opt3 = jnp.sum(x * s[0], axis=0, keepdims=True)
print(f"3. Sum over batch (axis=0):        {opt3} [shape {opt3.shape}]")
print(f"   Matches KeOps: {jnp.allclose(result, opt3)}")

# Now test with multiple points
print("\n" + "="*80)
print("Test with multiple points")
print("="*80)

x_multi = jnp.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]], dtype=jnp.float32)  # shape (2, 3)

result_multi = op(x_multi, s)
print(f"\nx_multi shape: {x_multi.shape}")
print(f"KeOps result: {result_multi}")
print(f"Result shape: {result_multi.shape}")

print("\nPossible interpretations:")
print("-"*80)

# Option 1: No reduction (just x * s)
opt1_multi = x_multi * s[0]
print(f"1. No reduction (x * s):")
print(f"   {opt1_multi}")
print(f"   Shape: {opt1_multi.shape}")
print(f"   Matches KeOps: {jnp.allclose(result_multi, opt1_multi)}")

# Option 2: Sum over features
opt2_multi = jnp.sum(x_multi * s[0], axis=-1, keepdims=True)
print(f"2. Sum over features (axis=-1):")
print(f"   {opt2_multi}")
print(f"   Shape: {opt2_multi.shape}")
print(f"   Matches KeOps: {jnp.allclose(result_multi, opt2_multi)}")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)
print("KeOps with axis=1 and only Vi means:")
print("  - With Vi/Vj: reduce over j-dimension")
print("  - With only Vi: ??? (unclear semantics)")
print("We need to understand what KeOps SHOULD return vs what JAX is returning")
print("="*80)