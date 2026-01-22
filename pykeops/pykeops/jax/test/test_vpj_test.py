#!/usr/bin/env python3
"""
Test Vi-only operations with vjp disabled to see if backward pass is interfering
"""
import os

os.environ['PYKEOPS_JAX_MODE'] = '1'

import jax
import jax.numpy as jnp
from pykeops.jax import Genred

print("="*80)
print("Testing Vi-only with VJP DISABLED")
print("="*80)

D = 3
N = 5

key = jax.random.PRNGKey(42)
x = jax.random.normal(jax.random.split(key, 3)[0], (N, D), dtype=jnp.float32)
s = jnp.array([0.5], dtype=jnp.float32)

print(f"x[0]: {x[0]}")
print(f"s: {s}")

# Test 1: WITH vjp (default)
print("\n" + "="*80)
print("Test 1: WITH VJP (enable_vjp=True, default)")
print("="*80)

op_with_vjp = Genred("x * s", ["x=Vi(3)", "s=Pm(1)"], 'Sum', axis=1)
res_with_vjp = op_with_vjp(x, s)
expected = x * s[0]

print(f"Result[0]: {res_with_vjp[0]}")
print(f"Expected[0]: {expected[0]}")
print(f"Match: {jnp.allclose(res_with_vjp, expected)}")
print(f"Max error: {jnp.max(jnp.abs(res_with_vjp - expected)):.6e}")
print(f"Ratio: {res_with_vjp[0] / expected[0]}")

# Test 2: WITHOUT vjp
print("\n" + "="*80)
print("Test 2: WITHOUT VJP (enable_vjp=False)")
print("="*80)

# Access the internal function to disable vjp
from pykeops.jax.generic.generic_ops import make_keops_jax_op

# Create operator without vjp
op_no_vjp = make_keops_jax_op(
    "x * s",
    ("x=Vi(3)", "s=Pm(1)"),
    'Sum',
    axis=1,
    dtype_str='float32',
    enable_vjp=False  # <-- DISABLE VJP
)

res_no_vjp = op_no_vjp(x, s)

print(f"Result[0]: {res_no_vjp[0]}")
print(f"Expected[0]: {expected[0]}")
print(f"Match: {jnp.allclose(res_no_vjp, expected)}")
print(f"Max error: {jnp.max(jnp.abs(res_no_vjp - expected)):.6e}")
if not jnp.allclose(res_no_vjp, expected):
    print(f"Ratio: {res_no_vjp[0] / expected[0]}")

# Test 3: Check if JIT compilation affects it
print("\n" + "="*80)
print("Test 3: WITH JIT compilation")
print("="*80)

@jax.jit
def compute_keops(x, s):
    op = Genred("x * s", ["x=Vi(3)", "s=Pm(1)"], 'Sum', axis=1)
    return op(x, s)

res_jit = compute_keops(x, s)

print(f"Result[0]: {res_jit[0]}")
print(f"Expected[0]: {expected[0]}")
print(f"Match: {jnp.allclose(res_jit, expected)}")
print(f"Max error: {jnp.max(jnp.abs(res_jit - expected)):.6e}")
if not jnp.allclose(res_jit, expected):
    print(f"Ratio: {res_jit[0] / expected[0]}")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)
print("If 'WITHOUT VJP' passes but 'WITH VJP' fails:")
print("  → Bug is in the custom_vjp implementation affecting forward pass")
print("If both fail:")
print("  → Bug is in forward pass itself")
print("If 'WITH JIT' differs from without:")
print("  → JIT compilation is causing issues")
print("="*80)