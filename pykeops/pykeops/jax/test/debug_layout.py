#!/usr/bin/env python3
"""
Debug: Check if JAX is automatically flattening batched arrays
"""
import os
os.environ['PYKEOPS_JAX_MODE'] = '1'

import jax
import jax.numpy as jnp
import numpy as np

print("=" * 70)
print("DATA LAYOUT TEST")
print("=" * 70)

# Create test data
x = jnp.array([
    [[1.0, 2.0], [3.0, 4.0]],  # Batch 0: 2 points
    [[5.0, 6.0], [7.0, 8.0]]  # Batch 1: 2 points
], dtype=jnp.float32)

print(f"\nOriginal array x:")
print(f"  Shape: {x.shape}")  # Should be (2, 2, 2) = (batch, points, dims)
print(f"  Data:\n{x}")

# Check memory layout
print(f"\n  Memory layout:")
print(f"  x.ravel() = {x.ravel()}")

# What the kernel expects for batched data:
# For batch 0: points at indices 0, 1
# For batch 1: points at indices 2, 3
expected_flat = jnp.array([
    [1.0, 2.0],  # Batch 0, point 0
    [3.0, 4.0],  # Batch 0, point 1
    [5.0, 6.0],  # Batch 1, point 0
    [7.0, 8.0],  # Batch 1, point 1
], dtype=jnp.float32)

print(f"\n  Expected flattened shape: {expected_flat.shape}")
print(f"  Expected flattened data:\n{expected_flat}")

# Check if reshape gives us the right layout
reshaped = x.reshape(-1, x.shape[-1])
print(f"\n  x.reshape(-1, {x.shape[-1]}) shape: {reshaped.shape}")
print(f"  x.reshape(-1, {x.shape[-1]}) data:\n{reshaped}")

if jnp.allclose(reshaped, expected_flat):
    print("\n  ✅ Reshape gives correct flattening!")
else:
    print("\n  ❌ Reshape does NOT give correct flattening!")

# Check CUDA array interface
print("\n" + "=" * 70)
print("CUDA ARRAY INTERFACE")
print("=" * 70)

try:
    cuda_iface = x.__cuda_array_interface__
    print(f"\nOriginal 3D array:")
    print(f"  Shape: {cuda_iface['shape']}")
    print(f"  Strides: {cuda_iface.get('strides', 'C-contiguous')}")
    print(f"  Data pointer: {hex(cuda_iface['data'][0])}")

    cuda_iface_flat = reshaped.__cuda_array_interface__
    print(f"\nReshaped 2D array:")
    print(f"  Shape: {cuda_iface_flat['shape']}")
    print(f"  Strides: {cuda_iface_flat.get('strides', 'C-contiguous')}")
    print(f"  Data pointer: {hex(cuda_iface_flat['data'][0])}")

    if cuda_iface['data'][0] == cuda_iface_flat['data'][0]:
        print("\n  ✅ Same memory location (reshape is a view)")
    else:
        print("\n  ❌ Different memory location (reshape made a copy)")

except AttributeError:
    print("\n  ⚠️  No CUDA array interface (array not on GPU)")

# Test what KeOps actually receives
print("\n" + "=" * 70)
print("WHAT KEOPS SEES")
print("=" * 70)

from pykeops.jax import Genred

formula = "Exp(-SqDist(x, y))"
aliases = ["x = Vi(2)", "y = Vj(2)"]

# Test 1: Pass 3D directly
print("\nTest 1: Passing 3D array directly")
x_3d = jnp.array([[[1.0, 0.0]], [[2.0, 0.0]]], dtype=jnp.float32)  # (2, 1, 2)
y_3d = jnp.array([[[0.0, 0.0]], [[0.0, 0.0]]], dtype=jnp.float32)  # (2, 1, 2)
print(f"  x_3d shape: {x_3d.shape}")
print(f"  y_3d shape: {y_3d.shape}")

try:
    op = Genred(formula, aliases, reduction_op='Sum', axis=1)
    result = op(x_3d, y_3d)
    print(f"  Result shape: {result.shape}")
    print(f"  Result:\n{result}")

    # Expected results:
    # Batch 0: x=[1,0] to y=[0,0]: dist^2=1, exp(-1)≈0.368
    # Batch 1: x=[2,0] to y=[0,0]: dist^2=4, exp(-4)≈0.018
    expected = jnp.array([[[np.exp(-1)]], [[np.exp(-4)]]], dtype=jnp.float32)
    print(f"  Expected:\n{expected}")

    if jnp.allclose(result, expected, atol=1e-5):
        print("  ✅ Results match expected!")
    else:
        print("  ❌ Results DO NOT match!")
        print(f"  Max error: {jnp.max(jnp.abs(result - expected)):.6f}")
except Exception as e:
    print(f"  ❌ Error: {e}")

print()