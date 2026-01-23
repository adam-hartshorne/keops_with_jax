#!/usr/bin/env python3
"""
Test batched operations with a complex multi-variable kernel
Similar to varifold kernels with positions, normals, and weights
"""
import os


import jax.numpy as jnp
from pykeops.jax import Genred

print("=" * 70)
print("TESTING COMPLEX MULTI-VARIABLE KERNEL (6 inputs)")
print("=" * 70)

# Simulate a varifold-like kernel with:
# - x_i, y_j: positions (3D)
# - nx_i, ny_j: normals (3D)
# - w_i, w_j: weights (1D)
# Total: 6 variables

batch_size = 2
n_i = 10  # Number of points in x
n_j = 8  # Number of points in y

# Create test data
x = jnp.ones((batch_size, n_i, 3), dtype=jnp.float32) * 1.0
y = jnp.ones((batch_size, n_j, 3), dtype=jnp.float32) * 2.0
nx = jnp.ones((batch_size, n_i, 3), dtype=jnp.float32) * 0.5
ny = jnp.ones((batch_size, n_j, 3), dtype=jnp.float32) * 0.5
w_i = jnp.ones((batch_size, n_i, 1), dtype=jnp.float32) * 0.1
w_j = jnp.ones((batch_size, n_j, 1), dtype=jnp.float32) * 0.2

print(f"\nInput shapes:")
print(f"  x:   {x.shape}")
print(f"  y:   {y.shape}")
print(f"  nx:  {nx.shape}")
print(f"  ny:  {ny.shape}")
print(f"  w_i: {w_i.shape}")
print(f"  w_j: {w_j.shape}")

# Simple kernel: weighted squared distance with normal alignment
# K(x_i, y_j) = w_i * w_j * exp(-||x_i - y_j||^2) * (nx_i · ny_j)^2
formula = "wi * wj * Exp(-SqDist(x, y)) * Square(nx|ny)"
aliases = [
    "x = Vi(3)",  # Position i (Vi variable 0)
    "y = Vj(3)",  # Position j (Vj variable 0)
    "nx = Vi(3)",  # Normal i (Vi variable 1)
    "ny = Vj(3)",  # Normal j (Vj variable 1)
    "wi = Vi(1)",  # Weight i (Vi variable 2)
    "wj = Vj(1)",  # Weight j (Vj variable 2)
]

print(f"\nFormula: {formula}")
print(f"Number of variables: {len(aliases)} (3 Vi, 3 Vj)")

try:
    op = Genred(formula, aliases, reduction_op='Sum', axis=1)
    print("\n✅ Kernel compiled successfully!")

    # Test forward pass
    print("\nRunning forward pass...")
    result = op(x, y, nx, ny, w_i, w_j)

    print(f"\nResult shape: {result.shape}")
    print(f"Expected shape: ({batch_size}, {n_i}, 1)")

    if result.shape == (batch_size, n_i, 1):
        print("✅ Output shape is CORRECT!")
    else:
        print("❌ Output shape is WRONG!")

    # Check if batches are different
    batch0_mean = jnp.mean(result[0])
    batch1_mean = jnp.mean(result[1])

    print(f"\nBatch 0 mean: {batch0_mean:.6f}")
    print(f"Batch 1 mean: {batch1_mean:.6f}")

    # Since inputs are the same for both batches, results should be similar
    if jnp.abs(batch0_mean - batch1_mean) < 1e-5:
        print("✅ Batches produce consistent results!")
    else:
        print("❌ Batches are inconsistent!")

    # Test with different data per batch to verify independence
    print("\n" + "=" * 70)
    print("Testing batch independence with different inputs")
    print("=" * 70)

    x2 = jnp.stack([
        jnp.ones((n_i, 3)) * 1.0,  # Batch 0
        jnp.ones((n_i, 3)) * 5.0,  # Batch 1 - different values
    ], axis=0).astype(jnp.float32)

    result2 = op(x2, y, nx, ny, w_i, w_j)

    batch0_mean2 = jnp.mean(result2[0])
    batch1_mean2 = jnp.mean(result2[1])

    print(f"\nWith different x values:")
    print(f"  Batch 0 mean: {batch0_mean2:.6f}")
    print(f"  Batch 1 mean: {batch1_mean2:.6f}")
    print(f"  Difference: {jnp.abs(batch0_mean2 - batch1_mean2):.6f}")

    if jnp.abs(batch0_mean2 - batch1_mean2) > 1e-3:
        print("✅ Batches are INDEPENDENT (different inputs → different outputs)!")
        print("\n🎉 COMPLEX MULTI-VARIABLE KERNEL TEST PASSED!")
    else:
        print("❌ Batches are NOT independent (might be reading same data)!")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback

    traceback.print_exc()