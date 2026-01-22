#!/usr/bin/env python3
"""
Minimal test to verify batched ranges are working correctly
"""
import sys
import os

os.environ['PYKEOPS_JAX_MODE'] = '1'

# Make sure we can find the test data
test_dir = "/media/adam/shared_folder/PycharmProjects/keops/pykeops/pykeops/jax/test"
if os.path.exists(test_dir):
    os.chdir(test_dir)
    sys.path.insert(0, test_dir)

import jax
import jax.numpy as jnp
import numpy as np

# Load test data
if os.path.exists("test_data.npz"):
    data = np.load("test_data.npz")
    x_np = data['x']
    y_np = data['y']
    print(f"Loaded test data: x={x_np.shape}, y={y_np.shape}")
else:
    print("Generating fresh test data...")
    np.random.seed(42)
    x_np = np.random.randn(2, 3, 3).astype(np.float32)
    y_np = np.random.randn(2, 2, 3).astype(np.float32)

# Convert to JAX
x_jax = jnp.array(x_np)
y_jax = jnp.array(y_np)

print("\n" + "=" * 60)
print("TEST: Batched computation with Genred")
print("=" * 60)

from pykeops.jax import Genred

# Create the reduction
formula = 'Sum((a-b)**2)'
aliases = ['a = Vi(3)', 'b = Vj(3)']
reduction = Genred(formula, aliases, reduction_op='Sum', axis=1)

# Compute
result = reduction(x_jax, y_jax)

print(f"\nInput shapes: x={x_jax.shape}, y={y_jax.shape}")
print(f"Result shape: {result.shape}")

print("\n--- Batch 0 ---")
print(result[0])

print("\n--- Batch 1 ---")
print(result[1])

# Compute ground truth manually for each batch
print("\n" + "=" * 60)
print("GROUND TRUTH (manual computation)")
print("=" * 60)

for b in range(2):
    x_b = x_jax[b]  # Shape: (3, 3)
    y_b = y_jax[b]  # Shape: (2, 3)

    # For each row in x, compute sum of squared distances to all rows in y
    result_b = jnp.zeros((3, 1))
    for i in range(3):
        dist_sum = 0.0
        for j in range(2):
            diff = x_b[i] - y_b[j]
            dist_sum += jnp.sum(diff ** 2)
        result_b = result_b.at[i, 0].set(dist_sum)

    print(f"\n--- Batch {b} (ground truth) ---")
    print(result_b)

# Check if results match ground truth
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

all_match = True
for b in range(2):
    x_b = x_jax[b]
    y_b = y_jax[b]

    # Manual ground truth
    gt = jnp.zeros((3, 1))
    for i in range(3):
        dist_sum = 0.0
        for j in range(2):
            diff = x_b[i] - y_b[j]
            dist_sum += jnp.sum(diff ** 2)
        gt = gt.at[i, 0].set(dist_sum)

    # Check match
    keops_result = result[b]
    max_diff = jnp.max(jnp.abs(keops_result - gt))

    if max_diff < 1e-5:
        print(f"✅ Batch {b}: MATCH (max diff = {max_diff:.2e})")
    else:
        print(f"❌ Batch {b}: MISMATCH (max diff = {max_diff:.2e})")
        print(f"   KeOps:  {keops_result.ravel()}")
        print(f"   Ground: {gt.ravel()}")
        all_match = False

if all_match:
    print("\n🎉 All batches PASSED!")
else:
    print("\n❌ Some batches FAILED - ranges bug still present")