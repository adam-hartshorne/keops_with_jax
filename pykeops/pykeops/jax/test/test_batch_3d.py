"""
Test with D=3 to see if the dimension multiplication matters
"""
import os
os.environ['PYKEOPS_JAX_MODE'] = '1'

import jax.numpy as jnp
from pykeops.jax import Genred

print("Testing with D=3 (where dimension multiplication MUST matter)")
print("="*70)

B, N, M, D = 2, 2, 3, 3

# Batch 0: x=[1,1,1], Batch 1: x=[10,10,10]
x = jnp.array([
    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],  # Batch 0: 2 points
    [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]  # Batch 1: 2 points
], dtype=jnp.float32)

y = jnp.zeros((B, M, D), dtype=jnp.float32)

print(f"x shape: {x.shape}")
print(f"Memory layout (total {B*N*D} elements):")
x_flat = x.reshape(-1)
for i, v in enumerate(x_flat):
    batch = i // (N*D)
    print(f"  [{i:2d}] = {v:5.1f}  (Batch {batch})")

formula = "SqDist(x, y)"
aliases = ["x=Vi(3)", "y=Vj(3)"]
op = Genred(formula, aliases, reduction_op='Sum', axis=1)

result = op(x, y)

print(f"\nResults:")
print(f"  Batch 0, point 0: {result[0, 0, 0]:.1f} (expected: 3*3 = 9.0)")
print(f"  Batch 1, point 0: {result[1, 0, 0]:.1f} (expected: 300*3 = 900.0)")

print(f"\nExpected calculations:")
print(f"  Batch 0: (1-0)^2 + (1-0)^2 + (1-0)^2 = 3, times M=3 points = 9.0")
print(f"  Batch 1: (10-0)^2 + (10-0)^2 + (10-0)^2 = 300, times M=3 points = 900.0")

if abs(result[1, 0, 0] - 900.0) < 0.1:
    print("\n✓ SUCCESS! Batching works correctly with D=3")
elif abs(result[1, 0, 0] - 9.0) < 0.1:
    print("\n✗ FAIL: Batch 1 reading Batch 0 (offset wrong)")
elif abs(result[1, 0, 0]) < 0.1:
    print("\n✗ FAIL: Batch 1 reading zeros (offset too far)")
else:
    print(f"\n? UNEXPECTED: Got {result[1, 0, 0]:.1f}")

print("\n" + "="*70)
print("Memory offset calculation:")
print(f"  WITHOUT * D: Batch 1 offset = 1 * {N} = {N} (points to element {N})")
print(f"  WITH * D:    Batch 1 offset = 1 * {N} * {D} = {N*D} (points to element {N*D})")
print(f"  Batch 1 actually starts at element {N*D}")