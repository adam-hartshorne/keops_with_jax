"""
Memory Layout Diagnostic Test
Prints exactly what's in memory and how KeOps should access it
"""
import os

os.environ['PYKEOPS_JAX_MODE'] = '1'

import jax
import jax.numpy as jnp
import numpy as np
from pykeops.jax import Genred

print("=" * 80)
print("MEMORY LAYOUT DIAGNOSTIC TEST")
print("=" * 80)

# Create test data with very specific values
B, N, M, D = 2, 3, 5, 1

# Batch 0: all values = 1.0
# Batch 1: all values = 10.0
x = jnp.array([
    [[1.0], [1.0], [1.0]],  # Batch 0: 3 points, each with value 1.0
    [[10.0], [10.0], [10.0]]  # Batch 1: 3 points, each with value 10.0
], dtype=jnp.float32)

y = jnp.zeros((B, M, D), dtype=jnp.float32)

print(f"\nInput Configuration:")
print(f"  B (batches) = {B}")
print(f"  N (Vi points per batch) = {N}")
print(f"  M (Vj points per batch) = {M}")
print(f"  D (dimensions) = {D}")

print(f"\nInput x shape: {x.shape}")
print(f"Input y shape: {y.shape}")

print(f"\nInput x values:")
print(f"  Batch 0: {x[0].flatten()}")
print(f"  Batch 1: {x[1].flatten()}")

# Show actual memory layout
x_flat = x.reshape(-1)
y_flat = y.reshape(-1)

print(f"\nMemory Layout of x (as stored in RAM/VRAM):")
print(f"  Total elements: {len(x_flat)}")
print(f"  Layout: {x_flat}")
print(f"  Element indices:")
for i, val in enumerate(x_flat):
    batch_idx = i // (N * D)
    point_idx = (i % (N * D)) // D
    dim_idx = i % D
    print(f"    [{i:2d}] = {val:5.1f}  (Batch {batch_idx}, Point {point_idx}, Dim {dim_idx})")

print(f"\nMemory Layout of y (as stored in RAM/VRAM):")
print(f"  Total elements: {len(y_flat)}")
print(f"  Layout: {y_flat[:10]}... (all zeros)")

print("\n" + "=" * 80)
print("EXPECTED COMPUTATION")
print("=" * 80)

print(f"\nFormula: SqDist(x, y) = sum((x - y)^2)")
print(f"\nFor BATCH 0:")
print(f"  x values: [1.0, 1.0, 1.0]")
print(f"  y values: [0.0, 0.0, 0.0, 0.0, 0.0]")
print(f"  For each x[i], sum over all y[j]:")
print(f"    x[0]=1.0: (1-0)^2 * 5 = 1 * 5 = 5.0")
print(f"    x[1]=1.0: (1-0)^2 * 5 = 1 * 5 = 5.0")
print(f"    x[2]=1.0: (1-0)^2 * 5 = 1 * 5 = 5.0")
print(f"  Result shape: (3, 1), values: [5.0, 5.0, 5.0]")

print(f"\nFor BATCH 1:")
print(f"  x values: [10.0, 10.0, 10.0]")
print(f"  y values: [0.0, 0.0, 0.0, 0.0, 0.0]")
print(f"  For each x[i], sum over all y[j]:")
print(f"    x[3]=10.0: (10-0)^2 * 5 = 100 * 5 = 500.0")
print(f"    x[4]=10.0: (10-0)^2 * 5 = 100 * 5 = 500.0")
print(f"    x[5]=10.0: (10-0)^2 * 5 = 100 * 5 = 500.0")
print(f"  Result shape: (3, 1), values: [500.0, 500.0, 500.0]")

print(f"\nCombined output shape: (2, 3, 1)")
print(f"Expected output[0, 0, 0] = 5.0")
print(f"Expected output[1, 0, 0] = 500.0")

print("\n" + "=" * 80)
print("OFFSET CALCULATION (What the fix should do)")
print("=" * 80)

print(f"\nFor Vi (x) with shape (B={B}, N={N}, D={D}):")
print(f"  Total elements in memory: {B * N * D}")
print(f"  Batch 0 starts at ELEMENT: 0 * {N} * {D} = 0")
print(f"  Batch 1 starts at ELEMENT: 1 * {N} * {D} = {N * D}")
print(f"\n  OLD BUG (missing * D):")
print(f"    Batch 0 offset: 0 * {N} = 0 ✓ (works by accident)")
print(f"    Batch 1 offset: 1 * {N} = {N} ✗ (WRONG! Points to element {N}, should be {N * D})")
print(f"\n  CORRECT (with * D):")
print(f"    Batch 0 offset: 0 * {N} * {D} = 0 ✓")
print(f"    Batch 1 offset: 1 * {N} * {D} = {N * D} ✓")

print(f"\nFor Vj (y) with shape (B={B}, M={M}, D={D}):")
print(f"  Total elements in memory: {B * M * D}")
print(f"  Batch 0 starts at ELEMENT: 0 * {M} * {D} = 0")
print(f"  Batch 1 starts at ELEMENT: 1 * {M} * {D} = {M * D}")

print("\n" + "=" * 80)
print("RUNNING KEOPS")
print("=" * 80)

formula = "SqDist(x, y)"
aliases = ["x=Vi(1)", "y=Vj(1)"]
op = Genred(formula, aliases, reduction_op='Sum', axis=1)

print(f"\nFormula: {formula}")
print(f"Aliases: {aliases}")
print(f"Reduction: Sum over axis=1 (Vj)")

result = op(x, y)

print(f"\n" + "=" * 80)
print("ACTUAL RESULTS")
print("=" * 80)

print(f"\nResult shape: {result.shape}")
print(f"Result values:")
print(f"  Batch 0, Point 0: {result[0, 0, 0]:.1f} (expected 5.0)")
print(f"  Batch 0, Point 1: {result[0, 1, 0]:.1f} (expected 5.0)")
print(f"  Batch 0, Point 2: {result[0, 2, 0]:.1f} (expected 5.0)")
print(f"  Batch 1, Point 0: {result[1, 0, 0]:.1f} (expected 500.0)")
print(f"  Batch 1, Point 1: {result[1, 1, 0]:.1f} (expected 500.0)")
print(f"  Batch 1, Point 2: {result[1, 2, 0]:.1f} (expected 500.0)")

print(f"\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if abs(result[0, 0, 0] - 5.0) < 0.01:
    print("✓ Batch 0 is CORRECT")
else:
    print(f"✗ Batch 0 is WRONG: got {result[0, 0, 0]:.1f}, expected 5.0")

if abs(result[1, 0, 0] - 500.0) < 0.01:
    print("✓ Batch 1 is CORRECT")
else:
    print(f"✗ Batch 1 is WRONG: got {result[1, 0, 0]:.1f}, expected 500.0")

    if abs(result[1, 0, 0] - 5.0) < 0.01:
        print("\n  ** Batch 1 returned Batch 0's result! **")
        print("  ** This means the offset calculation is still wrong! **")
        print(f"  ** The kernel read from element 0 instead of element {N * D} **")
        print("\n  Possible causes:")
        print("    1. The fix wasn't applied to the right file")
        print("    2. The cache wasn't cleared properly")
        print("    3. There's another offset calculation somewhere")
        print(f"\n  Check that line 245-246 in Gpu_link_compile.py has:")
        print(f"    h_offsets[...] = b * nx * dimsx[i]  (not just b * nx)")
        print(f"    h_offsets[...] = b * ny * dimsy[j]  (not just b * ny)")

print("\n" + "=" * 80)
print("MEMORY ACCESS PATTERN ANALYSIS")
print("=" * 80)

print("\nIf the bug is still present, here's what's happening:")
print(f"  Batch 0: Reads elements [0, 1, 2] ✓")
print(f"  Batch 1: Reads elements [0, 1, 2] ✗ (should read [{N * D}, {N * D + 1}, {N * D + 2}])")
print("\nThis is because offset is calculated as:")
print(f"  WRONG: batch_1_offset = 1 * {N} = {N}")
print(f"  RIGHT: batch_1_offset = 1 * {N} * {D} = {N * D}")