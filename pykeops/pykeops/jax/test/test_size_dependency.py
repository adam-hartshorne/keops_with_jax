#!/usr/bin/env python3
"""
Critical test: Does unbatched work at N=500, M=500?
If yes, we have a simple workaround.
"""
import os

os.environ['PYKEOPS_JAX_MODE'] = '1'

import numpy as np
import jax.numpy as jnp
from pykeops.jax import LazyTensor


def compute_reference(x_np, y_np, u_np, v_np, gamma_s, gamma_n):
    """Compute reference using pure numpy."""
    B, N, _ = x_np.shape
    _, M, _ = y_np.shape
    result = np.zeros((B, N), dtype=np.float32)

    for b in range(B):
        for i in range(N):
            x_i = x_np[b, i:i + 1, :]
            y_all = y_np[b, :, :]
            u_i = u_np[b, i:i + 1, :]
            v_all = v_np[b, :, :]

            sq_dist = np.sum((x_i - y_all) ** 2, axis=-1)
            dot_prod = np.sum(u_i * v_all, axis=-1)
            K = np.exp(-sq_dist * gamma_s) * np.exp(dot_prod * gamma_n)
            result[b, i] = K.sum()

    return result


def batched_keops(x, y, u, v, gamma_s, gamma_n):
    """Standard batched KeOps (buggy for large N/M)."""
    x_L = LazyTensor(x[:, :, None, :])
    y_L = LazyTensor(y[:, None, :, :])
    u_L = LazyTensor(u[:, :, None, :])
    v_L = LazyTensor(v[:, None, :, :])

    sq_dist = x_L.sqdist(y_L)
    dot_prod = (u_L * v_L).sum(-1)
    K = (-sq_dist * gamma_s).exp() * (dot_prod * gamma_n).exp()
    result = K.sum(axis=2)

    if result.ndim == 3 and result.shape[-1] == 1:
        result = result.squeeze(-1)

    return result


def unbatched_keops(x_single, y_single, u_single, v_single, gamma_s, gamma_n):
    """Unbatched KeOps (no batch dimension)."""
    x_L = LazyTensor(x_single[:, None, :])
    y_L = LazyTensor(y_single[None, :, :])
    u_L = LazyTensor(u_single[:, None, :])
    v_L = LazyTensor(v_single[None, :, :])

    sq_dist = x_L.sqdist(y_L)
    dot_prod = (u_L * v_L).sum(-1)
    K = (-sq_dist * gamma_s).exp() * (dot_prod * gamma_n).exp()
    result = K.sum(axis=1)

    if result.ndim == 2 and result.shape[-1] == 1:
        result = result.squeeze(-1)

    return result


def loop_over_batches(x, y, u, v, gamma_s, gamma_n):
    """Workaround: Loop over batches."""
    B = x.shape[0]
    results = []
    for b in range(B):
        result_b = unbatched_keops(x[b], y[b], u[b], v[b], gamma_s, gamma_n)
        results.append(result_b)
    return jnp.stack(results, axis=0)


# Test sizes
test_configs = [
    (150, 150, "Small (should pass)"),
    (200, 200, "Medium (at threshold)"),
    (500, 500, "Large (should fail for batched)"),
]

print("=" * 80)
print("Testing Different Approaches at Various Sizes")
print("=" * 80)

for N, M, desc in test_configs:
    print(f"\n{'=' * 80}")
    print(f"{desc}: N={N}, M={M}")
    print("=" * 80)

    # Create data
    rng = np.random.RandomState(42)
    B, D = 2, 3

    x_np = rng.randn(B, N, D).astype(np.float32)
    y_np = rng.randn(B, M, D).astype(np.float32)
    u_np = rng.randn(B, N, D).astype(np.float32)
    v_np = rng.randn(B, M, D).astype(np.float32)
    u_np /= np.linalg.norm(u_np, axis=-1, keepdims=True)
    v_np /= np.linalg.norm(v_np, axis=-1, keepdims=True)

    gamma_s = 0.25
    gamma_n = 0.75

    # Reference
    print("\nComputing reference (numpy)...")
    ref = compute_reference(x_np, y_np, u_np, v_np, gamma_s, gamma_n)
    print(f"  Reference [0, -1] = {ref[0, -1]:.6f}")

    # Convert to JAX
    x = jnp.array(x_np)
    y = jnp.array(y_np)
    u = jnp.array(u_np)
    v = jnp.array(v_np)

    # Test 1: Batched KeOps
    print("\n1. Batched KeOps (standard approach):")
    result_batched = batched_keops(x, y, u, v, gamma_s, gamma_n)
    result_batched_np = np.array(result_batched)
    diff1 = np.abs(ref - result_batched_np).max()
    print(f"   Max diff: {diff1:.2e}")
    print(f"   Status: {'✓ PASS' if diff1 < 1e-4 else '✗ FAIL'}")
    if diff1 >= 1e-4:
        print(f"   [0, -1]: ref={ref[0, -1]:.6f}, jax={result_batched_np[0, -1]:.6f}")

    # Test 2: Loop over batches
    print("\n2. Loop over batches (workaround):")
    result_loop = loop_over_batches(x, y, u, v, gamma_s, gamma_n)
    result_loop_np = np.array(result_loop)
    diff2 = np.abs(ref - result_loop_np).max()
    print(f"   Max diff: {diff2:.2e}")
    print(f"   Status: {'✓ PASS' if diff2 < 1e-4 else '✗ FAIL'}")
    if diff2 >= 1e-4:
        print(f"   [0, -1]: ref={ref[0, -1]:.6f}, jax={result_loop_np[0, -1]:.6f}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("If loop-over-batches passes at all sizes:")
print("  → The bug is specifically in batched LazyTensor implementation")
print("  → Workaround: Always loop over batches manually")
print("\nIf loop-over-batches also fails at large sizes:")
print("  → The bug is in the core kernel, not batch handling")
print("  → No simple workaround available")