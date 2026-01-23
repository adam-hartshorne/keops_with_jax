#!/usr/bin/env python3
"""
Test both LazyTensor and Genred with simple and complex kernels
to isolate where the batched bug occurs.
"""
import os

os.environ['PYKEOPS_JAX_MODE'] = '1'

import numpy as np
import jax.numpy as jnp
from pykeops.jax import LazyTensor, Genred


def compute_reference_gaussian(x_np, y_np, sigma):
    """Manual numpy Gaussian kernel: K(x,y) = exp(-||x-y||^2 / sigma^2)"""
    B, N, D = x_np.shape
    M = y_np.shape[1]
    result = np.zeros((B, N), dtype=np.float32)

    gamma = 1.0 / (sigma ** 2)

    for b in range(B):
        for i in range(N):
            x_i = x_np[b, i:i + 1, :]
            y_all = y_np[b, :, :]
            sq_dist = np.sum((x_i - y_all) ** 2, axis=-1)
            K = np.exp(-sq_dist * gamma)
            result[b, i] = K.sum()

    return result


def compute_reference_varifold(x_np, y_np, u_np, v_np, gamma_s, gamma_n):
    """Manual numpy varifold kernel."""
    B, N, D = x_np.shape
    M = y_np.shape[1]
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


def test_gaussian_kernel(N, M, D=3, B=2, seed=42):
    """Test simple Gaussian kernel: K(x,y) = exp(-||x-y||^2/sigma^2)"""
    print(f"\n{'=' * 80}")
    print(f"Gaussian Kernel Test: N={N}, M={M}, D={D}, B={B}")
    print(f"{'=' * 80}")

    rng = np.random.RandomState(seed)
    x_np = rng.randn(B, N, D).astype(np.float32)
    y_np = rng.randn(B, M, D).astype(np.float32)

    sigma = 1.0
    gamma = 1.0 / (sigma ** 2)

    # Reference
    print("\nComputing reference...")
    ref = compute_reference_gaussian(x_np, y_np, sigma)
    print(f"  Reference [0, -1] = {ref[0, -1]:.6f}")

    x = jnp.array(x_np)
    y = jnp.array(y_np)

    # Test 1: LazyTensor batched
    print("\n1. LazyTensor (batched):")
    try:
        x_L = LazyTensor(x[:, :, None, :])
        y_L = LazyTensor(y[:, None, :, :])
        sq_dist = x_L.sqdist(y_L)
        K = (-sq_dist * gamma).exp()
        result_lazy_batched = K.sum(axis=2)

        if result_lazy_batched.ndim == 3 and result_lazy_batched.shape[-1] == 1:
            result_lazy_batched = result_lazy_batched.squeeze(-1)

        result_lazy_batched_np = np.array(result_lazy_batched)
        diff = np.abs(ref - result_lazy_batched_np).max()
        print(f"   Max diff: {diff:.2e}")
        print(f"   Status: {'✓ PASS' if diff < 1e-4 else '✗ FAIL'}")
        if diff >= 1e-4:
            print(f"   [0, -1]: ref={ref[0, -1]:.6f}, jax={result_lazy_batched_np[0, -1]:.6f}")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")

    # Test 2: LazyTensor unbatched (loop)
    print("\n2. LazyTensor (unbatched loop):")
    try:
        results = []
        for b in range(B):
            x_b = x[b]
            y_b = y[b]
            x_L = LazyTensor(x_b[:, None, :])
            y_L = LazyTensor(y_b[None, :, :])
            sq_dist = x_L.sqdist(y_L)
            K = (-sq_dist * gamma).exp()
            result_b = K.sum(axis=1)

            if result_b.ndim == 2 and result_b.shape[-1] == 1:
                result_b = result_b.squeeze(-1)

            results.append(result_b)

        result_lazy_loop = jnp.stack(results, axis=0)
        result_lazy_loop_np = np.array(result_lazy_loop)
        diff = np.abs(ref - result_lazy_loop_np).max()
        print(f"   Max diff: {diff:.2e}")
        print(f"   Status: {'✓ PASS' if diff < 1e-4 else '✗ FAIL'}")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")

    # Test 3: Genred batched
    print("\n3. Genred (batched):")
    try:
        formula = "Exp(-g * SqDist(x, y))"
        aliases = [f"x = Vi({D})", f"y = Vj({D})", "g = Pm(1)"]
        genred = Genred(formula, aliases, reduction_op='Sum', axis=1)

        g = jnp.array([gamma])
        result_genred = genred(x, y, g)

        if result_genred.shape[-1] == 1:
            result_genred = result_genred.squeeze(-1)

        result_genred_np = np.array(result_genred)
        diff = np.abs(ref - result_genred_np).max()
        print(f"   Max diff: {diff:.2e}")
        print(f"   Status: {'✓ PASS' if diff < 1e-4 else '✗ FAIL'}")
        if diff >= 1e-4:
            print(f"   [0, -1]: ref={ref[0, -1]:.6f}, jax={result_genred_np[0, -1]:.6f}")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")

    # Test 4: Genred unbatched (loop)
    print("\n4. Genred (unbatched loop):")
    try:
        formula = "Exp(-g * SqDist(x, y))"
        aliases = [f"x = Vi({D})", f"y = Vj({D})", "g = Pm(1)"]
        genred = Genred(formula, aliases, reduction_op='Sum', axis=1)

        g = jnp.array([gamma])
        results = []
        for b in range(B):
            result_b = genred(x[b], y[b], g)
            if result_b.shape[-1] == 1:
                result_b = result_b.squeeze(-1)
            results.append(result_b)

        result_genred_loop = jnp.stack(results, axis=0)
        result_genred_loop_np = np.array(result_genred_loop)
        diff = np.abs(ref - result_genred_loop_np).max()
        print(f"   Max diff: {diff:.2e}")
        print(f"   Status: {'✓ PASS' if diff < 1e-4 else '✗ FAIL'}")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")


def test_varifold_kernel(N, M, D=3, B=2, seed=42):
    """Test complex varifold kernel."""
    print(f"\n{'=' * 80}")
    print(f"Varifold Kernel Test: N={N}, M={M}, D={D}, B={B}")
    print(f"{'=' * 80}")

    rng = np.random.RandomState(seed)
    x_np = rng.randn(B, N, D).astype(np.float32)
    y_np = rng.randn(B, M, D).astype(np.float32)
    u_np = rng.randn(B, N, D).astype(np.float32)
    v_np = rng.randn(B, M, D).astype(np.float32)
    u_np /= np.linalg.norm(u_np, axis=-1, keepdims=True)
    v_np /= np.linalg.norm(v_np, axis=-1, keepdims=True)

    gamma_s = 0.25
    gamma_n = 0.75

    # Reference
    print("\nComputing reference...")
    ref = compute_reference_varifold(x_np, y_np, u_np, v_np, gamma_s, gamma_n)
    print(f"  Reference [0, -1] = {ref[0, -1]:.6f}")

    x = jnp.array(x_np)
    y = jnp.array(y_np)
    u = jnp.array(u_np)
    v = jnp.array(v_np)

    # Test 1: LazyTensor batched
    print("\n1. LazyTensor (batched):")
    try:
        x_L = LazyTensor(x[:, :, None, :])
        y_L = LazyTensor(y[:, None, :, :])
        u_L = LazyTensor(u[:, :, None, :])
        v_L = LazyTensor(v[:, None, :, :])

        sq_dist = x_L.sqdist(y_L)
        dot_prod = (u_L * v_L).sum(-1)
        K = (-sq_dist * gamma_s).exp() * (dot_prod * gamma_n).exp()
        result_lazy_batched = K.sum(axis=2)

        if result_lazy_batched.ndim == 3 and result_lazy_batched.shape[-1] == 1:
            result_lazy_batched = result_lazy_batched.squeeze(-1)

        result_lazy_batched_np = np.array(result_lazy_batched)
        diff = np.abs(ref - result_lazy_batched_np).max()
        print(f"   Max diff: {diff:.2e}")
        print(f"   Status: {'✓ PASS' if diff < 1e-4 else '✗ FAIL'}")
        if diff >= 1e-4:
            print(f"   [0, -1]: ref={ref[0, -1]:.6f}, jax={result_lazy_batched_np[0, -1]:.6f}")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")

    # Test 2: LazyTensor unbatched (loop)
    print("\n2. LazyTensor (unbatched loop):")
    try:
        results = []
        for b in range(B):
            x_b, y_b, u_b, v_b = x[b], y[b], u[b], v[b]

            x_L = LazyTensor(x_b[:, None, :])
            y_L = LazyTensor(y_b[None, :, :])
            u_L = LazyTensor(u_b[:, None, :])
            v_L = LazyTensor(v_b[None, :, :])

            sq_dist = x_L.sqdist(y_L)
            dot_prod = (u_L * v_L).sum(-1)
            K = (-sq_dist * gamma_s).exp() * (dot_prod * gamma_n).exp()
            result_b = K.sum(axis=1)

            if result_b.ndim == 2 and result_b.shape[-1] == 1:
                result_b = result_b.squeeze(-1)

            results.append(result_b)

        result_lazy_loop = jnp.stack(results, axis=0)
        result_lazy_loop_np = np.array(result_lazy_loop)
        diff = np.abs(ref - result_lazy_loop_np).max()
        print(f"   Max diff: {diff:.2e}")
        print(f"   Status: {'✓ PASS' if diff < 1e-4 else '✗ FAIL'}")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")

    # Test 3: Genred batched
    print("\n3. Genred (batched):")
    try:
        formula = "Exp(-g * SqDist(x, y)) * Exp(g1 * (u | v))"
        aliases = [
            f"x = Vi({D})", f"y = Vj({D})",
            f"u = Vi({D})", f"v = Vj({D})",
            "g = Pm(1)", "g1 = Pm(1)"
        ]
        genred = Genred(formula, aliases, reduction_op='Sum', axis=1)

        g = jnp.array([gamma_s])
        g1 = jnp.array([gamma_n])
        result_genred = genred(x, y, u, v, g, g1)

        if result_genred.shape[-1] == 1:
            result_genred = result_genred.squeeze(-1)

        result_genred_np = np.array(result_genred)
        diff = np.abs(ref - result_genred_np).max()
        print(f"   Max diff: {diff:.2e}")
        print(f"   Status: {'✓ PASS' if diff < 1e-4 else '✗ FAIL'}")
        if diff >= 1e-4:
            print(f"   [0, -1]: ref={ref[0, -1]:.6f}, jax={result_genred_np[0, -1]:.6f}")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")

    # Test 4: Genred unbatched (loop)
    print("\n4. Genred (unbatched loop):")
    try:
        formula = "Exp(-g * SqDist(x, y)) * Exp(g1 * (u | v))"
        aliases = [
            f"x = Vi({D})", f"y = Vj({D})",
            f"u = Vi({D})", f"v = Vj({D})",
            "g = Pm(1)", "g1 = Pm(1)"
        ]
        genred = Genred(formula, aliases, reduction_op='Sum', axis=1)

        g = jnp.array([gamma_s])
        g1 = jnp.array([gamma_n])
        results = []
        for b in range(B):
            result_b = genred(x[b], y[b], u[b], v[b], g, g1)
            if result_b.shape[-1] == 1:
                result_b = result_b.squeeze(-1)
            results.append(result_b)

        result_genred_loop = jnp.stack(results, axis=0)
        result_genred_loop_np = np.array(result_genred_loop)
        diff = np.abs(ref - result_genred_loop_np).max()
        print(f"   Max diff: {diff:.2e}")
        print(f"   Status: {'✓ PASS' if diff < 1e-4 else '✗ FAIL'}")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")


print("=" * 80)
print("KeOps JAX Comprehensive Kernel Tests")
print("=" * 80)

# Test at different sizes
test_configs = [
    (150, 150, "Small (should pass)"),
    (200, 200, "Medium (at threshold)"),
    (500, 500, "Large (should fail)"),
]

for N, M, desc in test_configs:
    print(f"\n\n{'#' * 80}")
    print(f"# {desc}: N={N}, M={M}")
    print(f"{'#' * 80}")

    test_gaussian_kernel(N, M)
    test_varifold_kernel(N, M)

print("\n\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("Key questions to answer:")
print("1. Does Genred also fail at N>200? (If yes, bug is in C++ backend)")
print("2. Does simple Gaussian fail same as varifold? (If yes, not formula-specific)")
print("3. Do unbatched versions always work? (If yes, confirms workaround)")