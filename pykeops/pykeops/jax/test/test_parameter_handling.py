#!/usr/bin/env python3
"""
FIXED: Test basic kernels - passes correct number of arguments
"""
import os
os.environ['PYKEOPS_JAX_MODE'] = '1'

import jax
import jax.numpy as jnp
from pykeops.jax import Genred

print("=" * 80)
print("BASIC KERNEL TESTS WITH PARAMETERS (FIXED)")
print("=" * 80)

D = 3


def test_kernel(name, formula, aliases, compute_expected, uses_y=True):
    """Test a kernel at different sizes"""
    print(f"\n{'=' * 80}")
    print(f"Test: {name}")
    print(f"Formula: {formula}")
    print(f"{'=' * 80}")

    op = Genred(formula, aliases, 'Sum', axis=1)
    sizes = [(5, 5), (10, 10), (20, 20), (50, 50), (100, 50)]

    for N, M in sizes:
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(jax.random.split(key, 3)[0], (N, D), dtype=jnp.float32)
        y = jax.random.normal(jax.random.split(key, 3)[1], (M, D), dtype=jnp.float32)
        s = jnp.array([0.5], dtype=jnp.float32)

        # Pass correct arguments
        if uses_y:
            res = op(x, y, s)
        else:
            res = op(x, s)  # Don't pass y if not in formula!

        expected = compute_expected(x, y, s[0])

        diff = jnp.abs(res - expected)
        rel_err = diff / (jnp.abs(expected) + 1e-10)

        rtol_5 = jnp.allclose(res, expected, rtol=1e-5)
        rtol_3 = jnp.allclose(res, expected, rtol=1e-3)

        status = '✓' if rtol_5 else ('~' if rtol_3 else '✗')
        print(f"  N={N:3d}, M={M:2d}: max_err={jnp.max(diff):.2e}, max_rel={jnp.max(rel_err):.2e} {status}")


# Test 1: Vi only - DON'T pass y!
print("\n" + "-" * 80)
print("TEST 1: x * s (Vi only, no y)")
print("-" * 80)

test_kernel(
    "Multiply by parameter",
    "x * s",
    [f"x=Vi({D})", "s=Pm(1)"],
    lambda x, y, s: x * s,
    uses_y=False  # KEY FIX!
)

# Test 2: Vi only - DON'T pass y!
print("\n" + "-" * 80)
print("TEST 2: x * x * s (Vi only, no y)")
print("-" * 80)

test_kernel(
    "Square times parameter",
    "(x * x) * s",
    [f"x=Vi({D})", "s=Pm(1)"],
    lambda x, y, s: (x * x) * s,
    uses_y=False  # KEY FIX!
)

# Test 3: Vi only - DON'T pass y!
print("\n" + "-" * 80)
print("TEST 3: SqNorm2(x) * s (Vi only, no y)")
print("-" * 80)

test_kernel(
    "SqNorm times parameter",
    "SqNorm2(x) * s",
    [f"x=Vi({D})", "s=Pm(1)"],
    lambda x, y, s: jnp.sum(x ** 2, axis=-1, keepdims=True) * s,
    uses_y=False  # KEY FIX!
)

# Test 4: Has both Vi and Vj - pass y
print("\n" + "-" * 80)
print("TEST 4: (x - y) * s (has both Vi and Vj)")
print("-" * 80)

test_kernel(
    "Difference times parameter",
    "(x - y) * s",
    [f"x=Vi({D})", f"y=Vj({D})", "s=Pm(1)"],
    lambda x, y, s: jnp.sum((x[:, None, :] - y[None, :, :]) * s, axis=1),
    uses_y=True
)

# Test 5: Has both Vi and Vj - pass y
print("\n" + "-" * 80)
print("TEST 5: SqDist(x, y) * s")
print("-" * 80)

test_kernel(
    "SqDist times parameter",
    "SqDist(x, y) * s",
    [f"x=Vi({D})", f"y=Vj({D})", "s=Pm(1)"],
    lambda x, y, s: jnp.sum(jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1, keepdims=True) * s, axis=1),
    uses_y=True
)

# Test 6: Has both Vi and Vj - pass y
print("\n" + "-" * 80)
print("TEST 6: Exp(-SqDist(x, y) * s)")
print("-" * 80)

test_kernel(
    "Exp of SqDist times parameter",
    "Exp(-SqDist(x, y) * s)",
    [f"x=Vi({D})", f"y=Vj({D})", "s=Pm(1)"],
    lambda x, y, s: jnp.sum(jnp.exp(-jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1) * s), axis=1,
                            keepdims=True),
    uses_y=True
)

print("\n" + "=" * 80)
print("ALL TESTS SHOULD NOW PASS!")
print("=" * 80)