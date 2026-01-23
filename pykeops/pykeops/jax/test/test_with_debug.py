#!/usr/bin/env python3
"""
Run this after adding debug output to keops_jax.cpp
It will test exactly at the threshold to see what differs.
"""
import os

os.environ['PYKEOPS_JAX_MODE'] = '1'

import numpy as np
import jax.numpy as jnp
from pykeops.jax import Genred


def test_size(N, M, desc):
    """Test a single size with Genred."""
    print("\n" + "=" * 80)
    print(f"{desc}: N={N}, M={M}")
    print("=" * 80)

    rng = np.random.RandomState(42)
    B, D = 2, 3

    x_np = rng.randn(B, N, D).astype(np.float32)
    y_np = rng.randn(B, M, D).astype(np.float32)

    x = jnp.array(x_np)
    y = jnp.array(y_np)

    gamma = 1.0

    print(f"\nInput shapes:")
    print(f"  x: {x.shape}")
    print(f"  y: {y.shape}")

    # Genred batched
    formula = "Exp(-g * SqDist(x, y))"
    aliases = [f"x = Vi({D})", f"y = Vj({D})", "g = Pm(1)"]
    genred = Genred(formula, aliases, reduction_op='Sum', axis=1)

    g = jnp.array([gamma])

    print(f"\nCalling Genred...")
    print(f"Expected to trigger C++ debug output below:")
    print("-" * 80)

    result = genred(x, y, g)

    print("-" * 80)
    print(f"Result shape: {result.shape}")

    # Handle the trailing dimension
    if result.shape[-1] == 1:
        result = result.squeeze(-1)

    result_np = np.array(result)
    print(f"Result[0, 0]: {result_np[0, 0]:.6f}")
    print(f"Result[0, -1]: {result_np[0, -1]:.6f}")


print("=" * 80)
print("KeOps JAX Debug Test")
print("=" * 80)
print("This will show C++ debug output for each size")
print("Compare the values between working (N=150) and broken (N=200) cases")

# Test at threshold
test_size(150, 150, "WORKING SIZE (N=150)")
test_size(200, 200, "BROKEN SIZE (N=200)")

print("\n" + "=" * 80)
print("WHAT TO LOOK FOR:")
print("=" * 80)
print("1. Are nx_kernel and ny_kernel extracted correctly?")
print("2. Is blocks_per_batch different?")
print("3. Is nblocks calculated correctly?")
print("4. Are scratch buffer sizes reasonable?")
print("5. Does ranges_enc.as_int have the right value?")
print("6. Are there any CUDA errors?")