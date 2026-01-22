import os

os.environ['PYKEOPS_JAX_MODE'] = '1'

import numpy as np
import jax
import jax.numpy as jnp
from pykeops.jax import LazyTensor


def check_accuracy(jax_res, keops_res, name):
    """Helper to check if two results are numerically close."""
    is_close = jnp.allclose(jax_res, keops_res, atol=1e-4, rtol=1e-4)
    if is_close:
        print(f"✅ Accuracy Check PASSED for {name}")
    else:
        diff = jnp.abs(jax_res - keops_res).max()
        print(f"❌ Accuracy Check FAILED for {name}. Max diff: {diff}")
    return is_close


print("=" * 80)
print("Test 1: Non-batched (2D inputs)")
print("=" * 80)

N, M, D = 100, 80, 3
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (N, D))
y = jax.random.normal(key, (M, D))

# --- Pure JAX Version ---
# Using broadcasting: (N, 1, D) - (1, M, D) -> (N, M, D)
jax_dist = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
jax_result_nonbatch = jax_dist.sum(axis=1)

print(f"Pure JAX result: {jax_result_nonbatch}")

# --- KeOps Version ---
try:
    x_i = LazyTensor(x[:, None, :])
    y_j = LazyTensor(y[None, :, :])
    D_ij = ((x_i - y_j) ** 2).sum(-1)
    keops_result_nonbatch = D_ij.sum(axis=1)

    print(f"KeOps result: {keops_result_nonbatch}")

    check_accuracy(jax_result_nonbatch, keops_result_nonbatch, "Non-batched")
except Exception as e:
    print(f"✗ Non-batched FAILED: {e}")
    keops_result_nonbatch = None

print("\n" + "=" * 80)
print("Test 2: Batched (3D inputs)")
print("=" * 80)

B = 4
x_batch = jax.random.normal(key, (B, N, D))
y_batch = jax.random.normal(key, (B, M, D))

# --- Pure JAX Version ---
# (B, N, 1, D) - (B, 1, M, D) -> (B, N, M, D)
jax_dist_batch = jnp.sum((x_batch[:, :, None, :] - y_batch[:, None, :, :]) ** 2, axis=-1)
jax_result_batch = jax_dist_batch.sum(axis=2)

print(f"Pure JAX result: {jax_result_batch}")

# --- KeOps Version ---
try:
    x_i_b = LazyTensor(x_batch[:, :, None, :])
    y_j_b = LazyTensor(y_batch[:, None, :, :])
    D_ij_b = ((x_i_b - y_j_b) ** 2).sum(-1)
    keops_result_batch = D_ij_b.sum(axis=2)

    print(f"KeOps result: {keops_result_batch}")

    check_accuracy(jax_result_batch, keops_result_batch, "Batched")
except Exception as e:
    print(f"✗ Batched FAILED: {e}")
    keops_result_batch = None

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
# Final verification logic can remain similar to your original script