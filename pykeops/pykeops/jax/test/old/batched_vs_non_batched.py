#!/usr/bin/env python3
"""
Test batched vs non-batched LazyTensor operations.
Note: KeOps reductions keep a trailing dimension of 1 (standard behavior).
"""
import numpy as np
import jax
import jax.numpy as jnp
from pykeops.jax import LazyTensor


def check_accuracy(jax_res, keops_res, name):
    """Helper to check if two results are numerically close.

    Note: KeOps keeps trailing dim=1 after reductions, so we squeeze it.
    """
    # Squeeze KeOps result to match JAX shape
    keops_squeezed = jnp.squeeze(keops_res)

    is_close = jnp.allclose(jax_res, keops_squeezed, atol=1e-4, rtol=1e-4)
    if is_close:
        print(f"✅ Accuracy Check PASSED for {name}")
        print(f"   JAX shape: {jax_res.shape}, KeOps shape: {keops_res.shape} -> squeezed: {keops_squeezed.shape}")
    else:
        diff = jnp.abs(jax_res - keops_squeezed).max()
        print(f"❌ Accuracy Check FAILED for {name}. Max diff: {diff}")
        print(f"   JAX shape: {jax_res.shape}, KeOps shape: {keops_res.shape}")
    return is_close


print("=" * 80)
print("Test 1: Non-batched (2D inputs)")
print("=" * 80)

N, M, D = 100, 80, 3
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (N, D))
y = jax.random.normal(key, (M, D))

# --- Pure JAX Version ---
jax_dist = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
jax_result_nonbatch = jax_dist.sum(axis=1)

print(f"Pure JAX result shape: {jax_result_nonbatch.shape}")
print(f"Pure JAX result[:5]: {jax_result_nonbatch[:5]}")

# --- KeOps Version ---
try:
    x_i = LazyTensor(x[:, None, :])
    y_j = LazyTensor(y[None, :, :])
    D_ij = ((x_i - y_j) ** 2).sum(-1)
    keops_result_nonbatch = D_ij.sum(axis=1)

    print(f"KeOps result shape: {keops_result_nonbatch.shape}")
    print(f"KeOps result[:5]: {jnp.squeeze(keops_result_nonbatch)[:5]}")

    check_accuracy(jax_result_nonbatch, keops_result_nonbatch, "Non-batched")
except Exception as e:
    print(f"✗ Non-batched FAILED: {e}")
    import traceback

    traceback.print_exc()
    keops_result_nonbatch = None

print("\n" + "=" * 80)
print("Test 2: Batched (3D inputs)")
print("=" * 80)

B = 4
x_batch = jax.random.normal(key, (B, N, D))
y_batch = jax.random.normal(key, (B, M, D))

# --- Pure JAX Version ---
jax_dist_batch = jnp.sum((x_batch[:, :, None, :] - y_batch[:, None, :, :]) ** 2, axis=-1)
jax_result_batch = jax_dist_batch.sum(axis=2)

print(f"Pure JAX result shape: {jax_result_batch.shape}")
print(f"Pure JAX result[0, :5]: {jax_result_batch[0, :5]}")

# --- KeOps Version ---
try:
    x_i_b = LazyTensor(x_batch[:, :, None, :])
    y_j_b = LazyTensor(y_batch[:, None, :, :])
    D_ij_b = ((x_i_b - y_j_b) ** 2).sum(-1)
    keops_result_batch = D_ij_b.sum(axis=2)

    print(f"KeOps result shape: {keops_result_batch.shape}")
    print(f"KeOps result[0, :5]: {jnp.squeeze(keops_result_batch)[0, :5]}")

    check_accuracy(jax_result_batch, keops_result_batch, "Batched")
except Exception as e:
    print(f"✗ Batched FAILED: {e}")
    import traceback

    traceback.print_exc()
    keops_result_batch = None

print("\n" + "=" * 80)
print("Test 3: Gradients")
print("=" * 80)


# Test gradients work correctly
def jax_forward(x_var):
    dist = jnp.sum((x_var[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    return dist.sum()


def keops_forward(x_var):
    x_i = LazyTensor(x_var[:, None, :])
    y_j = LazyTensor(y[None, :, :])
    D_ij = ((x_i - y_j) ** 2).sum(-1)
    return D_ij.sum(axis=1).sum()


try:
    jax_grad = jax.grad(jax_forward)(x)
    keops_grad = jax.grad(keops_forward)(x)

    print(f"JAX gradient shape: {jax_grad.shape}")
    print(f"KeOps gradient shape: {keops_grad.shape}")

    grad_close = jnp.allclose(jax_grad, keops_grad, atol=1e-4, rtol=1e-4)
    if grad_close:
        print(f"✅ Gradient Check PASSED")
    else:
        diff = jnp.abs(jax_grad - keops_grad).max()
        print(f"❌ Gradient Check FAILED. Max diff: {diff}")
except Exception as e:
    print(f"✗ Gradient test FAILED: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("Note: KeOps reductions keep a trailing dimension of 1.")
print("This is standard KeOps behavior (same in PyTorch).")
print("Use jnp.squeeze() to remove the trailing dimension if needed.")