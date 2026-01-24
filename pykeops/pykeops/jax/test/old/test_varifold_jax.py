#!/usr/bin/env python3
"""
JAX KeOps LazyTensor Test - Exact match to PyTorch version
"""
import os

import numpy as np
import jax
import jax.numpy as jnp
from pykeops.jax import LazyTensor


def main():
    # 1. Setup Parameters
    batch_size = 2
    n_i = 10
    n_j = 8

    # 2. Create test data (Reproduce exact PyTorch setup)
    np.random.seed(42)
    x_np = np.random.randn(batch_size, n_i, 3).astype(np.float32)
    y_np = np.random.randn(batch_size, n_j, 3).astype(np.float32)
    nx_np = np.random.randn(batch_size, n_i, 3).astype(np.float32)
    ny_np = np.random.randn(batch_size, n_j, 3).astype(np.float32)
    wi_np = np.random.randn(batch_size, n_i, 1).astype(np.float32)
    wj_np = np.random.randn(batch_size, n_j, 1).astype(np.float32)

    nx_np = nx_np / (np.linalg.norm(nx_np, axis=-1, keepdims=True) + 1e-8)
    ny_np = ny_np / (np.linalg.norm(ny_np, axis=-1, keepdims=True) + 1e-8)

    # 3. Convert to JAX arrays
    x = jnp.array(x_np)
    y = jnp.array(y_np)
    nx = jnp.array(nx_np)
    ny = jnp.array(ny_np)
    wi = jnp.array(wi_np)
    wj = jnp.array(wj_np)

    # 4. Symbolic Formulation using LazyTensor
    # Same 4D tensor shapes: (Batch, N, M, Dim)
    x_i = LazyTensor(x.reshape(batch_size, n_i, 1, 3))
    y_j = LazyTensor(y.reshape(batch_size, 1, n_j, 3))

    nx_i = LazyTensor(nx.reshape(batch_size, n_i, 1, 3))
    ny_j = LazyTensor(ny.reshape(batch_size, 1, n_j, 3))

    wi_i = LazyTensor(wi.reshape(batch_size, n_i, 1, 1))
    wj_j = LazyTensor(wj.reshape(batch_size, 1, n_j, 1))

    # Computation - exact same as PyTorch
    sq_dist = ((x_i - y_j) ** 2).sum(-1)
    dot_prod = (nx_i | ny_j)

    # Kernel: w_i * w_j * exp(-|x-y|^2) * (nx . ny)^2
    K_ij = wi_i * wj_j * (-sq_dist).exp() * dot_prod.square()

    # Reduction over 'j' (dimension 2)
    # Result shape: (Batch, N, 1)
    result_jax = K_ij.sum(dim=2)

    print("=" * 80)
    print("JAX KEOPS LAZYTENSOR: VARIFOLD KERNEL TEST")
    print("=" * 80)
    print(f"\nBatch size: {batch_size}, n_i: {n_i}, n_j: {n_j}")
    print(f"Result shape: {result_jax.shape}")

    # 5. Forward pass results
    print("\n" + "=" * 80)
    print("FORWARD PASS")
    print("=" * 80)
    print(f"\nBatch 0:")
    print(f"  First 3 values: {result_jax[0, :3, 0]}")
    print(f"  Mean: {jnp.mean(result_jax[0]):.10f}")
    print(f"  Sum:  {jnp.sum(result_jax[0]):.10f}")

    print(f"\nBatch 1:")
    print(f"  First 3 values: {result_jax[1, :3, 0]}")
    print(f"  Mean: {jnp.mean(result_jax[1]):.10f}")
    print(f"  Sum:  {jnp.sum(result_jax[1]):.10f}")

    # 6. Gradient computation
    print("\n" + "=" * 80)
    print("GRADIENT w.r.t. x")
    print("=" * 80)

    def forward_fn(x_var):
        x_i = LazyTensor(x_var.reshape(batch_size, n_i, 1, 3))
        y_j = LazyTensor(y.reshape(batch_size, 1, n_j, 3))
        nx_i = LazyTensor(nx.reshape(batch_size, n_i, 1, 3))
        ny_j = LazyTensor(ny.reshape(batch_size, 1, n_j, 3))
        wi_i = LazyTensor(wi.reshape(batch_size, n_i, 1, 1))
        wj_j = LazyTensor(wj.reshape(batch_size, 1, n_j, 1))

        sq_dist = ((x_i - y_j) ** 2).sum(-1)
        dot_prod = (nx_i | ny_j)
        K_ij = wi_i * wj_j * (-sq_dist).exp() * dot_prod.square()
        return K_ij.sum(dim=2).sum()

    grad_x = jax.grad(forward_fn)(x)

    print(f"\nBatch 0:")
    print(f"  First point grad: {grad_x[0, 0, :]}")
    print(f"  Grad norm: {jnp.linalg.norm(grad_x[0]):.6f}")

    print(f"\nBatch 1:")
    print(f"  First point grad: {grad_x[1, 0, :]}")
    print(f"  Grad norm: {jnp.linalg.norm(grad_x[1]):.6f}")

    # 7. Verification
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    batch0_norm = jnp.linalg.norm(grad_x[0]).item()
    batch1_norm = jnp.linalg.norm(grad_x[1]).item()

    print(f"Batch 0 Grad Norm: {batch0_norm:.6f}")
    print(f"Batch 1 Grad Norm: {batch1_norm:.6f}")

    if batch1_norm == 0:
        print("\n❌ Error: Batch 1 gradient is zero!")
        return False
    else:
        print("\n✅ Success: Both batches have non-zero gradients!")

        # Compare with expected PyTorch values
        print(f"\nExpected (from PyTorch):")
        print(f"  Batch 0 norm: ~1.174244")
        print(f"  Batch 1 norm: ~0.704466")

        # Check if close enough (within 1%)
        expected_0 = 1.174244
        expected_1 = 0.704466

        diff_0 = abs(batch0_norm - expected_0) / expected_0
        diff_1 = abs(batch1_norm - expected_1) / expected_1

        if diff_0 < 0.01 and diff_1 < 0.01:
            print(f"\n🎉 PERFECT MATCH! Gradients within 1% of PyTorch!")
            return True
        else:
            print(f"\n⚠️  Values differ from PyTorch:")
            print(f"   Batch 0: {diff_0 * 100:.2f}% difference")
            print(f"   Batch 1: {diff_1 * 100:.2f}% difference")
            return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)