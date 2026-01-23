#!/usr/bin/env python3
"""
JAX KeOps: Varifold kernel with separate lengthscales for position and normals
Formula: wi * wj * exp(-|x-y|^2 / sigma_pos^2) * exp(-|nx-ny|^2 / sigma_norm^2)
"""
import os

os.environ['PYKEOPS_JAX_MODE'] = '0'

import numpy as np
import jax
import jax.numpy as jnp
from pykeops.jax import LazyTensor

print("=" * 80)
print("VARIFOLD KERNEL WITH LENGTHSCALES")
print("=" * 80)

# Setup
batch_size = 2
n_i = 10
n_j = 8

# Create test data
np.random.seed(42)
x_np = np.random.randn(batch_size, n_i, 3).astype(np.float32)
y_np = np.random.randn(batch_size, n_j, 3).astype(np.float32)
nx_np = np.random.randn(batch_size, n_i, 3).astype(np.float32)
ny_np = np.random.randn(batch_size, n_j, 3).astype(np.float32)
wi_np = np.random.randn(batch_size, n_i, 1).astype(np.float32)
wj_np = np.random.randn(batch_size, n_j, 1).astype(np.float32)

# Normalize normals
nx_np = nx_np / (np.linalg.norm(nx_np, axis=-1, keepdims=True) + 1e-8)
ny_np = ny_np / (np.linalg.norm(ny_np, axis=-1, keepdims=True) + 1e-8)

# Lengthscale parameters (scalar, same for all batches)
sigma_pos = 1.5
sigma_norm = 0.8

print(f"\nConfiguration:")
print(f"  batch_size: {batch_size}, n_i: {n_i}, n_j: {n_j}")
print(f"  sigma_pos:  {sigma_pos}")
print(f"  sigma_norm: {sigma_norm}")

# Convert to JAX
x = jnp.array(x_np)
y = jnp.array(y_np)
nx = jnp.array(nx_np)
ny = jnp.array(ny_np)
wi = jnp.array(wi_np)
wj = jnp.array(wj_np)

# Method 1: LazyTensor (symbolic)
print("\n" + "=" * 80)
print("METHOD 1: LAZYTENSOR")
print("=" * 80)

x_i = LazyTensor(x.reshape(batch_size, n_i, 1, 3))
y_j = LazyTensor(y.reshape(batch_size, 1, n_j, 3))
nx_i = LazyTensor(nx.reshape(batch_size, n_i, 1, 3))
ny_j = LazyTensor(ny.reshape(batch_size, 1, n_j, 3))
wi_i = LazyTensor(wi.reshape(batch_size, n_i, 1, 1))
wj_j = LazyTensor(wj.reshape(batch_size, 1, n_j, 1))

# Kernel with lengthscales
sq_dist_pos = ((x_i - y_j) ** 2).sum(-1)
sq_dist_norm = ((nx_i - ny_j) ** 2).sum(-1)

K_ij = wi_i * wj_j * (-(sq_dist_pos / (sigma_pos ** 2))).exp() * (-(sq_dist_norm / (sigma_norm ** 2))).exp()

result_lazy = K_ij.sum(dim=2)

print(f"\nForward pass:")
print(f"  Result shape: {result_lazy.shape}")
print(f"  Batch 0 sum: {jnp.sum(result_lazy[0]):.10f}")
print(f"  Batch 1 sum: {jnp.sum(result_lazy[1]):.10f}")


# Gradient
def forward_lazy(x_var):
    x_i = LazyTensor(x_var.reshape(batch_size, n_i, 1, 3))
    y_j = LazyTensor(y.reshape(batch_size, 1, n_j, 3))
    nx_i = LazyTensor(nx.reshape(batch_size, n_i, 1, 3))
    ny_j = LazyTensor(ny.reshape(batch_size, 1, n_j, 3))
    wi_i = LazyTensor(wi.reshape(batch_size, n_i, 1, 1))
    wj_j = LazyTensor(wj.reshape(batch_size, 1, n_j, 1))

    sq_dist_pos = ((x_i - y_j) ** 2).sum(-1)
    sq_dist_norm = ((nx_i - ny_j) ** 2).sum(-1)
    K_ij = wi_i * wj_j * (-(sq_dist_pos / (sigma_pos ** 2))).exp() * (-(sq_dist_norm / (sigma_norm ** 2))).exp()
    return K_ij.sum(dim=2).sum()


grad_lazy = jax.grad(forward_lazy)(x)

print(f"\nGradients:")
print(f"  Batch 0 norm: {jnp.linalg.norm(grad_lazy[0]):.6f}")
print(f"  Batch 1 norm: {jnp.linalg.norm(grad_lazy[1]):.6f}")

# Method 2: Genred (explicit formula)
print("\n" + "=" * 80)
print("METHOD 2: GENRED")
print("=" * 80)

from pykeops.jax import Genred

# Explicit formula with lengthscales as parameters
# We'll pass sigma_pos and sigma_norm as Pm (parameter) variables
formula = "wi * wj * Exp(-Sum((x-y)*(x-y)) / (sigma_pos*sigma_pos)) * Exp(-Sum((nx-ny)*(nx-ny)) / (sigma_norm*sigma_norm))"
aliases = [
    "x = Vi(3)",
    "y = Vj(3)",
    "nx = Vi(3)",
    "ny = Vj(3)",
    "wi = Vi(1)",
    "wj = Vj(1)",
    "sigma_pos = Pm(1)",  # Parameter (constant)
    "sigma_norm = Pm(1)",  # Parameter (constant)
]

op_genred = Genred(formula, aliases, reduction_op='Sum', axis=1)

# Parameters need to be arrays
sigma_pos_arr = jnp.array([[sigma_pos]], dtype=jnp.float32)  # Shape (1, 1)
sigma_norm_arr = jnp.array([[sigma_norm]], dtype=jnp.float32)

result_genred = op_genred(x, y, nx, ny, wi, wj, sigma_pos_arr, sigma_norm_arr)

print(f"\nForward pass:")
print(f"  Result shape: {result_genred.shape}")
print(f"  Batch 0 sum: {jnp.sum(result_genred[0]):.10f}")
print(f"  Batch 1 sum: {jnp.sum(result_genred[1]):.10f}")


# Gradient
def forward_genred(x_var):
    return jnp.sum(op_genred(x_var, y, nx, ny, wi, wj, sigma_pos_arr, sigma_norm_arr))


grad_genred = jax.grad(forward_genred)(x)

print(f"\nGradients:")
print(f"  Batch 0 norm: {jnp.linalg.norm(grad_genred[0]):.6f}")
print(f"  Batch 1 norm: {jnp.linalg.norm(grad_genred[1]):.6f}")

# Comparison
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

fwd_diff = jnp.abs(jnp.sum(result_lazy[0]) - jnp.sum(result_genred[0]))
grad_diff = jnp.linalg.norm(grad_lazy[0] - grad_genred[0])

print(f"\nBatch 0:")
print(f"  Forward difference: {fwd_diff:.2e}")
print(f"  Gradient difference: {grad_diff:.2e}")

fwd_diff_1 = jnp.abs(jnp.sum(result_lazy[1]) - jnp.sum(result_genred[1]))
grad_diff_1 = jnp.linalg.norm(grad_lazy[1] - grad_genred[1])

print(f"\nBatch 1:")
print(f"  Forward difference: {fwd_diff_1:.2e}")
print(f"  Gradient difference: {grad_diff_1:.2e}")

# Check if both batches have non-zero gradients
lazy_batch1_nonzero = jnp.linalg.norm(grad_lazy[1]) > 1e-6
genred_batch1_nonzero = jnp.linalg.norm(grad_genred[1]) > 1e-6

if lazy_batch1_nonzero and genred_batch1_nonzero:
    print("\n✅ SUCCESS: Both methods have non-zero gradients for both batches!")
    if fwd_diff < 1e-5 and grad_diff < 1e-5 and fwd_diff_1 < 1e-5 and grad_diff_1 < 1e-5:
        print("🎉 PERFECT: LazyTensor and Genred match exactly!")
    else:
        print("⚠️  Note: Small numerical differences between methods")
else:
    print("\n❌ ERROR: Batch 1 gradients are zero!")
    print(f"   LazyTensor Batch 1 norm: {jnp.linalg.norm(grad_lazy[1])}")
    print(f"   Genred Batch 1 norm: {jnp.linalg.norm(grad_genred[1])}")

# Output for comparison with PyTorch
print("\n" + "=" * 80)
print("JAX VALUES (COMPARE WITH PYTORCH)")
print("=" * 80)
print(f"Forward Batch 0: {jnp.sum(result_lazy[0]):.10f}")
print(f"Forward Batch 1: {jnp.sum(result_lazy[1]):.10f}")
print(f"Gradient Batch 0 norm: {jnp.linalg.norm(grad_lazy[0]):.6f}")
print(f"Gradient Batch 1 norm: {jnp.linalg.norm(grad_lazy[1]):.6f}")
print(f"Gradient Batch 0 first point: {grad_lazy[0, 0, :]}")
print(f"Gradient Batch 1 first point: {grad_lazy[1, 0, :]}")

print("\n" + "=" * 80)
print("EXPECTED FROM PYTORCH")
print("=" * 80)
print(f"Forward Batch 0: 0.9765453339")
print(f"Forward Batch 1: 1.1674156189")
print(f"Gradient Batch 0 norm: 0.497996")
print(f"Gradient Batch 1 norm: 0.895036")
print(f"Gradient Batch 0 first point: [-0.08702002  0.0798163  -0.14790986]")
print(f"Gradient Batch 1 first point: [0.00809133 0.01497076 0.00472245]")

# Final verification
print("\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)
pytorch_fwd_0 = 0.9765453339
pytorch_fwd_1 = 1.1674156189
pytorch_grad_0 = 0.497996
pytorch_grad_1 = 0.895036

jax_fwd_0 = float(jnp.sum(result_lazy[0]))
jax_fwd_1 = float(jnp.sum(result_lazy[1]))
jax_grad_0 = float(jnp.linalg.norm(grad_lazy[0]))
jax_grad_1 = float(jnp.linalg.norm(grad_lazy[1]))

fwd_match_0 = abs(jax_fwd_0 - pytorch_fwd_0) / pytorch_fwd_0 < 1e-5
fwd_match_1 = abs(jax_fwd_1 - pytorch_fwd_1) / pytorch_fwd_1 < 1e-5
grad_match_0 = abs(jax_grad_0 - pytorch_grad_0) / pytorch_grad_0 < 1e-3
grad_match_1 = abs(jax_grad_1 - pytorch_grad_1) / pytorch_grad_1 < 1e-3

all_match = fwd_match_0 and fwd_match_1 and grad_match_0 and grad_match_1

if all_match:
    print("🎉🎉🎉 PERFECT MATCH WITH PYTORCH! 🎉🎉🎉")
    print("JAX batched multi-variable kernels are working correctly!")
else:
    print("⚠️  Differences detected:")
    if not fwd_match_0: print(f"   Forward Batch 0: {abs(jax_fwd_0 - pytorch_fwd_0) / pytorch_fwd_0 * 100:.3f}% diff")
    if not fwd_match_1: print(f"   Forward Batch 1: {abs(jax_fwd_1 - pytorch_fwd_1) / pytorch_fwd_1 * 100:.3f}% diff")
    if not grad_match_0: print(
        f"   Gradient Batch 0: {abs(jax_grad_0 - pytorch_grad_0) / pytorch_grad_0 * 100:.3f}% diff")
    if not grad_match_1: print(
        f"   Gradient Batch 1: {abs(jax_grad_1 - pytorch_grad_1) / pytorch_grad_1 * 100:.3f}% diff")