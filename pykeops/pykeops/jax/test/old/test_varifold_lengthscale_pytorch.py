#!/usr/bin/env python3
"""
PyTorch KeOps: Varifold kernel with separate lengthscales for position and normals
Formula: wi * wj * exp(-|x-y|^2 / sigma_pos^2) * exp(-|nx-ny|^2 / sigma_norm^2)
"""
import numpy as np
import torch
from pykeops.torch import LazyTensor

print("="*80)
print("PYTORCH: VARIFOLD KERNEL WITH LENGTHSCALES")
print("="*80)

# Setup
batch_size = 2
n_i = 10
n_j = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create test data (SAME SEED AS JAX)
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
print(f"  device: {device}")

# Convert to PyTorch
x = torch.from_numpy(x_np).to(device).requires_grad_(True)
y = torch.from_numpy(y_np).to(device)
nx = torch.from_numpy(nx_np).to(device)
ny = torch.from_numpy(ny_np).to(device)
wi = torch.from_numpy(wi_np).to(device)
wj = torch.from_numpy(wj_np).to(device)

# Method 1: LazyTensor (symbolic)
print("\n" + "="*80)
print("METHOD 1: LAZYTENSOR")
print("="*80)

x_i = LazyTensor(x.view(batch_size, n_i, 1, 3))
y_j = LazyTensor(y.view(batch_size, 1, n_j, 3))
nx_i = LazyTensor(nx.view(batch_size, n_i, 1, 3))
ny_j = LazyTensor(ny.view(batch_size, 1, n_j, 3))
wi_i = LazyTensor(wi.view(batch_size, n_i, 1, 1))
wj_j = LazyTensor(wj.view(batch_size, 1, n_j, 1))

# Kernel with lengthscales
sq_dist_pos = ((x_i - y_j) ** 2).sum(-1)
sq_dist_norm = ((nx_i - ny_j) ** 2).sum(-1)

K_ij = wi_i * wj_j * (-(sq_dist_pos / (sigma_pos**2))).exp() * (-(sq_dist_norm / (sigma_norm**2))).exp()

result_lazy = K_ij.sum(dim=2)

print(f"\nForward pass:")
print(f"  Result shape: {result_lazy.shape}")
print(f"  Batch 0 sum: {result_lazy[0].sum().item():.10f}")
print(f"  Batch 1 sum: {result_lazy[1].sum().item():.10f}")

# Gradient
loss_lazy = result_lazy.sum()
loss_lazy.backward()
grad_lazy = x.grad.clone()
x.grad.zero_()

print(f"\nGradients:")
print(f"  Batch 0 norm: {grad_lazy[0].norm().item():.6f}")
print(f"  Batch 1 norm: {grad_lazy[1].norm().item():.6f}")
print(f"  Batch 0 first point: {grad_lazy[0, 0, :].cpu().numpy()}")
print(f"  Batch 1 first point: {grad_lazy[1, 0, :].cpu().numpy()}")

# Method 2: Genred (explicit formula)
print("\n" + "="*80)
print("METHOD 2: GENRED")
print("="*80)

from pykeops.torch import Genred

# Explicit formula with lengthscales as parameters
formula = "wi * wj * Exp(-Sum((x-y)*(x-y)) / (sigma_pos*sigma_pos)) * Exp(-Sum((nx-ny)*(nx-ny)) / (sigma_norm*sigma_norm))"
aliases = [
    "x = Vi(3)",
    "y = Vj(3)",
    "nx = Vi(3)",
    "ny = Vj(3)",
    "wi = Vi(1)",
    "wj = Vj(1)",
    "sigma_pos = Pm(1)",   # Parameter (constant)
    "sigma_norm = Pm(1)",  # Parameter (constant)
]

op_genred = Genred(formula, aliases, reduction_op='Sum', axis=1)

# Parameters need to be tensors
sigma_pos_arr = torch.tensor([[sigma_pos]], dtype=torch.float32, device=device)
sigma_norm_arr = torch.tensor([[sigma_norm]], dtype=torch.float32, device=device)

# Reset x.grad
x.grad = None

result_genred = op_genred(x, y, nx, ny, wi, wj, sigma_pos_arr, sigma_norm_arr)

print(f"\nForward pass:")
print(f"  Result shape: {result_genred.shape}")
print(f"  Batch 0 sum: {result_genred[0].sum().item():.10f}")
print(f"  Batch 1 sum: {result_genred[1].sum().item():.10f}")

# Gradient
loss_genred = result_genred.sum()
loss_genred.backward()
grad_genred = x.grad.clone()

print(f"\nGradients:")
print(f"  Batch 0 norm: {grad_genred[0].norm().item():.6f}")
print(f"  Batch 1 norm: {grad_genred[1].norm().item():.6f}")
print(f"  Batch 0 first point: {grad_genred[0, 0, :].cpu().numpy()}")
print(f"  Batch 1 first point: {grad_genred[1, 0, :].cpu().numpy()}")

# Comparison
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

fwd_diff_0 = abs(result_lazy[0].sum().item() - result_genred[0].sum().item())
fwd_diff_1 = abs(result_lazy[1].sum().item() - result_genred[1].sum().item())

grad_diff_0 = (grad_lazy[0] - grad_genred[0]).norm().item()
grad_diff_1 = (grad_lazy[1] - grad_genred[1]).norm().item()

print(f"\nBatch 0:")
print(f"  Forward difference: {fwd_diff_0:.2e}")
print(f"  Gradient difference: {grad_diff_0:.2e}")

print(f"\nBatch 1:")
print(f"  Forward difference: {fwd_diff_1:.2e}")
print(f"  Gradient difference: {grad_diff_1:.2e}")

# Check if both batches have non-zero gradients
lazy_batch1_nonzero = grad_lazy[1].norm().item() > 1e-6
genred_batch1_nonzero = grad_genred[1].norm().item() > 1e-6

if lazy_batch1_nonzero and genred_batch1_nonzero:
    print("\n✅ SUCCESS: Both methods have non-zero gradients for both batches!")
    if fwd_diff_0 < 1e-5 and grad_diff_0 < 1e-5 and fwd_diff_1 < 1e-5 and grad_diff_1 < 1e-5:
        print("🎉 PERFECT: LazyTensor and Genred match exactly!")
    else:
        print("⚠️  Note: Small numerical differences between methods")
else:
    print("\n❌ ERROR: Batch 1 gradients are zero!")
    print(f"   LazyTensor Batch 1 norm: {grad_lazy[1].norm().item()}")
    print(f"   Genred Batch 1 norm: {grad_genred[1].norm().item()}")

print("\n" + "="*80)
print("EXPECTED VALUES FOR JAX COMPARISON")
print("="*80)
print(f"Forward Batch 0: {result_lazy[0].sum().item():.10f}")
print(f"Forward Batch 1: {result_lazy[1].sum().item():.10f}")
print(f"Gradient Batch 0 norm: {grad_lazy[0].norm().item():.6f}")
print(f"Gradient Batch 1 norm: {grad_lazy[1].norm().item():.6f}")