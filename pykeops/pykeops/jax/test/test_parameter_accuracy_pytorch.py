#!/usr/bin/env python3
"""
Save PyTorch KeOps results to file for comparison
"""
import torch
import numpy as np
from pykeops.torch import Genred as GenredTorch
import pickle

print("="*80)
print("PYTORCH KEOPS: Saving results to file")
print("="*80)

# Use SAME random seed as JAX
np.random.seed(42)
N, M, D, E = 100, 50, 3, 2

# Generate data
x_np = np.random.randn(N, D).astype(np.float32)
y_np = np.random.randn(M, D).astype(np.float32)
b_np = np.random.randn(M, E).astype(np.float32)
s_val = 0.5

# Convert to torch
x_torch = torch.from_numpy(x_np)
y_torch = torch.from_numpy(y_np)
b_torch = torch.from_numpy(b_np)
s_torch = torch.tensor([s_val], dtype=torch.float32)

print(f"\nData shapes: x={x_torch.shape}, y={y_torch.shape}, b={b_torch.shape}")
print(f"Parameter s: {s_val}")

# Compute NumPy expected (ground truth)
print("\nComputing NumPy expected value...")
K_np = np.exp(-np.sum((x_np[:, None, :] - y_np[None, :, :]) ** 2, axis=-1) * s_val)
expected_np = K_np @ b_np

# PyTorch KeOps with parameters
print("Computing PyTorch KeOps result...")
formula = "Exp(-SqDist(x, y) * s) * b"
aliases = ["x=Vi(3)", "y=Vj(3)", "b=Vj(2)", "s=Pm(1)"]
op = GenredTorch(formula, aliases, reduction_op='Sum', axis=1, dtype='float32')
res_torch = op(x_torch, y_torch, b_torch, s_torch).cpu().numpy()

# Save results
output = {
    'platform': 'pytorch',
    'x': x_np,
    'y': y_np,
    'b': b_np,
    's': s_val,
    'numpy_expected': expected_np,
    'keops_result': res_torch,
    'formula': formula,
    'N': N,
    'M': M,
    'D': D,
    'E': E
}

filename = 'pytorch_results.pkl'
with open(filename, 'wb') as f:
    pickle.dump(output, f)

print(f"\nSaved results to {filename}")
print(f"  NumPy expected shape: {expected_np.shape}")
print(f"  PyTorch KeOps shape:  {res_torch.shape}")
print(f"  Expected[0]: {expected_np[0]}")
print(f"  Result[0]:   {res_torch[0]}")

# Quick verification
diff = np.abs(res_torch - expected_np)
rel = diff / (np.abs(expected_np) + 1e-10)
print(f"\nPyTorch vs NumPy accuracy:")
print(f"  Max absolute error: {diff.max():.6e}")
print(f"  Max relative error: {rel.max():.6e}")

print("="*80)