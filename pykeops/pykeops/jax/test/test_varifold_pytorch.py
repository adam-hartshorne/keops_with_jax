#!/usr/bin/env python3
import numpy as np
import torch
from pykeops.torch import LazyTensor


def main():
    # 1. Setup Parameters
    batch_size = 2
    n_i = 10
    n_j = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Create test data (Reproduce your exact setup)
    np.random.seed(42)
    x_np = np.random.randn(batch_size, n_i, 3).astype(np.float32)
    y_np = np.random.randn(batch_size, n_j, 3).astype(np.float32)
    nx_np = np.random.randn(batch_size, n_i, 3).astype(np.float32)
    ny_np = np.random.randn(batch_size, n_j, 3).astype(np.float32)
    wi_np = np.random.randn(batch_size, n_i, 1).astype(np.float32)
    wj_np = np.random.randn(batch_size, n_j, 1).astype(np.float32)

    nx_np = nx_np / (np.linalg.norm(nx_np, axis=-1, keepdims=True) + 1e-8)
    ny_np = ny_np / (np.linalg.norm(ny_np, axis=-1, keepdims=True) + 1e-8)

    # 3. Proper Tensor Initialization
    # We move to device FIRST, then set requires_grad
    x = torch.from_numpy(x_np).to(device).requires_grad_(True)
    y = torch.from_numpy(y_np).to(device)
    nx = torch.from_numpy(nx_np).to(device)
    ny = torch.from_numpy(ny_np).to(device)
    wi = torch.from_numpy(wi_np).to(device)
    wj = torch.from_numpy(wj_np).to(device)

    # 4. Symbolic Formulation
    # We use 4D tensors: (Batch, N, M, Dim)
    # x_i is (B, N, 1, 3), y_j is (B, 1, M, 3)
    x_i = LazyTensor(x.view(batch_size, n_i, 1, 3))
    y_j = LazyTensor(y.view(batch_size, 1, n_j, 3))

    nx_i = LazyTensor(nx.view(batch_size, n_i, 1, 3))
    ny_j = LazyTensor(ny.view(batch_size, 1, n_j, 3))

    wi_i = LazyTensor(wi.view(batch_size, n_i, 1, 1))
    wj_j = LazyTensor(wj.view(batch_size, 1, n_j, 1))

    # Computation
    sq_dist = ((x_i - y_j) ** 2).sum(-1)
    dot_prod = (nx_i | ny_j)

    # Kernel: w_i * w_j * exp(-|x-y|^2) * (nx . ny)^2
    K_ij = wi_i * wj_j * (-sq_dist).exp() * dot_prod.square()

    # Reduction over 'j' (dimension 2)
    # Result shape: (Batch, N, 1)
    result_torch = K_ij.sum(dim=2)

    # 5. Verification
    loss = result_torch.sum()
    loss.backward()

    print(f"Batch 0 Grad Norm: {x.grad[0].norm().item():.6f}")
    print(f"Batch 1 Grad Norm: {x.grad[1].norm().item():.6f}")

    if x.grad[1].norm() == 0:
        print("\n[!] Error still detected: Batch 1 gradient is zero.")
    else:
        print("\n[✓] Success: Both batches have non-zero gradients.")


if __name__ == "__main__":
    main()