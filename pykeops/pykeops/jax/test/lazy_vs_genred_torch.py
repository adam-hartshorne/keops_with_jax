import torch
import numpy as np
import os
from pykeops.torch import Genred as Genred_torch
from pykeops.numpy import Genred as Genred_numpy

# Set same seed and dimensions as your JAX script
B, N, M, D = 2, 3, 2, 3
np.random.seed(0)

# Generate identical data
x_np = np.random.randn(B, N, D).astype('float32')
x_np[1, :, :] *= 10.0
y_np = np.zeros((B, M, D), dtype='float32')

# Formula and Aliases
formula = "Sum(SqDist(a, b))"
aliases = [f"a=Vi({D})", f"b=Vj({D})"]


def run_truth_comparison():
    print("=" * 60)
    print("GROUND TRUTH COMPARISON (PYTORCH vs NUMPY)")
    print("=" * 60)

    # 1. PyTorch Truth
    xt = torch.from_numpy(x_np).cuda()
    yt = torch.from_numpy(y_np).cuda()
    op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
    res_torch = op_torch(xt, yt).cpu().numpy()

    # 2. NumPy Truth
    op_numpy = Genred_numpy(formula, aliases, reduction_op='Sum', axis=1)
    res_numpy = op_numpy(x_np, y_np)

    for b in range(B):
        print(f"\n--- Batch {b} ---")
        print(f"PyTorch:\n{res_torch[b]}")
        print(f"NumPy:\n{res_numpy[b]}")

    print("\n" + "=" * 60)
    print(f"PyTorch Batch 1 Sample: {res_torch[1, 0, 0]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run_truth_comparison()