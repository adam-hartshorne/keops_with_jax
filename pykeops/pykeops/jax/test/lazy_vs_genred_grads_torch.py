import torch
import numpy as np
from pykeops.torch import Genred, LazyTensor

# Setup Data (Identical to JAX seed and dimensions)
B, N, M, D = 2, 3, 2, 3
np.random.seed(0)
x_np = np.random.randn(B, N, D).astype('float32')
y_np = np.random.randn(B, M, D).astype('float32')

formula = "Sum(SqDist(a, b))"
aliases = [f"a=Vi({D})", f"b=Vj({D})"]


def run_pytorch_full_comparison():
    print("=" * 80)
    print("PYTORCH GROUND TRUTH: GENRED VS LAZYTENSOR")
    print("=" * 80)

    # --- 1. GENRED ---
    xt_g = torch.from_numpy(x_np).cuda().requires_grad_(True)
    yt_g = torch.from_numpy(y_np).cuda()
    op_genred = Genred(formula, aliases, reduction_op='Sum', axis=1)
    res_genred = op_genred(xt_g, yt_g)
    res_genred.backward(torch.ones_like(res_genred))

    # --- 2. LAZYTENSOR ---
    xt_l = torch.from_numpy(x_np).cuda().requires_grad_(True)
    yt_l = torch.from_numpy(y_np).cuda()
    xi, yj = LazyTensor(xt_l[:, :, None, :]), LazyTensor(yt_l[:, None, :, :])
    res_lazy = ((xi - yj) ** 2).sum(-1).sum(dim=2)
    res_lazy.backward(torch.ones_like(res_lazy))

    for b in range(B):
        print(f"\n--- Batch {b} ---")
        print(f"Fwd (Genred):\n{res_genred.detach().cpu().numpy()[b]}")
        print(f"Fwd (Lazy):  \n{res_lazy.detach().cpu().numpy()[b]}")
        print(f"Grad (Genred):\n{xt_g.grad.cpu().numpy()[b]}")
        print(f"Grad (Lazy):  \n{xt_l.grad.cpu().numpy()[b]}")


if __name__ == '__main__':
    run_pytorch_full_comparison()