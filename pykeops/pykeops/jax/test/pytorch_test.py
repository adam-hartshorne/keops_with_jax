import torch
import numpy as np
from pykeops.torch import Genred as Genred_torch
from pykeops.numpy import Genred as Genred_numpy

np.set_printoptions(threshold=np.inf, precision=5, suppress=True)

def run_torch_numpy_test():
    N, M, D, B = 10, 5, 3, 2  # D is 3
    np.random.seed(0)
    x_np = np.random.randn(N, D).astype('float32')
    y_np = np.random.randn(M, D).astype('float32')
    xb_np = np.random.randn(B, N, D).astype('float32')
    yb_np = np.random.randn(B, M, D).astype('float32')

    # CORRECTED ALIASES (Vi(3) instead of Vi(2))
    formula = "Sum((a-b)**2)"
    aliases = [f"a=Vi({D})", f"b=Vj({D})"]

    print("=" * 60)
    print("TEST 1: NON-BATCHED (2D)")
    print("=" * 60)

    # NumPy Result
    op_numpy = Genred_numpy(formula, aliases, reduction_op='Sum', axis=1)
    res_numpy_2d = op_numpy(x_np, y_np)

    # PyTorch Result
    op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
    x_t, y_t = torch.from_numpy(x_np).cuda(), torch.from_numpy(y_np).cuda()
    res_torch_2d = op_torch(x_t, y_t).cpu().numpy()

    print(f"PyTorch Result:\n{res_torch_2d.flatten()}")
    print(f"NumPy Result:\n{res_numpy_2d.flatten()}")

    print("\n" + "=" * 60)
    print("TEST 2: BATCHED (3D)")
    print("=" * 60)

    # NumPy Batched
    res_numpy_3d = op_numpy(xb_np, yb_np)

    # PyTorch Batched
    xb_t, yb_t = torch.from_numpy(xb_np).cuda(), torch.from_numpy(yb_np).cuda()
    res_torch_3d = op_torch(xb_t, yb_t).cpu().numpy()

    print(f"PyTorch Result:\n{res_torch_3d}")
    print(f"NumPy Result:\n{res_numpy_3d}")

if __name__ == "__main__":
    run_torch_numpy_test()