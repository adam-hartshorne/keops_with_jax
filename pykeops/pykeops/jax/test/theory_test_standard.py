import torch
import numpy as np
import os
from pykeops.torch import Genred as Genred_torch
from pykeops.numpy import Genred as Genred_numpy

# Setup Data
B, N, M, D = 2, 2, 3, 3
x_np = np.zeros((B, N, D), dtype='float32')
x_np[0, :, :] = 1.0   # SqDist = 1+1+1 = 3
x_np[1, :, :] = 10.0  # SqDist = 100+100+100 = 300
y_np = np.zeros((B, M, D), dtype='float32')

# Use the exact same string formula for both
formula = "Sum(SqDist(x, y))"
aliases = [f"x=Vi({D})", f"y=Vj({D})"]

# 1. PyTorch
xt, yt = torch.from_numpy(x_np).cuda(), torch.from_numpy(y_np).cuda()
op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
res_torch = op_torch(xt, yt)

# 2. NumPy
op_numpy = Genred_numpy(formula, aliases, reduction_op='Sum', axis=1)
res_numpy = op_numpy(x_np, y_np)

print("=== PYTORCH (String Formula) ===")
print(res_torch.cpu().numpy())

print("\n=== NUMPY (String Formula) ===")
print(res_numpy)