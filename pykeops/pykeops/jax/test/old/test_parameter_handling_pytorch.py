#!/usr/bin/env python3
"""
Test parameters with PyTorch KeOps to compare with JAX
"""
import torch
from pykeops.torch import Genred

print("=" * 80)
print("PYTORCH KEOPS: Basic Parameter Tests")
print("=" * 80)

# Use the same test data
x_np = [[1.0, 2.0, 3.0]]
y_np = [[0.0, 0.0, 0.0]]
s_np = [0.5]

x = torch.tensor(x_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)
s = torch.tensor(s_np, dtype=torch.float32)

print(f"x: {x}")
print(f"y: {y}")
print(f"s: {s}")

# Test 1: Vi with parameter, axis=1
print("\n" + "=" * 80)
print("Test 1: x * s (Vi with parameter, axis=1)")
print("=" * 80)

formula1 = "x * s"
aliases1 = ["x=Vi(3)", "s=Pm(1)"]
op1 = Genred(formula1, aliases1, reduction_op='Sum', axis=1, dtype='float32')

try:
    res1 = op1(x, s)
    expected1 = (x * s).sum(dim=0, keepdim=True)

    print(f"PyTorch KeOps result: {res1}")
    print(f"Expected result:      {expected1}")
    print(f"Match: {torch.allclose(res1, expected1)}")

    diff = torch.abs(res1 - expected1)
    rel_err = diff / (torch.abs(expected1) + 1e-10)
    print(f"Max difference: {diff.max().item():.6e}")
    print(f"Max rel error:  {rel_err.max().item():.6e}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()

# Test 2: (x-y) * s with both Vi and Vj
print("\n" + "=" * 80)
print("Test 2: (x-y) * s (Vi and Vj with parameter, axis=1)")
print("=" * 80)

formula2 = "(x - y) * s"
aliases2 = ["x=Vi(3)", "y=Vj(3)", "s=Pm(1)"]
op2 = Genred(formula2, aliases2, reduction_op='Sum', axis=1, dtype='float32')

try:
    res2 = op2(x, y, s)
    expected2 = ((x[:, None, :] - y[None, :, :]) * s).sum(dim=1)

    print(f"PyTorch KeOps result: {res2}")
    print(f"Expected result:      {expected2}")
    print(f"Match: {torch.allclose(res2, expected2)}")

    diff = torch.abs(res2 - expected2)
    rel_err = diff / (torch.abs(expected2) + 1e-10)
    print(f"Max difference: {diff.max().item():.6e}")
    print(f"Max rel error:  {rel_err.max().item():.6e}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()

# Test 3: Exp(-SqDist(x,y) * s) * b (full failing case from JAX)
print("\n" + "=" * 80)
print("Test 3: Exp(-SqDist(x,y) * s) * b (full formula)")
print("=" * 80)

b = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
formula3 = "Exp(-SqDist(x, y) * s) * b"
aliases3 = ["x=Vi(3)", "y=Vj(3)", "b=Vj(2)", "s=Pm(1)"]
op3 = Genred(formula3, aliases3, reduction_op='Sum', axis=1, dtype='float32')

try:
    res3 = op3(x, y, b, s)
    K = torch.exp(-torch.sum((x[:, None, :] - y[None, :, :]) ** 2, dim=-1) * s)
    expected3 = K @ b

    print(f"PyTorch KeOps result: {res3}")
    print(f"Expected result:      {expected3}")
    print(f"Match: {torch.allclose(res3, expected3)}")

    diff = torch.abs(res3 - expected3)
    rel_err = diff / (torch.abs(expected3) + 1e-10)
    print(f"Max difference: {diff.max().item():.6e}")
    print(f"Max rel error:  {rel_err.max().item():.6e}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print("If PyTorch version works but JAX fails:")
print("  → Bug is in JAX implementation (generic_ops.py)")
print("If PyTorch version also fails:")
print("  → Bug is in KeOps C++ core")
print("=" * 80)