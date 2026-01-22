"""
Test if scalar * vector multiplication works in KeOps PyTorch
"""
import torch
from pykeops.torch import Genred

# Simple test: just return vector b
x = torch.ones(10, 3)
y = torch.zeros(5, 3)
b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])

print(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")
print(f"b shape: {b.shape}")
print()

# Test 1: Just b (should work)
formula1 = "b"
aliases1 = ["x=Vi(3)", "y=Vj(3)", "b=Vj(2)"]
op1 = Genred(formula1, aliases1, reduction_op='Sum', axis=1)

result1 = op1(x, y, b)
print(f"Test 1: formula='b'")
print(f"Result shape: {result1.shape} (expect torch.Size([10, 2]))")
print(f"Result: {result1[:3]}")
print()

# Ground truth
expected = b.sum(dim=0, keepdim=True).repeat(10, 1)
print(f"Expected: {expected[:3]}")
print(f"Match: {torch.allclose(result1, expected)}")
print()

# Test 2: SqDist(x,y) * b  (scalar * vector)
formula2 = "SqDist(x, y) * b"
aliases2 = ["x=Vi(3)", "y=Vj(3)", "b=Vj(2)"]
op2 = Genred(formula2, aliases2, reduction_op='Sum', axis=1)

result2 = op2(x, y, b)
print(f"Test 2: formula='SqDist(x,y) * b'")
print(f"Result shape: {result2.shape} (expect torch.Size([10, 2]))")
print(f"Result: {result2[:3]}")

# Ground truth
sq_dist = ((x[:, None, :] - y[None, :, :])**2).sum(dim=-1)  # (10, 5)
expected2 = (sq_dist[:, :, None] * b[None, :, :]).sum(dim=1)  # (10, 2)
print(f"Expected: {expected2[:3]}")
print(f"Match: {torch.allclose(result2, expected2)}")