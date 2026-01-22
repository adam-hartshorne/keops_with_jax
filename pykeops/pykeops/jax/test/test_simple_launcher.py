"""
Test if scalar * vector multiplication works in KeOps - NO BATCHING (simple launcher)
"""
import os
os.environ['PYKEOPS_JAX_MODE'] = '1'
import sys
sys.path.insert(0, '/media/adam/shared_folder/PycharmProjects/keops/pykeops')
sys.path.insert(0, '/media/adam/shared_folder/PycharmProjects/keops/keopscore')
import jax.numpy as jnp
from pykeops.jax import Genred

# NO batch dimension to test simple launcher
x = jnp.ones((10, 3))
y = jnp.zeros((5, 3))
b = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])

print(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")
print(f"b shape: {b.shape}")
print()

# Test 1: Just b (should work with simple launcher now)
formula1 = "b"
aliases1 = ["x=Vi(3)", "y=Vj(3)", "b=Vj(2)"]
op1 = Genred(formula1, aliases1, reduction_op='Sum', axis=1)
result1 = op1(x, y, b)
print(f"Test 1: formula='b'")
print(f"Result shape: {result1.shape} (expect (10, 2))")
print(f"Result: {result1[:3]}")
expected1 = jnp.array([[25., 30.]] * 10)  # Sum of all b values
print(f"Expected: {expected1[:3]}")
print(f"Match: {jnp.allclose(result1, expected1)}")
print()

# Test 2: SqDist(x,y) * b  (scalar * vector)
formula2 = "SqDist(x, y) * b"
aliases2 = ["x=Vi(3)", "y=Vj(3)", "b=Vj(2)"]
op2 = Genred(formula2, aliases2, reduction_op='Sum', axis=1)
result2 = op2(x, y, b)
print(f"Test 2: formula='SqDist(x,y) * b'")
print(f"Result shape: {result2.shape} (expect (10, 2))")
print(f"Result: {result2[:3]}")

# Ground truth
sq_dist = jnp.sum((x[:, None, :] - y[None, :, :])**2, axis=-1)  # (10, 5)
expected2 = jnp.sum(sq_dist[:, :, None] * b[None, :, :], axis=1)  # (10, 2)
print(f"Expected: {expected2[:3]}")
print(f"Match: {jnp.allclose(result2, expected2)}")