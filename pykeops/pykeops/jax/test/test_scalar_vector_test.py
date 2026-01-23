"""
Test if scalar * vector multiplication works in KeOps - WITH BATCHING
"""
import os
import sys
sys.path.insert(0, '/media/adam/shared_folder/PycharmProjects/keops/pykeops')
sys.path.insert(0, '/media/adam/shared_folder/PycharmProjects/keops/keopscore')
import jax.numpy as jnp
from pykeops.jax import Genred

# Add batch dimension to force using ranges launcher
batch_size = 2
x = jnp.ones((batch_size, 10, 3))
y = jnp.zeros((batch_size, 5, 3))
b = jnp.array([
    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
])

print(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")
print(f"b shape: {b.shape}")
print()

# Test 1: Just b (should work with batching)
formula1 = "b"
aliases1 = ["x=Vi(3)", "y=Vj(3)", "b=Vj(2)"]
op1 = Genred(formula1, aliases1, reduction_op='Sum', axis=1)
result1 = op1(x, y, b)
print(f"Test 1: formula='b'")
print(f"Result shape: {result1.shape} (expect ({batch_size}, 10, 2))")
print(f"Result batch 0: {result1[0, :3]}")
print(f"Result batch 1: {result1[1, :3]}")
print()

# Test 2: SqDist(x,y) alone
formula2 = "SqDist(x, y)"
aliases2 = ["x=Vi(3)", "y=Vj(3)"]
op2 = Genred(formula2, aliases2, reduction_op='Sum', axis=1)
result2 = op2(x, y)
print(f"Test 2: formula='SqDist(x,y)'")
print(f"Result shape: {result2.shape} (expect ({batch_size}, 10, 1))")
print(f"Result batch 0: {result2[0, :3]}")
print()

# Test 3: SqDist(x,y) * b  (scalar * vector)
formula3 = "SqDist(x, y) * b"
aliases3 = ["x=Vi(3)", "y=Vj(3)", "b=Vj(2)"]
op3 = Genred(formula3, aliases3, reduction_op='Sum', axis=1)
result3 = op3(x, y, b)
print(f"Test 3: formula='SqDist(x,y) * b'")
print(f"Result shape: {result3.shape} (expect ({batch_size}, 10, 2))")
print(f"Result batch 0: {result3[0, :3]}")

# Ground truth for batch 0
sq_dist = jnp.sum((x[0, :, None, :] - y[0, None, :, :])**2, axis=-1)  # (10, 5)
expected = jnp.sum(sq_dist[:, :, None] * b[0, None, :, :], axis=1)  # (10, 2)
print(f"Expected batch 0: {expected[:3]}")
print(f"Match: {jnp.allclose(result3[0], expected)}")