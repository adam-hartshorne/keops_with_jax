"""
Simple gradient test for debugging
"""
import os
os.environ['PYKEOPS_JAX_MODE'] = '1'

import jax
import jax.numpy as jnp

from pykeops.jax import Genred

x = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
y = jnp.array([[0.0, 0.0]], dtype=jnp.float32)

print("x:", x)
print("y:", y)

formula = "Sum((x-y)**2)"
aliases = ["x=Vi(2)", "y=Vj(2)"]
op = Genred(formula, aliases, reduction_op='Sum', axis=1)

def loss_fn(x):
    result = op(x, y)
    print(f"\nForward result: {result}")
    return jnp.sum(result)

print("\n" + "="*70)
print("Computing gradient...")
print("="*70)

grad_fn = jax.grad(loss_fn)
grad_x = grad_fn(x)

print(f"\nGrad result: {grad_x}")
print(f"Expected: [[2., 4.], [6., 8.]]")