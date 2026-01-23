"""
Minimal test for KeOps JAX gradients - using GenericReduction directly
"""
import os


import jax
import jax.numpy as jnp
from pykeops.jax import Genred

print("="*60)
print("MINIMAL GRADIENT TEST (GenericReduction)")
print("="*60)

# Simple test case
x = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
y = jnp.array([[0.0, 0.0]])              # (1, 2)

print(f"\nInputs:")
print(f"x shape: {x.shape}, values:\n{x}")
print(f"y shape: {y.shape}, values:\n{y}")

# Create KeOps operator: Sum_j((x_i - y_j)^2)
formula = 'Sum((x-y)**2)'
aliases = [
    'x=Vi(2)',  # Vi variable of dimension 2
    'y=Vj(2)',  # Vj variable of dimension 2
]
reduction_op = Genred(formula, aliases, reduction_op='Sum', axis=1)

print("\n" + "="*60)
print("Forward pass...")
print("="*60)

result = reduction_op(x, y)
print(f"\nForward pass result: {result[:, 0]}")
print(f"Expected: [5.0, 25.0]")

# Check forward pass
expected_fwd = jnp.array([[5.0], [25.0]])
if jnp.allclose(result, expected_fwd):
    print("✓ FORWARD PASS CORRECT!")
else:
    print(f"✗ FORWARD PASS WRONG!")
    print(f"Difference: {result - expected_fwd}")

print("\n" + "="*60)
print("Computing gradient w.r.t. x...")
print("="*60)

def loss_fn(x_val):
    return jnp.sum(reduction_op(x_val, y))

try:
    grad_x = jax.grad(loss_fn)(x)
    print(f"\nGradient w.r.t. x:")
    print(f"Shape: {grad_x.shape}")
    print(f"Values:\n{grad_x}")
    print(f"\nExpected shape: (2, 2)")
    print(f"Expected values:\n[[2., 4.],\n [6., 8.]]")

    # Check if correct
    expected = jnp.array([[2., 4.], [6., 8.]])
    if jnp.allclose(grad_x, expected, atol=1e-5):
        print("\n✓ GRADIENT W.R.T. X TEST PASSED!")
    else:
        print(f"\n✗ GRADIENT W.R.T. X TEST FAILED!")
        print(f"Difference:\n{grad_x - expected}")

except Exception as e:
    print(f"\n✗ GRADIENT TEST FAILED WITH ERROR:")
    print(f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Testing gradient w.r.t. y...")
print("="*60)

def loss_fn_y(y_val):
    return jnp.sum(reduction_op(x, y_val))

try:
    grad_y = jax.grad(loss_fn_y)(y)
    print(f"\nGradient w.r.t. y:")
    print(f"Shape: {grad_y.shape}")
    print(f"Values:\n{grad_y}")
    print(f"\nExpected shape: (1, 2)")
    print(f"Expected values:\n[[-4., -8.]]")

    # Check if correct
    expected = jnp.array([[-4., -8.]])
    if jnp.allclose(grad_y, expected, atol=1e-5):
        print("\n✓ GRADIENT W.R.T. Y TEST PASSED!")
    else:
        print(f"\n✗ GRADIENT W.R.T. Y TEST FAILED!")
        print(f"Difference:\n{grad_y - expected}")

except Exception as e:
    print(f"\n✗ GRADIENT TEST FAILED WITH ERROR:")
    print(f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()