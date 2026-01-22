#!/usr/bin/env python3
"""
Minimal test for second-order gradients
"""
import os
os.environ['PYKEOPS_JAX_MODE'] = '1'

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
os.environ['JAX_KEOPS_DEBUG'] = '1'

import jax
import jax.numpy as jnp
from pykeops.jax import LazyTensor

print("=" * 70)
print("Minimal Second-Order Gradient Test")
print("=" * 70)

# Simple test case
x = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
y = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)

print(f"\nInput shapes: x={x.shape}, y={y.shape}")


def loss(x):
    x_i = LazyTensor(x[:, None, :])
    y_j = LazyTensor(y[None, :, :])
    result = ((x_i - y_j) ** 2).sum(-1).sum(1)
    return result.sum()


# Test forward
print("\n1. Testing forward pass...")
try:
    result = loss(x)
    print(f"✓ Forward: {float(result)}")
except Exception as e:
    print(f"✗ Forward failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Test first derivative
print("\n2. Testing first derivative...")
try:
    grad1 = jax.grad(loss)(x)
    print(f"✓ First derivative: {grad1}")
    print(f"  Shape: {grad1.shape}")
except Exception as e:
    print(f"✗ First derivative failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Test second derivative
print("\n3. Testing second derivative...")
try:
    def grad_norm(x):
        g = jax.grad(loss)(x)
        return jnp.sum(g * x)


    print("  Computing grad(grad(loss) · x)...")
    hess_diag = jax.grad(grad_norm)(x)
    print(f"✓ Second derivative: {hess_diag}")
    print(f"  Shape: {hess_diag.shape}")
    print(f"  Non-zero: {jnp.any(hess_diag != 0)}")
except ValueError as e:
    print(f"✗ Second derivative failed with ValueError")
    print(f"  Error: {e}")
    print("\n  This typically means the FFI call cannot be differentiated.")
    print("  The grad_op needs enable_vjp=True and proper custom_vjp implementation.")
    import traceback

    traceback.print_exc()
    exit(1)
except Exception as e:
    print(f"✗ Second derivative failed: {type(e).__name__}")
    print(f"  Error: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Compare with pure JAX
print("\n4. Comparing with pure JAX...")
try:
    def loss_jax(x):
        return jnp.sum((x[:, None, :] - y[None, :, :]) ** 2)


    hess_jax = jax.grad(lambda x: jnp.sum(jax.grad(loss_jax)(x) * x))(x)

    print(f"Pure JAX Hessian: {hess_jax}")
    print(f"KeOps Hessian:    {hess_diag}")
    print(f"Match: {jnp.allclose(hess_diag, hess_jax, rtol=1e-4)}")

    if not jnp.allclose(hess_diag, hess_jax, rtol=1e-4):
        print(f"\n⚠ Values don't match!")
        print(f"  Max difference: {jnp.max(jnp.abs(hess_diag - hess_jax))}")
        if jnp.all(hess_diag == 0):
            print(f"  KeOps Hessian is all zeros - grad_op_with_vjp_bwd needs implementation")
    else:
        print(f"\n✓ All tests passed!")

except Exception as e:
    print(f"Comparison failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 70)