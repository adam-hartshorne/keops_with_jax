"""
Test gradient of Gaussian kernel with Pm parameter
"""
import os
os.environ['PYKEOPS_JAX_MODE'] = '1'

import jax
import jax.numpy as jnp
from pykeops.jax import Genred

print("="*70)
print("Testing Gaussian Kernel Gradients with Pm Parameter")
print("="*70)

x = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
y = jnp.array([[0.0, 0.0]], dtype=jnp.float32)

print("\nx:", x)
print("y:", y)

sigma = 1.0
inv_two_sigma_sq = jnp.array([1.0 / (2 * sigma**2)], dtype=jnp.float32)

# KeOps operator with Pm parameter
formula = "Exp(-SqNorm2(x-y)*inv_sigma)"
aliases = ["x=Vi(2)", "y=Vj(2)", "inv_sigma=Pm(1)"]
op = Genred(formula, aliases, reduction_op='Sum', axis=1)

# Pure JAX reference implementation
def pure_jax_gaussian(x, y, inv_sigma):
    diff = x[:, None, :] - y[None, :, :]  # (2, 1, 2)
    sqnorm = jnp.sum(diff ** 2, axis=2)    # (2, 1)
    kernel = jnp.exp(-sqnorm * inv_sigma[0])
    result = jnp.sum(kernel, axis=1, keepdims=True)  # (2, 1)
    return jnp.sum(result)  # scalar for easier gradient

def keops_gaussian(x, y, inv_sigma):
    result = op(x, y, inv_sigma)
    return jnp.sum(result)  # scalar for easier gradient

# Test 1: Gradient w.r.t. x
print("\n" + "="*70)
print("Test 1: Gradient w.r.t. x")
print("="*70)

grad_x_pure = jax.grad(pure_jax_gaussian, argnums=0)(x, y, inv_two_sigma_sq)
print("Pure JAX gradient w.r.t. x:")
print(grad_x_pure)

grad_x_keops = jax.grad(keops_gaussian, argnums=0)(x, y, inv_two_sigma_sq)
print("\nKeOps JAX gradient w.r.t. x:")
print(grad_x_keops)

match_x = jnp.allclose(grad_x_pure, grad_x_keops, rtol=1e-3, atol=1e-4)
print(f"\n{'✓ PASS' if match_x else '✗ FAIL'}: Gradient w.r.t. x")
if not match_x:
    print(f"Max diff: {jnp.max(jnp.abs(grad_x_pure - grad_x_keops))}")

# Test 2: Gradient w.r.t. y
print("\n" + "="*70)
print("Test 2: Gradient w.r.t. y")
print("="*70)

grad_y_pure = jax.grad(pure_jax_gaussian, argnums=1)(x, y, inv_two_sigma_sq)
print("Pure JAX gradient w.r.t. y:")
print(grad_y_pure)

grad_y_keops = jax.grad(keops_gaussian, argnums=1)(x, y, inv_two_sigma_sq)
print("\nKeOps JAX gradient w.r.t. y:")
print(grad_y_keops)

match_y = jnp.allclose(grad_y_pure, grad_y_keops,  rtol=1e-3, atol=1e-4)
print(f"\n{'✓ PASS' if match_y else '✗ FAIL'}: Gradient w.r.t. y")
if not match_y:
    print(f"Max diff: {jnp.max(jnp.abs(grad_y_pure - grad_y_keops))}")

# Test 3: Can we take gradient w.r.t. the parameter?
print("\n" + "="*70)
print("Test 3: Gradient w.r.t. inv_sigma parameter")
print("="*70)

try:
    grad_param_pure = jax.grad(pure_jax_gaussian, argnums=2)(x, y, inv_two_sigma_sq)
    print("Pure JAX gradient w.r.t. inv_sigma:")
    print(grad_param_pure)

    grad_param_keops = jax.grad(keops_gaussian, argnums=2)(x, y, inv_two_sigma_sq)
    print("\nKeOps JAX gradient w.r.t. inv_sigma:")
    print(grad_param_keops)

    match_param = jnp.allclose(grad_param_pure, grad_param_keops,  rtol=1e-3, atol=1e-4)
    print(f"\n{'✓ PASS' if match_param else '✗ FAIL'}: Gradient w.r.t. parameter")
    if not match_param:
        print(f"Max diff: {jnp.max(jnp.abs(grad_param_pure - grad_param_keops))}")
except Exception as e:
    print(f"✗ FAIL: Gradient w.r.t. parameter - {e}")
    import traceback
    traceback.print_exc()
    match_param = False

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Gradient w.r.t. x: {'✓ PASS' if match_x else '✗ FAIL'}")
print(f"Gradient w.r.t. y: {'✓ PASS' if match_y else '✗ FAIL'}")
print(f"Gradient w.r.t. parameter: {'✓ PASS' if match_param else '✗ FAIL'}")

if match_x and match_y and match_param:
    print("\n🎉 All Gaussian kernel gradient tests passed!")
else:
    print("\n⚠️ Some tests failed")