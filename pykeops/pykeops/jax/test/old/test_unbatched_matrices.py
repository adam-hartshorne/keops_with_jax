"""
Comprehensive non-batched tests for KeOps JAX
Tests: forward pass, gradients, JIT, LazyTensor, and generic operations
"""

import os
os.environ['PYKEOPS_JAX_MODE'] = '1'

import jax
import jax.numpy as jnp
import numpy as np

print("=" * 70)
print("NON-BATCHED KEOPS JAX TESTS")
print("=" * 70)

from pykeops.jax import Genred

# ============================================================================
# Test 1: Forward Pass
# ============================================================================
print("\n" + "=" * 70)
print("TEST 1: Forward Pass")
print("=" * 70)

x = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
y = jnp.array([[0.0, 0.0]], dtype=jnp.float32)

print(f"x shape: {x.shape}")  # (2, 2)
print(f"y shape: {y.shape}")  # (1, 2)
print(f"x:\n{x}")
print(f"y:\n{y}")

# Pure JAX reference
def pure_jax_sqdist(x, y):
    diff = x[:, None, :] - y[None, :, :]  # (2, 1, 2)
    sqdist = jnp.sum(diff ** 2, axis=2)    # (2, 1)
    return jnp.sum(sqdist, axis=1, keepdims=True)  # (2, 1)

result_pure = pure_jax_sqdist(x, y)
print(f"\nPure JAX result:\n{result_pure}")
print(f"Pure JAX result shape: {result_pure.shape}")

# KeOps JAX
formula = "Sum((x-y)**2)"
aliases = ["x=Vi(2)", "y=Vj(2)"]
op = Genred(formula, aliases, reduction_op='Sum', axis=1)

try:
    result_keops = op(x, y)
    print(f"\nKeOps JAX result:\n{result_keops}")
    print(f"KeOps JAX result shape: {result_keops.shape}")

    forward_match = jnp.allclose(result_keops, result_pure, rtol=1e-5)
    print(f"\n{'✓ PASS' if forward_match else '✗ FAIL'}: Forward pass")

    if not forward_match:
        print(f"Max diff: {jnp.max(jnp.abs(result_keops - result_pure))}")
except Exception as e:
    print(f"✗ FAIL: Forward pass - {e}")
    import traceback
    traceback.print_exc()
    forward_match = False

# ============================================================================
# Test 2: Gradient w.r.t. x (Vi variable)
# ============================================================================
print("\n" + "=" * 70)
print("TEST 2: Gradient w.r.t. x (Vi variable)")
print("=" * 70)

def loss_x_pure(x):
    return jnp.sum(pure_jax_sqdist(x, y))

grad_x_pure = jax.grad(loss_x_pure)(x)
print(f"Pure JAX grad w.r.t. x:\n{grad_x_pure}")
print(f"Shape: {grad_x_pure.shape}")

def loss_x_keops(x):
    return jnp.sum(op(x, y))

try:
    grad_x_keops = jax.grad(loss_x_keops)(x)
    print(f"\nKeOps JAX grad w.r.t. x:\n{grad_x_keops}")
    print(f"Shape: {grad_x_keops.shape}")

    grad_x_match = jnp.allclose(grad_x_keops, grad_x_pure, rtol=1e-5)
    print(f"\n{'✓ PASS' if grad_x_match else '✗ FAIL'}: Gradient w.r.t. x")

    if not grad_x_match:
        print(f"Max diff: {jnp.max(jnp.abs(grad_x_keops - grad_x_pure))}")
except Exception as e:
    print(f"✗ FAIL: Gradient w.r.t. x - {e}")
    import traceback
    traceback.print_exc()
    grad_x_match = False

# ============================================================================
# Test 3: Gradient w.r.t. y (Vj variable)
# ============================================================================
print("\n" + "=" * 70)
print("TEST 3: Gradient w.r.t. y (Vj variable)")
print("=" * 70)

def loss_y_pure(y):
    return jnp.sum(pure_jax_sqdist(x, y))

grad_y_pure = jax.grad(loss_y_pure)(y)
print(f"Pure JAX grad w.r.t. y:\n{grad_y_pure}")
print(f"Shape: {grad_y_pure.shape}")

def loss_y_keops(y):
    return jnp.sum(op(x, y))

try:
    grad_y_keops = jax.grad(loss_y_keops)(y)
    print(f"\nKeOps JAX grad w.r.t. y:\n{grad_y_keops}")
    print(f"Shape: {grad_y_keops.shape}")

    grad_y_match = jnp.allclose(grad_y_keops, grad_y_pure, rtol=1e-5)
    print(f"\n{'✓ PASS' if grad_y_match else '✗ FAIL'}: Gradient w.r.t. y")

    if not grad_y_match:
        print(f"Max diff: {jnp.max(jnp.abs(grad_y_keops - grad_y_pure))}")
except Exception as e:
    print(f"✗ FAIL: Gradient w.r.t. y - {e}")
    import traceback
    traceback.print_exc()
    grad_y_match = False

# ============================================================================
# Test 4: JIT compilation
# ============================================================================
print("\n" + "=" * 70)
print("TEST 4: JIT compilation")
print("=" * 70)

@jax.jit
def jitted_keops(x, y):
    return op(x, y)

try:
    # First call (compilation)
    result_jit1 = jitted_keops(x, y)
    print(f"First JIT call result:\n{result_jit1}")

    # Second call (cached)
    result_jit2 = jitted_keops(x, y)
    print(f"Second JIT call result:\n{result_jit2}")

    jit_match = jnp.allclose(result_jit1, result_pure, rtol=1e-5) and jnp.allclose(result_jit2, result_pure, rtol=1e-5)
    print(f"\n{'✓ PASS' if jit_match else '✗ FAIL'}: JIT compilation")

    if not jit_match:
        print(f"Max diff (call 1): {jnp.max(jnp.abs(result_jit1 - result_pure))}")
        print(f"Max diff (call 2): {jnp.max(jnp.abs(result_jit2 - result_pure))}")
except Exception as e:
    print(f"✗ FAIL: JIT compilation - {e}")
    import traceback
    traceback.print_exc()
    jit_match = False

# ============================================================================
# Test 5: Different formula (Gaussian kernel)
# ============================================================================
print("\n" + "=" * 70)
print("TEST 5: Different formula (Gaussian kernel)")
print("=" * 70)

sigma = 1.0
# KeOps doesn't accept float literals - pass constants as Pm parameters
formula_gauss = "Exp(-SqNorm2(x-y)*inv_sigma)"
aliases_gauss = ["x=Vi(2)", "y=Vj(2)", "inv_sigma=Pm(1)"]
op_gauss = Genred(formula_gauss, aliases_gauss, reduction_op='Sum', axis=1)

# Pass the inverse of 2*sigma^2 as a parameter
inv_two_sigma_sq = jnp.array([1.0 / (2 * sigma**2)], dtype=jnp.float32)

def pure_jax_gaussian(x, y, inv_sigma):
    diff = x[:, None, :] - y[None, :, :]  # (2, 1, 2)
    sqnorm = jnp.sum(diff ** 2, axis=2)    # (2, 1)
    kernel = jnp.exp(-sqnorm * inv_sigma[0])
    return jnp.sum(kernel, axis=1, keepdims=True)  # (2, 1)

result_gauss_pure = pure_jax_gaussian(x, y, inv_two_sigma_sq)
print(f"Pure JAX Gaussian result:\n{result_gauss_pure}")

try:
    result_gauss_keops = op_gauss(x, y, inv_two_sigma_sq)
    print(f"KeOps JAX Gaussian result:\n{result_gauss_keops}")

    gauss_match = jnp.allclose(result_gauss_keops, result_gauss_pure, rtol=1e-5)
    print(f"\n{'✓ PASS' if gauss_match else '✗ FAIL'}: Gaussian kernel")

    if not gauss_match:
        print(f"Max diff: {jnp.max(jnp.abs(result_gauss_keops - result_gauss_pure))}")
except Exception as e:
    print(f"✗ FAIL: Gaussian kernel - {e}")
    import traceback
    traceback.print_exc()
    gauss_match = False

# ============================================================================
# Test 6: Larger arrays
# ============================================================================
print("\n" + "=" * 70)
print("TEST 6: Larger arrays")
print("=" * 70)

x_large = jnp.arange(20, dtype=jnp.float32).reshape(10, 2)
y_large = jnp.zeros((5, 2), dtype=jnp.float32)

print(f"x_large shape: {x_large.shape}")  # (10, 2)
print(f"y_large shape: {y_large.shape}")  # (5, 2)

result_large_pure = pure_jax_sqdist(x_large, y_large)
print(f"Pure JAX result shape: {result_large_pure.shape}")

try:
    result_large_keops = op(x_large, y_large)
    print(f"KeOps JAX result shape: {result_large_keops.shape}")

    large_match = jnp.allclose(result_large_keops, result_large_pure, rtol=1e-5)
    print(f"\n{'✓ PASS' if large_match else '✗ FAIL'}: Larger arrays")

    if not large_match:
        print(f"Max diff: {jnp.max(jnp.abs(result_large_keops - result_large_pure))}")
        print(f"Expected:\n{result_large_pure[:3]}")
        print(f"Got:\n{result_large_keops[:3]}")
except Exception as e:
    print(f"✗ FAIL: Larger arrays - {e}")
    import traceback
    traceback.print_exc()
    large_match = False

# ============================================================================
# Test 7: Different reduction (axis=0)
# ============================================================================
print("\n" + "=" * 70)
print("TEST 7: Different reduction (axis=0)")
print("=" * 70)

formula_axis0 = "Sum((x-y)**2)"
aliases_axis0 = ["x=Vi(2)", "y=Vj(2)"]
op_axis0 = Genred(formula_axis0, aliases_axis0, reduction_op='Sum', axis=0)

def pure_jax_sqdist_axis0(x, y):
    diff = x[:, None, :] - y[None, :, :]  # (2, 1, 2)
    sqdist = jnp.sum(diff ** 2, axis=2)    # (2, 1)
    return jnp.sum(sqdist, axis=0, keepdims=True)  # (1, 1)

result_axis0_pure = pure_jax_sqdist_axis0(x, y)
print(f"Pure JAX result (axis=0):\n{result_axis0_pure}")
print(f"Pure JAX result shape: {result_axis0_pure.shape}")

try:
    result_axis0_keops = op_axis0(x, y)
    print(f"KeOps JAX result (axis=0):\n{result_axis0_keops}")
    print(f"KeOps JAX result shape: {result_axis0_keops.shape}")

    axis0_match = jnp.allclose(result_axis0_keops, result_axis0_pure, rtol=1e-5)
    print(f"\n{'✓ PASS' if axis0_match else '✗ FAIL'}: axis=0 reduction")

    if not axis0_match:
        print(f"Max diff: {jnp.max(jnp.abs(result_axis0_keops - result_axis0_pure))}")
except Exception as e:
    print(f"✗ FAIL: axis=0 reduction - {e}")
    import traceback
    traceback.print_exc()
    axis0_match = False

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

results = {
    "Forward pass": forward_match,
    "Gradient w.r.t. x": grad_x_match,
    "Gradient w.r.t. y": grad_y_match,
    "JIT compilation": jit_match,
    "Gaussian kernel": gauss_match,
    "Larger arrays": large_match,
    "axis=0 reduction": axis0_match,
}

for name, result in results.items():
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"{status}: {name}")

passed = sum(1 for r in results.values() if r)
total = len(results)
print(f"\nPassed: {passed}/{total}")

if passed == total:
    print("\n🎉 ALL TESTS PASSED! 🎉")
else:
    print(f"\n❌ {total - passed} test(s) failed")