#!/usr/bin/env python3
"""
Comprehensive test script for KeOps JAX - Gaussian and Laplacian kernels
Tests both Genred API and LazyTensor API with various configurations
"""

import os
os.environ['PYKEOPS_JAX_MODE'] = '1'

import jax
import jax.numpy as jnp
from pykeops.jax import Genred, LazyTensor, Vi, Vj

print("=" * 80)
print("GAUSSIAN AND LAPLACIAN KERNEL COMPREHENSIVE TEST")
print("=" * 80)
print()

# Test configuration
N = 50  # Number of points in x
M = 30  # Number of points in y
D = 3  # Feature dimension
OUT_DIM = 2  # Output dimension for vector-valued kernels
sigma = jnp.array([[0.5]])

print(f"Test configuration:")
print(f"  N (x points): {N}")
print(f"  M (y points): {M}")
print(f"  D (features): {D}")
print(f"  Output dimension: {OUT_DIM}")
print(f"  sigma: {sigma[0, 0]}")
print()

# Generate test data
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (N, D))
y = jax.random.normal(jax.random.PRNGKey(1), (M, D))
b_scalar = jax.random.normal(jax.random.PRNGKey(2), (M, 1))
b_vector = jax.random.normal(jax.random.PRNGKey(3), (M, OUT_DIM))


# Manual computation for verification
def manual_gaussian_scalar(x, y, b, sigma):
    """Manual Gaussian kernel computation with scalar output"""
    # Compute pairwise distances: (N, M)
    x_expand = x[:, None, :]  # (N, 1, D)
    y_expand = y[None, :, :]  # (1, M, D)
    dist2 = jnp.sum((x_expand - y_expand) ** 2, axis=-1)  # (N, M)

    # Gaussian kernel
    K = jnp.exp(-dist2 * sigma[0, 0])  # (N, M)

    # Weighted sum
    result = jnp.sum(K[:, :, None] * b[None, :, :], axis=1)  # (N, 1)
    return result


def manual_gaussian_vector(x, y, b, sigma):
    """Manual Gaussian kernel computation with vector output"""
    # Compute pairwise distances: (N, M)
    x_expand = x[:, None, :]  # (N, 1, D)
    y_expand = y[None, :, :]  # (1, M, D)
    dist2 = jnp.sum((x_expand - y_expand) ** 2, axis=-1)  # (N, M)

    # Gaussian kernel
    K = jnp.exp(-dist2 * sigma[0, 0])  # (N, M)

    # Weighted sum
    result = jnp.sum(K[:, :, None] * b[None, :, :], axis=1)  # (N, OUT_DIM)
    return result


def manual_laplacian_scalar(x, y, b, sigma):
    """Manual Laplacian kernel computation with scalar output"""
    x_expand = x[:, None, :]
    y_expand = y[None, :, :]
    dist = jnp.sqrt(jnp.sum((x_expand - y_expand) ** 2, axis=-1))
    K = jnp.exp(-dist * sigma[0, 0])
    result = jnp.sum(K[:, :, None] * b[None, :, :], axis=1)
    return result


def manual_laplacian_vector(x, y, b, sigma):
    """Manual Laplacian kernel computation with vector output"""
    x_expand = x[:, None, :]
    y_expand = y[None, :, :]
    dist = jnp.sqrt(jnp.sum((x_expand - y_expand) ** 2, axis=-1))
    K = jnp.exp(-dist * sigma[0, 0])
    result = jnp.sum(K[:, :, None] * b[None, :, :], axis=1)
    return result


def compare_results(result, expected, test_name):
    """Compare results and print diagnostics"""
    print(f"Result shape: {result.shape}")
    print(f"Manual shape: {expected.shape}")

    if result.shape != expected.shape:
        print(f"✗ FAIL: Shape mismatch!")
        return False

    abs_error = jnp.max(jnp.abs(result - expected))
    rel_error = jnp.max(jnp.abs(result - expected) / (jnp.abs(expected) + 1e-10))

    print(f"Max absolute error: {abs_error:.2e}")
    print(f"Max relative error: {rel_error:.2e}")

    if abs_error < 1e-5:
        print(f"✓ PASS")
        return True
    else:
        print(f"✗ FAIL")
        print(f"First 3 results:")
        print(result[:3])
        print(f"First 3 manual:")
        print(expected[:3])
        return False


# ============================================================================
# TEST 1: Gaussian Kernel - Genred API - Scalar Output
# ============================================================================
print("=" * 80)
print("TEST 1: Gaussian Kernel - Genred API - Scalar Output")
print("-" * 80)

formula = "Exp(-(Sum((a-b)**2)*d))*c"
aliases = ["a=Vi(3)", "b=Vj(3)", "c=Vj(1)", "d=Pm(1)"]
print(f"Formula: {formula}")
print(f"Expected output shape: ({N}, 1)")
print()

genred_gaussian_scalar = Genred(formula, aliases, reduction_op='Sum', axis=1)
result1 = genred_gaussian_scalar(x, y, b_scalar, sigma)
expected1 = manual_gaussian_scalar(x, y, b_scalar, sigma)

test1_pass = compare_results(result1, expected1, "Genred Gaussian Scalar")
print()

# ============================================================================
# TEST 2: Gaussian Kernel - Genred API - Vector Output
# ============================================================================
print("=" * 80)
print("TEST 2: Gaussian Kernel - Genred API - Vector Output")
print("-" * 80)

formula = "Exp(-(Sum((a-b)**2)*d))*c"
aliases = ["a=Vi(3)", "b=Vj(3)", "c=Vj(2)", "d=Pm(1)"]
print(f"Formula: {formula}")
print(f"Expected output shape: ({N}, {OUT_DIM})")
print()

genred_gaussian_vector = Genred(formula, aliases, reduction_op='Sum', axis=1)
result2 = genred_gaussian_vector(x, y, b_vector, sigma)
expected2 = manual_gaussian_vector(x, y, b_vector, sigma)

test2_pass = compare_results(result2, expected2, "Genred Gaussian Vector")
print()

# ============================================================================
# TEST 3: Laplacian Kernel - Genred API - Scalar Output
# ============================================================================
print("=" * 80)
print("TEST 3: Laplacian Kernel - Genred API - Scalar Output")
print("-" * 80)

formula = "Exp(-(Sqrt(Sum((a-b)**2))*d))*c"
aliases = ["a=Vi(3)", "b=Vj(3)", "c=Vj(1)", "d=Pm(1)"]
print(f"Formula: {formula}")
print(f"Expected output shape: ({N}, 1)")
print()

genred_laplacian_scalar = Genred(formula, aliases, reduction_op='Sum', axis=1)
result3 = genred_laplacian_scalar(x, y, b_scalar, sigma)
expected3 = manual_laplacian_scalar(x, y, b_scalar, sigma)

test3_pass = compare_results(result3, expected3, "Genred Laplacian Scalar")
print()

# ============================================================================
# TEST 4: Laplacian Kernel - Genred API - Vector Output
# ============================================================================
print("=" * 80)
print("TEST 4: Laplacian Kernel - Genred API - Vector Output")
print("-" * 80)

formula = "Exp(-(Sqrt(Sum((a-b)**2))*d))*c"
aliases = ["a=Vi(3)", "b=Vj(3)", "c=Vj(2)", "d=Pm(1)"]
print(f"Formula: {formula}")
print(f"Expected output shape: ({N}, {OUT_DIM})")
print()

genred_laplacian_vector = Genred(formula, aliases, reduction_op='Sum', axis=1)
result4 = genred_laplacian_vector(x, y, b_vector, sigma)
expected4 = manual_laplacian_vector(x, y, b_vector, sigma)

test4_pass = compare_results(result4, expected4, "Genred Laplacian Vector")
print()

# ============================================================================
# TEST 5: Gaussian Kernel - LazyTensor API - Vector Output (Implicit Axis)
# ============================================================================
print("=" * 80)
print("TEST 5: Gaussian Kernel - LazyTensor API - Vector Output")
print("-" * 80)
print("Using implicit axis inference from 3D tensor shapes")
print()
print(f"Expected output shape: ({N}, {OUT_DIM})")
print()

# Create LazyTensors with explicit reshaping
x_i = LazyTensor(x[:, None, :])  # (N, 1, D)
y_j = LazyTensor(y[None, :, :])  # (1, M, D)
b_j = LazyTensor(b_vector[None, :, :])  # (1, M, OUT_DIM)

print(f"x_i: shape {x_i.variables[0].shape}, axis={x_i.axis} (expected: axis=0)")
print(f"y_j: shape {y_j.variables[0].shape}, axis={y_j.axis} (expected: axis=1)")
print(f"b_j: shape {b_j.variables[0].shape}, axis={b_j.axis} (expected: axis=1)")
print()

# Compute Gaussian kernel
D2 = ((x_i - y_j) ** 2).sum(-1)  # Squared distances
K = (-D2 * sigma[0, 0]).exp()  # Gaussian kernel
result5 = (K * b_j).sum(1)  # weighted sum

expected5 = manual_gaussian_vector(x, y, b_vector, sigma)

test5_pass = compare_results(result5, expected5, "LazyTensor Gaussian Vector")
print()

# ============================================================================
# TEST 6: LazyTensor with Explicit Axis (Should Fail by Design)
# ============================================================================
print("=" * 80)
print("TEST 6: LazyTensor with Explicit Axis (Expected to Fail)")
print("-" * 80)
print("Testing explicit axis parameter with 3D tensors")
print("This SHOULD fail because base class doesn't allow explicit axis with 3D tensors")
print()

try:
    x_i = LazyTensor(x[:, None, :], axis=0)  # Should raise ValueError
    print("✗ UNEXPECTED: No error raised!")
    test6_pass = False
except ValueError as e:
    print(f"✓ EXPECTED ERROR: {e}")
    print("This is correct behavior - axis is automatically inferred from 3D shapes")
    test6_pass = True  # This is expected behavior
print()

# ============================================================================
# TEST 7: LazyTensor with Vi/Vj Constructors (Now Fixed!)
# ============================================================================
print("=" * 80)
print("TEST 7: LazyTensor with Vi/Vj Constructors (Best Practice)")
print("-" * 80)
print("Testing the Vi/Vj/Pm helper constructors with auto-reshaping")
print()

# Use Vi/Vj helpers (now with auto-reshaping)
x_i = Vi(x)  # Auto-reshapes (N, D) -> (N, 1, D)
y_j = Vj(y)  # Auto-reshapes (M, D) -> (1, M, D)
b_j = Vj(b_vector)  # Auto-reshapes (M, OUT_DIM) -> (1, M, OUT_DIM)

print(f"x_i: axis={x_i.axis}")
print(f"y_j: axis={y_j.axis}")
print(f"b_j: axis={b_j.axis}")
print()

# Compute Gaussian kernel
D2 = ((x_i - y_j) ** 2).sum(-1)
K = (-D2 * sigma[0, 0]).exp()
result7 = (K * b_j).sum(1)

expected7 = manual_gaussian_vector(x, y, b_vector, sigma)

test7_pass = compare_results(result7, expected7, "LazyTensor Vi/Vj Vector")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

tests = [
    ("TEST 1: Gaussian Kernel - Genred - Scalar Output", test1_pass),
    ("TEST 2: Gaussian Kernel - Genred - Vector Output", test2_pass),
    ("TEST 3: Laplacian Kernel - Genred - Scalar Output", test3_pass),
    ("TEST 4: Laplacian Kernel - Genred - Vector Output", test4_pass),
    ("TEST 5: LazyTensor - Implicit Axis - Vector Output", test5_pass),
    ("TEST 6: LazyTensor - Explicit Axis (Expected Fail)", test6_pass),
    ("TEST 7: LazyTensor - Vi/Vj Constructors", test7_pass),
]

passed = sum(1 for _, result in tests if result)
total = len(tests)

print(f"Results: {passed}/{total} tests passed")
print()

for test_name, result in tests:
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"{status}: {test_name}")

print()
if passed == total:
    print("🎉 ALL TESTS PASSED! KeOps JAX is fully functional!")
else:
    print(f"⚠️  {total - passed} test(s) failed")

print()
print("Expected results:")
print("  Tests 1-5: Should ALL PASS ✓")
print("  Test 6: Should PASS (expected error caught) ✓")
print("  Test 7: Should PASS (Vi/Vj now auto-reshape) ✓")