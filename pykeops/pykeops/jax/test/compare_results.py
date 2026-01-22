#!/usr/bin/env python3
"""
Compare saved PyTorch and JAX results
"""
import numpy as np
import pickle

print("=" * 80)
print("COMPARING PYTORCH vs JAX KEOPS RESULTS")
print("=" * 80)

# Load PyTorch results
print("\nLoading PyTorch results...")
with open('pytorch_results.pkl', 'rb') as f:
    pytorch = pickle.load(f)

# Load JAX results
print("Loading JAX results...")
with open('jax_results.pkl', 'rb') as f:
    jax = pickle.load(f)

# Verify same inputs
print("\n" + "=" * 80)
print("VERIFYING INPUTS ARE IDENTICAL")
print("=" * 80)
assert np.allclose(pytorch['x'], jax['x']), "x arrays differ!"
assert np.allclose(pytorch['y'], jax['y']), "y arrays differ!"
assert np.allclose(pytorch['b'], jax['b']), "b arrays differ!"
assert pytorch['s'] == jax['s'], "s values differ!"
print("✓ All inputs are identical")

# Compare NumPy expected values
print("\n" + "=" * 80)
print("COMPARING NUMPY EXPECTED VALUES")
print("=" * 80)
diff_expected = np.abs(pytorch['numpy_expected'] - jax['numpy_expected'])
print(f"Max difference: {diff_expected.max():.6e}")
if diff_expected.max() < 1e-10:
    print("✓ NumPy expected values are identical")
else:
    print("⚠ NumPy expected values differ slightly (due to different computation)")

# Compare KeOps results
print("\n" + "=" * 80)
print("COMPARING KEOPS RESULTS: PyTorch vs JAX")
print("=" * 80)

pytorch_result = pytorch['keops_result']
jax_result = jax['keops_result']

diff = np.abs(pytorch_result - jax_result)
rel = diff / (np.abs(pytorch_result) + 1e-10)

print(f"PyTorch KeOps result shape: {pytorch_result.shape}")
print(f"JAX KeOps result shape:      {jax_result.shape}")
print(f"\nDifference statistics:")
print(f"  Max absolute error: {diff.max():.6e}")
print(f"  Mean absolute error: {diff.mean():.6e}")
print(f"  Max relative error:  {rel.max():.6e}")
print(f"  Mean relative error: {rel.mean():.6e}")

# Show first few elements
print(f"\nFirst element comparison:")
print(f"  PyTorch: {pytorch_result[0]}")
print(f"  JAX:     {jax_result[0]}")
print(f"  Diff:    {diff[0]}")

# Compare both against NumPy ground truth
print("\n" + "=" * 80)
print("ACCURACY vs NUMPY GROUND TRUTH")
print("=" * 80)

numpy_expected = pytorch['numpy_expected']

# PyTorch vs NumPy
diff_pytorch = np.abs(pytorch_result - numpy_expected)
rel_pytorch = diff_pytorch / (np.abs(numpy_expected) + 1e-10)

print(f"PyTorch KeOps vs NumPy:")
print(f"  Max absolute error: {diff_pytorch.max():.6e}")
print(f"  Max relative error:  {rel_pytorch.max():.6e}")

# JAX vs NumPy
diff_jax = np.abs(jax_result - numpy_expected)
rel_jax = diff_jax / (np.abs(numpy_expected) + 1e-10)

print(f"\nJAX KeOps vs NumPy:")
print(f"  Max absolute error: {diff_jax.max():.6e}")
print(f"  Max relative error:  {rel_jax.max():.6e}")

# JAX vs JAX expected (if available)
if 'jax_expected' in jax:
    print("\n" + "=" * 80)
    print("JAX KEOPS vs JAX EXPECTED (showing the problem)")
    print("=" * 80)
    diff_jax_vs_jax_expected = np.abs(jax_result - jax['jax_expected'])
    rel_jax_vs_jax_expected = diff_jax_vs_jax_expected / (np.abs(jax['jax_expected']) + 1e-10)

    print(f"JAX KeOps vs JAX expected:")
    print(f"  Max absolute error: {diff_jax_vs_jax_expected.max():.6e}")
    print(f"  Max relative error:  {rel_jax_vs_jax_expected.max():.6e}")

    diff_jax_expected_vs_numpy = np.abs(jax['jax_expected'] - numpy_expected)
    print(f"\nJAX expected vs NumPy expected:")
    print(f"  Max absolute error: {diff_jax_expected_vs_numpy.max():.6e}")
    print(f"  (This is why the test was failing!)")

# Final verdict
print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

if diff.max() < 1e-6:
    print("✓ PyTorch and JAX KeOps produce IDENTICAL results")
    print(f"  Difference: {diff.max():.6e} (< 1e-6)")
else:
    print("⚠ PyTorch and JAX KeOps produce DIFFERENT results")
    print(f"  Difference: {diff.max():.6e}")

if rel_pytorch.max() < 1e-5 and rel_jax.max() < 1e-5:
    print("\n✓ Both PyTorch and JAX KeOps match NumPy with high accuracy")
    print(f"  PyTorch max error: {rel_pytorch.max():.6e}")
    print(f"  JAX max error:     {rel_jax.max():.6e}")
elif rel_pytorch.max() < 1e-5 and rel_jax.max() > 1e-3:
    print("\n⚠ JAX KeOps has much worse accuracy than PyTorch!")
    print(f"  PyTorch max error: {rel_pytorch.max():.6e} ✓")
    print(f"  JAX max error:     {rel_jax.max():.6e} ✗")
    print("  → This indicates a bug in JAX KeOps implementation")
else:
    print("\n✓ Both have similar accuracy vs NumPy")

print("=" * 80)