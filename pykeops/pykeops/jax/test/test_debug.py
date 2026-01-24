#!/usr/bin/env python3
"""
Debug test for high-dimensional JAX/KeOps support.

This tests the dynamic block size adjustment fix that prevents
"invalid argument" errors when using high-dimensional variables.

The bug was: CMake launcher used fixed block size (192) regardless of
shared memory constraints. For D=128:
  shared_mem = 192 * 128 * 4 = 98,304 bytes (96KB) > 48KB limit!

The fix: Dynamically adjust block size like NVRTC does:
  effective_block_size = min(192, 48000 / (128 * 4)) = 93
  shared_mem = 93 * 128 * 4 = 47,616 bytes (46.5KB) ✓
"""

import os
import sys

# Force recompilation by clearing caches
print("=" * 60)
print("Clearing KeOps caches...")
print("=" * 60)

import shutil
import glob

# Clear file caches
cache_dirs = [
    os.path.expanduser("~/.cache/keops"),
    os.path.expanduser("~/.cache/keops2.2"),
    os.path.expanduser("~/.cache/keops2.3"),
]
for d in cache_dirs:
    if os.path.exists(d):
        shutil.rmtree(d)
        print(f"  Removed {d}")

# Clear build folder (CRITICAL - contains old .so with buggy launcher)
try:
    from keopscore.config import config
    build_folder = config.get_build_folder()
    if os.path.exists(build_folder):
        for f in glob.glob(os.path.join(build_folder, "*.so")):
            os.remove(f)
            print(f"  Removed {f}")
        for f in glob.glob(os.path.join(build_folder, "*_cache.pkl")):
            os.remove(f)
            print(f"  Removed {f}")
except Exception as e:
    print(f"  Warning: Could not clear build folder: {e}")

print("\n" + "=" * 60)
print("Starting tests...")
print("=" * 60)

os.environ["PYKEOPS_JAX_MODE"] = "1"

import numpy as np
import jax
import jax.numpy as jnp

print(f"\nJAX devices: {jax.devices()}")

from pykeops.jax import Genred


def test_high_dim(d, formula_name, formula, aliases_template):
    """Test a formula with dimension d."""
    print(f"\n{'='*60}")
    print(f"Test: {formula_name} with D={d}")
    print(f"{'='*60}")
    
    # Calculate expected values
    cuda_block_size = 192
    dtype_bytes = 4
    old_shared = cuda_block_size * d * dtype_bytes
    effective_block = min(cuda_block_size, 48000 // (d * dtype_bytes))
    new_shared = effective_block * d * dtype_bytes
    
    print(f"\nExpected calculations:")
    print(f"  OLD (buggy):  block_size=192, shared_mem={old_shared} bytes ({old_shared/1024:.1f}KB)")
    print(f"  NEW (fixed):  block_size={effective_block}, shared_mem={new_shared} bytes ({new_shared/1024:.1f}KB)")
    print(f"  48KB limit exceeded? OLD={old_shared > 49152}, NEW={new_shared > 49152}")
    
    n, m = 100, 100
    np.random.seed(42)
    x = jnp.array(np.random.randn(n, d).astype(np.float32))
    y = jnp.array(np.random.randn(m, d).astype(np.float32))
    
    aliases = [a.format(d=d) for a in aliases_template]
    
    print(f"\nFormula: {formula}")
    print(f"Aliases: {aliases}")
    print(f"Input shapes: x={x.shape}, y={y.shape}")
    
    try:
        print("\nCreating kernel...")
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)
        
        print("Running kernel...")
        result = op(x, y)
        result.block_until_ready()
        
        print(f"\n✓ SUCCESS! Result shape: {result.shape}")
        print(f"  Result sample: {result[:3].flatten()}")
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        return False


def test_gradient(d):
    """Test gradient computation with high-dimensional inputs."""
    print(f"\n{'='*60}")
    print(f"Test: Gradient with D={d}")
    print(f"{'='*60}")
    
    n, m = 100, 100
    np.random.seed(42)
    x = jnp.array(np.random.randn(n, d).astype(np.float32))
    y = jnp.array(np.random.randn(m, d).astype(np.float32))
    
    formula = "Exp(-SqNorm2(x-y))"
    aliases = [f"x=Vi({d})", f"y=Vj({d})"]
    
    print(f"Formula: {formula}")
    print(f"Testing jax.grad()...")
    
    try:
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)
        
        def loss_fn(x_in):
            return jnp.sum(op(x_in, y))
        
        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(x)
        grad.block_until_ready()
        
        print(f"\n✓ SUCCESS! Gradient shape: {grad.shape}")
        print(f"  Gradient sample: {grad[0, :3]}")
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# Run tests
results = []

# Test 1: D=128 (would fail without fix: 96KB > 48KB)
results.append(("SqDist D=128", test_high_dim(
    128, "SqDist", "SqDist(x,y)", ["x=Vi({d})", "y=Vj({d})"]
)))

# Test 2: D=128 Gaussian kernel  
results.append(("Gaussian D=128", test_high_dim(
    128, "Gaussian", "Exp(-SqNorm2(x-y))", ["x=Vi({d})", "y=Vj({d})"]
)))

# Test 3: D=256 (even more extreme: would need 192KB without fix)
results.append(("SqDist D=256", test_high_dim(
    256, "SqDist", "SqDist(x,y)", ["x=Vi({d})", "y=Vj({d})"]
)))

# Test 4: Gradient with D=128
results.append(("Gradient D=128", test_gradient(128)))

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
all_passed = True
for name, passed in results:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if not passed:
        all_passed = False

if all_passed:
    print("\n✓ All tests passed! The fix is working correctly.")
else:
    print("\n✗ Some tests failed. Check that:")
    print("  1. Cuda_link_compile.py was copied to site-packages/keopscore/binders/cuda/")
    print("  2. The build folder was cleared (old .so files contain buggy launcher)")
    print("  3. Run: rm -rf $(python -c 'from keopscore.config import config; print(config.get_build_folder())')/*")

sys.exit(0 if all_passed else 1)
