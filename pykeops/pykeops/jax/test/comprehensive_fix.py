#!/usr/bin/env python3
"""
Comprehensive fix script for KeOps JAX gradient bug.
This script:
1. Finds and removes .pyc bytecode files
2. Clears function caches
3. Verifies the fix is applied
"""

import os
os.environ['PYKEOPS_JAX_MODE'] = '1'

import sys
from pathlib import Path

print("=" * 80)
print("KEOPS JAX COMPREHENSIVE FIX SCRIPT")
print("=" * 80)

# Step 1: Find pykeops installation
print("\n1. Finding pykeops installation...")
try:
    import pykeops

    pykeops_path = Path(pykeops.__file__).parent
    print(f"   Found: {pykeops_path}")
except ImportError:
    print("   ERROR: Could not import pykeops!")
    sys.exit(1)

# Step 2: Remove .pyc files
print("\n2. Removing compiled bytecode (.pyc) files...")
pyc_files = list(pykeops_path.rglob("*.pyc"))
pycache_dirs = list(pykeops_path.rglob("__pycache__"))

for pyc in pyc_files:
    try:
        pyc.unlink()
        print(f"   Removed: {pyc.name}")
    except Exception as e:
        print(f"   Warning: Could not remove {pyc}: {e}")

for pycache in pycache_dirs:
    try:
        import shutil

        shutil.rmtree(pycache)
        print(f"   Removed directory: {pycache}")
    except Exception as e:
        print(f"   Warning: Could not remove {pycache}: {e}")

print(f"   Cleaned up {len(pyc_files)} .pyc files and {len(pycache_dirs)} __pycache__ directories")

# Step 3: Clear function caches
print("\n3. Clearing function caches...")
try:
    from pykeops.jax.generic import generic_ops

    if hasattr(generic_ops, '_compute_output_shape_cached'):
        generic_ops._compute_output_shape_cached.cache_clear()
        print("   ✓ Cleared _compute_output_shape_cached cache")

    if hasattr(generic_ops, '_compute_kernel_hash'):
        generic_ops._compute_kernel_hash.cache_clear()
        print("   ✓ Cleared _compute_kernel_hash cache")
except Exception as e:
    print(f"   Warning: {e}")

# Step 4: Verify the fix
print("\n4. Verifying the fix is applied...")
generic_ops_file = pykeops_path / "jax" / "generic" / "generic_ops.py"
print(f"   Checking: {generic_ops_file}")

if not generic_ops_file.exists():
    print(f"   ERROR: File not found!")
    sys.exit(1)

with open(generic_ops_file, 'r') as f:
    content = f.read()

has_num_args = "num_args = len(args_shapes)" in content
has_bounds_check = "if i < num_args and cat ==" in content

print(f"   Has 'num_args = len(args_shapes)': {has_num_args}")
print(f"   Has 'if i < num_args' bounds check: {has_bounds_check}")

if has_num_args and has_bounds_check:
    print("\n   ✓✓✓ FIX IS APPLIED! ✓✓✓")
else:
    print("\n   ✗✗✗ FIX IS NOT APPLIED! ✗✗✗")
    print("\n   ACTION REQUIRED:")
    print(f"   1. Replace {generic_ops_file}")
    print(f"      with the fixed version (generic_ops_fixed.py)")
    print(f"   2. Run this script again")
    sys.exit(1)

print("\n" + "=" * 80)
print("ALL CHECKS PASSED!")
print("=" * 80)
print("\nNow you can run your test. The fix should work properly.")
print("\nIf you still get errors:")
print("1. Make sure you RESTART your Python interpreter")
print("2. Or run: python3 -B your_test.py (the -B flag skips .pyc files)")