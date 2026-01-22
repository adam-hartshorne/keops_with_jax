#!/usr/bin/env python3
"""
Diagnostic script to verify the bounds checking fix is properly applied.
"""
import os
os.environ['PYKEOPS_JAX_MODE'] = '1'

import sys
import inspect

# Add the path to your pykeops installation
sys.path.insert(0, '/media/adam/shared_folder/PycharmProjects/keops/pykeops')

from pykeops.jax.generic import generic_ops

print("=" * 80)
print("DIAGNOSTIC: Checking if bounds checking fix is applied")
print("=" * 80)

# Get the source code of _compute_output_shape_cached
func = generic_ops._compute_output_shape_cached.__wrapped__  # Get unwrapped version
source = inspect.getsource(func)

print("\n1. Checking _compute_output_shape_cached function...")
print("-" * 80)

# Check for the bounds check pattern
has_num_args = "num_args = len(args_shapes)" in source
has_bounds_check = "if i < num_args" in source

print(f"   Has 'num_args = len(args_shapes)': {has_num_args}")
print(f"   Has 'if i < num_args' check: {has_bounds_check}")

if has_num_args and has_bounds_check:
    print("\n   ✓ FIX IS APPLIED!")
else:
    print("\n   ✗ FIX IS NOT APPLIED!")
    print("\n   ACTION REQUIRED:")
    print("   1. Apply the fix to generic_ops.py")
    print("   2. Restart your Python interpreter")
    print("   3. Run this diagnostic again")

print("\n" + "=" * 80)
print("Source code preview (first 50 lines):")
print("=" * 80)
for i, line in enumerate(source.split('\n')[:50], 1):
    if 'num_args' in line or 'i < num_args' in line:
        print(f"{i:3}: >>> {line}")  # Highlight important lines
    else:
        print(f"{i:3}:     {line}")

print("\n" + "=" * 80)
print("Cache information:")
print("=" * 80)
print(f"Cache size: {generic_ops._compute_output_shape_cached.cache_info().currsize}")
print(f"Cache hits: {generic_ops._compute_output_shape_cached.cache_info().hits}")
print(f"Cache misses: {generic_ops._compute_output_shape_cached.cache_info().misses}")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("=" * 80)
if not (has_num_args and has_bounds_check):
    print("The fix has NOT been applied. Please:")
    print("1. Apply the changes to generic_ops.py")
    print("2. Restart Python")
else:
    print("The fix IS applied. If you're still getting errors:")
    print("1. Clear the cache: generic_ops._compute_output_shape_cached.cache_clear()")
    print("2. Or restart your Python interpreter")
    print("3. Re-run your test")