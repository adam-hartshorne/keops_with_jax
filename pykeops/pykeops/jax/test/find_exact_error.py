#!/usr/bin/env python3
"""
Run this FIRST, before importing any KeOps modules.
This patches the function before it gets imported by LazyTensor.
"""
import os

os.environ['PYKEOPS_JAX_MODE'] = '1'

# Import JAX first (it's fine to import this early)
import jax.numpy as jnp
from typing import Tuple, List

print("=" * 80)
print("STEP 1: Importing generic_ops module DIRECTLY")
print("=" * 80)

# Import generic_ops BEFORE LazyTensor imports it
from pykeops.jax.generic import generic_ops

print("✓ generic_ops imported")

print("\n" + "=" * 80)
print("STEP 2: Applying monkey-patch to _compute_output_shape")
print("=" * 80)

# Save original
_original_compute_output_shape = generic_ops._compute_output_shape

call_count = [0]  # Use list to make it mutable in closure


def _debug_compute_output_shape(args: List[jnp.ndarray], var_cats: List[int],
                                axis: int, dimout: int, target_cat: int = None) -> Tuple[int, ...]:
    """Wrapper with debug output."""
    call_count[0] += 1

    print(f"\n{'=' * 80}")
    print(f"DEBUG _compute_output_shape CALL #{call_count[0]}")
    print(f"{'=' * 80}")
    print(f"len(args) = {len(args)}")
    print(f"len(var_cats) = {len(var_cats)}")
    print(f"var_cats = {var_cats}")
    print(f"axis = {axis}")
    print(f"dimout = {dimout}")
    print(f"target_cat = {target_cat}")

    print(f"\nArgument shapes:")
    for i, arg in enumerate(args):
        print(f"  args[{i}].shape = {arg.shape}")

    if len(args) != len(var_cats):
        print(f"\n🔥 LENGTH MISMATCH DETECTED!")
        print(f"   len(args) = {len(args)}")
        print(f"   len(var_cats) = {len(var_cats)}")
        print(f"   Difference = {len(var_cats) - len(args)}")
        print(f"\n   This means var_cats has {len(var_cats) - len(args)} extra element(s)")
        print(f"   that don't correspond to any actual arguments!")

    # Call original
    try:
        result = _original_compute_output_shape(args, var_cats, axis, dimout, target_cat)
        print(f"\n✓ Success! Result shape: {result}")
        return result
    except IndexError as e:
        print(f"\n❌ IndexError CAUGHT!")
        print(f"   Error: {e}")
        print(f"\n   The error happened when trying to access args_shapes[i][1]")
        print(f"   where i is out of bounds because len(args) < len(var_cats)")
        print(f"\n   This is the bug we're tracking!")
        raise


# Apply monkey-patch
generic_ops._compute_output_shape = _debug_compute_output_shape

print("✓ Monkey-patch applied to _compute_output_shape")
print(f"  Function identity check:")
print(f"    Original: {id(_original_compute_output_shape)}")
print(f"    Patched:  {id(generic_ops._compute_output_shape)}")
print(f"    Different? {id(_original_compute_output_shape) != id(generic_ops._compute_output_shape)}")

print("\n" + "=" * 80)
print("STEP 3: Now you can import LazyTensor and run your test")
print("=" * 80)
print("\nThe debug output will appear whenever _compute_output_shape is called.")
print("This will show us EXACTLY when and why the IndexError occurs.")
print("\n" + "=" * 80)