"""
KeOps JAX generic operations with lazy FFI registration
OPTIMIZED VERSION - January 2026

Optimizations:
1. Module-level extension import (avoid repeated import overhead)
2. Hash computation caching with LRU
3. Module-level sys.path setup (avoid repeated manipulation)
4. Pre-compiled regex patterns for alias parsing
5. Thread-safe FFI registration (Double-Checked Locking)
6. Bounded LRU cache (maxsize=4096)
7. Exit cleanup handler (atexit)
8. Fast path using is_kernel_registered check
"""

import os
import sys
import jax
import jax.numpy as jnp
import hashlib
import re
import threading
import atexit
import numpy as np
import importlib
import importlib.util
from functools import lru_cache
from typing import Tuple, List

# Debug flag
DEBUG = os.environ.get('JAX_KEOPS_DEBUG', '0') == '1'

# OPTIMIZATION: Module-level sys.path setup (do once, not per registration)
_JAX_DIR = os.path.dirname(os.path.abspath(__file__))
if _JAX_DIR not in sys.path:
    sys.path.insert(0, _JAX_DIR)

# OPTIMIZATION: Pre-compiled regex patterns for alias parsing
_ALIAS_PATTERN_1 = re.compile(r'(\w+)=(Vi|Vj|Pm)\((\d+)\)')
_ALIAS_PATTERN_2 = re.compile(r'Var\((\d+),(\d+),(\d+)\)')

# PYKEOPS_JAX_MODE is set by jax/generic/__init__.py
from keopscore.formulas import Var
import keopscore.formulas.GetReduction as GetReductionModule

# Global lock for FFI registration
_registration_lock = threading.Lock()

# OPTIMIZATION: Module-level extension import (cached)
_keops_ext_cache = None
_keops_ext_lock = threading.Lock()

def _get_keops_ext():
    """Get KeOps extension module with caching to avoid repeated imports."""
    global _keops_ext_cache

    if _keops_ext_cache is not None:
        return _keops_ext_cache

    with _keops_ext_lock:
        if _keops_ext_cache is not None:
            return _keops_ext_cache

        # Try 1: Import as pykeops.jax.keops_jax_ext (installed package)
        keops_jax_ext = None
        try:
            keops_jax_ext = importlib.import_module('pykeops.jax.keops_jax_ext')
        except ImportError:
            pass

        # Try 2: Import as keops_jax_ext (direct import)
        if keops_jax_ext is None:
            try:
                import keops_jax_ext as kext
                keops_jax_ext = kext
            except ImportError:
                pass

        # Try 3: Look for .so file in current directory
        if keops_jax_ext is None:
            so_files = [f for f in os.listdir(_JAX_DIR) if f.startswith('keops_jax_ext') and f.endswith('.so')]

            if so_files:
                so_path = os.path.join(_JAX_DIR, so_files[0])
                spec = importlib.util.spec_from_file_location("keops_jax_ext", so_path)
                if spec and spec.loader:
                    keops_jax_ext = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(keops_jax_ext)

        if keops_jax_ext is None:
            raise ImportError("KeOps JAX extension not found")

        # Register cleanup on first import
        if not hasattr(keops_jax_ext, '_atexit_registered'):
            atexit.register(keops_jax_ext.cleanup_all_kernels)
            keops_jax_ext._atexit_registered = True

        _keops_ext_cache = keops_jax_ext
        return _keops_ext_cache

# Set up KeOps variable factories
def Vj_factory(*args):
    if len(args) == 2:
        return Var(args[0], args[1], 1)
    return Var(0, args[0], 1)

def Vi_factory(*args):
    if len(args) == 2:
        return Var(args[0], args[1], 0)
    return Var(0, args[0], 0)

def Pm_factory(*args):
    if len(args) == 2:
        return Var(args[0], args[1], 2)
    return Var(0, args[0], 2)

GetReductionModule.Vi = Vi_factory
GetReductionModule.Vj = Vj_factory
GetReductionModule.Pm = Pm_factory

def _canon_alias(alias: str) -> str:
    return alias.replace(" ", "")

def _parse_aliases(aliases: List[str]) -> Tuple[List[str], List[int], List[int]]:
    """
    Parse alias strings into variable names, dimensions, and categories.
    OPTIMIZATION: Uses pre-compiled regex patterns.
    """
    var_names = []
    var_dims = []
    var_cats = []

    for i, alias in enumerate(aliases):
        alias = _canon_alias(alias)

        # OPTIMIZATION: Use pre-compiled patterns
        match = _ALIAS_PATTERN_1.match(alias)
        if match:
            name, cat_str, dim = match.groups()
            var_names.append(name)
            var_dims.append(int(dim))
            if cat_str == 'Vi': var_cats.append(0)
            elif cat_str == 'Vj': var_cats.append(1)
            elif cat_str == 'Pm': var_cats.append(2)
            continue

        match = _ALIAS_PATTERN_2.match(alias)
        if match:
            ind, dim, cat = match.groups()
            var_names.append(f"var{i}")
            var_dims.append(int(dim))
            var_cats.append(int(cat))
            continue

        raise ValueError(f"Invalid alias format: {alias}")

    return var_names, var_dims, var_cats

# OPTIMIZATION: Cache hash computation for repeated formulas
@lru_cache(maxsize=1024)
def _compute_kernel_hash(formula: str, aliases: tuple, reduction_op: str,
                         axis: int, dtype_str: str, batch_key: str, is_grad: bool = False) -> Tuple[int, str]:
    """Compute deterministic kernel ID and target name from configuration."""
    suffix = "_grad" if is_grad else ""
    config_str = f"{formula}_{aliases}_{reduction_op}_{axis}_{dtype_str}_{batch_key}{suffix}"
    kernel_hash = hashlib.md5(config_str.encode()).hexdigest()
    kernel_id = int(kernel_hash[:8], 16) % (2 ** 31)
    target_prefix = "keops_jax_grad" if is_grad else "keops"
    target_name = f"{target_prefix}_{kernel_hash[:16]}"
    return kernel_id, target_name

# OPTIMIZATION: Cache output shape computation
@lru_cache(maxsize=2048)
def _compute_output_shape_cached(args_shapes: tuple, var_cats: tuple,
                                  axis: int, dimout: int, target_cat: int = None) -> Tuple[int, ...]:
    """Cached version of output shape computation."""
    first_shape = args_shapes[0]
    ndims = len(first_shape)

    if ndims == 3:
        batch_size = first_shape[0]
        if target_cat is not None:
            for i, cat in enumerate(var_cats):
                if cat == target_cat:
                    return (batch_size, args_shapes[i][1], dimout)
            return (batch_size, args_shapes[0][1], dimout)

        if axis == 0:
            for i, cat in enumerate(var_cats):
                if cat == 1: return (batch_size, args_shapes[i][1], dimout)
            return (batch_size, args_shapes[0][1], dimout)
        else:
            for i, cat in enumerate(var_cats):
                if cat == 0: return (batch_size, args_shapes[i][1], dimout)
            return (batch_size, args_shapes[0][1], dimout)

    if target_cat is not None:
        for i, cat in enumerate(var_cats):
            if cat == target_cat: return (args_shapes[i][0], dimout)
        return (args_shapes[0][0], dimout)

    if axis == 0:
        for i, cat in enumerate(var_cats):
            if cat == 1: return (args_shapes[i][0], dimout)
        return (args_shapes[0][0], dimout)
    else:
        for i, cat in enumerate(var_cats):
            if cat == 0: return (args_shapes[i][0], dimout)
        return (args_shapes[0][0], dimout)

def _compute_max_scratch_size(nx, ny, batch_size, nvi, nvj, nvp, indsi, indsj, indsp):
    """
    Compute conservative upper bound for XLA scratch memory.

    This calculates the maximum scratch space needed for the CUDA launcher
    to store temporary data (offsets, lookup tables, argument pointers).

    Args:
        nx: N dimension
        ny: M dimension
        batch_size: Batch size (1 for non-batched)
        nvi, nvj, nvp: Number of variables per category
        indsi, indsj, indsp: Variable indices

    Returns:
        Maximum scratch bytes needed (with 20% safety margin)
    """
    cuda_block_size = 1024  # Default CUDA block size
    blocks_per_batch = (nx + cuda_block_size - 1) // cuda_block_size
    nblocks = blocks_per_batch * batch_size

    total_offsets = nvi + nvj + nvp
    if total_offsets == 0:
        total_offsets = 2

    # Calculate max variable index
    max_var_idx = -1
    if indsi:
        max_var_idx = max(max_var_idx, max(indsi))
    if indsj:
        max_var_idx = max(max_var_idx, max(indsj))
    if indsp:
        max_var_idx = max(max_var_idx, max(indsp))
    sparse_args_count = max_var_idx + 1 if max_var_idx >= 0 else 0

    # Calculate sizes (sizeof(long) = 8, sizeof(float*) = 8)
    size_offsets = 8 * nblocks * total_offsets
    size_lookup = 8 * 3 * nblocks
    size_slices = 8 * batch_size
    size_ranges = 8 * 2 * batch_size
    size_args = 8 * max(sparse_args_count, nvi + nvj + nvp)  # Take max for safety

    total = size_offsets + size_lookup + size_slices + size_ranges + size_args

    # Add 20% safety margin and round up to 256-byte boundary for alignment
    total_with_margin = int(total * 1.2)
    aligned_total = ((total_with_margin + 255) // 256) * 256

    return aligned_total

def _compute_output_shape(args: List[jnp.ndarray], var_cats: List[int],
                          axis: int, dimout: int, target_cat: int = None) -> Tuple[int, ...]:
    """Wrapper that converts args to shapes and calls cached version."""
    args_shapes = tuple(arg.shape for arg in args)
    return _compute_output_shape_cached(args_shapes, tuple(var_cats), axis, dimout, target_cat)

def _compute_arg_order(var_cats: List[int]) -> List[int]:
    return list(range(len(var_cats)))

def _make_keops_grad_op(formula: str, aliases: Tuple[str, ...],
                        reduction_op: str, axis: int, dtype_str: str,
                        expected_dimout: int, target_cat: int):
    _patch_nvcc_flags()
    var_names, var_dims, var_cats = _parse_aliases(list(aliases))

    # Create state dict to track registrations per batch mode
    kernel_cache = {}

    def grad_op_impl(*args):
        jax_args = list(args)

        # Determine batch mode
        first_shape = jax_args[0].shape
        is_batched = len(first_shape) == 3
        batch_key = "3d" if is_batched else "2d"

        # OPTIMIZATION: Use cached hash computation
        if batch_key not in kernel_cache:
            kernel_id, target_name = _compute_kernel_hash(
                formula, aliases, reduction_op, axis, dtype_str, batch_key, is_grad=True
            )

            kernel_cache[batch_key] = {
                'kernel_id': kernel_id,
                'target_name': target_name,
                'registered': False
            }

        ffi_state = kernel_cache[batch_key]
        kernel_id = ffi_state['kernel_id']
        target_name = ffi_state['target_name']

        if not ffi_state['registered']:
            with _registration_lock:
                if not ffi_state['registered']:
                    # OPTIMIZATION: Use cached extension getter
                    keops_jax_ext = _get_keops_ext()

                    if hasattr(keops_jax_ext, 'is_kernel_registered') and \
                       keops_jax_ext.is_kernel_registered(kernel_id):
                        ffi_state['registered'] = True
                    else:
                        myconv_orig = _create_keops_backend(
                            formula, list(aliases), reduction_op, axis, dtype_str,
                            jax_args, use_ranges=is_batched
                        )
                        keops_jax_ext.register_keops_kernel(kernel_id, myconv_orig)
                        jax.ffi.register_ffi_target(target_name, keops_jax_ext.get_ffi_handler(), platform="CUDA")
                        ffi_state['registered'] = True

        reordered_args = jax_args
        first_shape = jax_args[0].shape
        batch_size = int(first_shape[0]) if len(first_shape) == 3 else 1
        output_shape = _compute_output_shape(jax_args, var_cats, axis, expected_dimout, target_cat=target_cat)
        result = jax.ShapeDtypeStruct(shape=output_shape, dtype=jax_args[0].dtype)

        # NEW: Compute scratch size for gradient op
        nx = 1
        ny = 1
        for i, cat in enumerate(var_cats):
            if cat == 0 and len(jax_args[i].shape) >= 2:
                nx = int(jax_args[i].shape[-2]) if len(jax_args[i].shape) == 3 else int(jax_args[i].shape[0])
            elif cat == 1 and len(jax_args[i].shape) >= 2:
                ny = int(jax_args[i].shape[-2]) if len(jax_args[i].shape) == 3 else int(jax_args[i].shape[0])

        max_scratch_bytes = _compute_max_scratch_size(
            nx=nx, ny=ny, batch_size=batch_size,
            nvi=len([c for c in var_cats if c == 0]),
            nvj=len([c for c in var_cats if c == 1]),
            nvp=len([c for c in var_cats if c == 2]),
            indsi=list(range(len([c for c in var_cats if c == 0]))),
            indsj=list(range(len([c for c in var_cats if c == 1]))),
            indsp=list(range(len([c for c in var_cats if c == 2])))
        )

        output = jax.ffi.ffi_call(
            target_name,
            result,
            vmap_method="broadcast_all",
            has_side_effect=False
        )(*reordered_args, kernel_id=int(kernel_id), batch_size=int(batch_size),
          max_scratch_bytes=int(max_scratch_bytes))  # NEW

        return output

    return grad_op_impl

def _patch_nvcc_flags():
    os.environ["PYKEOPS_JAX_MODE"] = "1"

def _create_keops_backend(formula, aliases, reduction_op, axis, dtype_str, jax_args, apply_axis_flip=True, use_ranges=False):
    from pykeops.common.keops_io import keops_binder
    from pykeops.common.get_options import get_tag_backend

    var_names, var_dims, var_cats = _parse_aliases(aliases)

    cat = (axis + 1) % 2 if apply_axis_flip else axis
    reduction_formula_str = f"{reduction_op}_Reduction({formula},{cat})"
    tagCPUGPU, tag1D2D, tagHostDevice = get_tag_backend("GPU", jax_args)

    return keops_binder["cpp"](
        tagCPUGPU, tag1D2D, tagHostDevice,
        use_ranges,
        -1,
        reduction_formula_str, aliases, len(jax_args), dtype_str, "jax",
        {
            'dtype_acc': dtype_str,
            'sum_scheme': 'block_sum',
            'enable_chunks': False,
            'use_fast_math': True,
            'multVar_highdim': False
        }
    ).import_module()

def make_keops_jax_op(formula: str, aliases: Tuple[str, ...], reduction_op: str, axis: int, dtype_str: str, enable_vjp: bool = True):
    """
    Creates a JAX op for KeOps kernel with optimized caching.
    """
    _patch_nvcc_flags()
    var_names, var_dims, var_cats = _parse_aliases(list(aliases))

    # Create state dict to track registrations per batch mode
    kernel_cache = {}

    def keops_jax_op_impl(*args):
        # Determine if we're in batched mode
        first_shape = args[0].shape
        is_batched = len(first_shape) == 3
        batch_key = "3d" if is_batched else "2d"

        # OPTIMIZATION: Use cached hash computation
        if batch_key not in kernel_cache:
            kernel_id, target_name = _compute_kernel_hash(
                formula, aliases, reduction_op, axis, dtype_str, batch_key
            )

            kernel_cache[batch_key] = {
                'kernel_id': kernel_id,
                'target_name': target_name,
                'registered': False,
                'dimout': 1
            }

        ffi_state = kernel_cache[batch_key]
        kernel_id = ffi_state['kernel_id']
        target_name = ffi_state['target_name']

        def ffi_wrapper(*jax_args):
            if not ffi_state['registered']:
                with _registration_lock:
                    if not ffi_state['registered']:
                        # OPTIMIZATION: Use cached extension getter
                        keops_jax_ext = _get_keops_ext()

                        if hasattr(keops_jax_ext, 'is_kernel_registered') and \
                           keops_jax_ext.is_kernel_registered(kernel_id):
                            ffi_state['registered'] = True
                        else:
                            is_batched = len(jax_args[0].shape) == 3
                            myconv_orig = _create_keops_backend(
                                formula, list(aliases), reduction_op, axis, dtype_str,
                                jax_args, use_ranges=is_batched
                            )

                            # Extract correct dimout from compiled kernel
                            if hasattr(myconv_orig, 'dim'):
                                ffi_state['dimout'] = myconv_orig.dim
                            elif hasattr(myconv_orig, 'dimout'):
                                ffi_state['dimout'] = myconv_orig.dimout

                            keops_jax_ext.register_keops_kernel(kernel_id, myconv_orig)
                            jax.ffi.register_ffi_target(target_name, keops_jax_ext.get_ffi_handler(), platform="CUDA")
                            ffi_state['registered'] = True

            reordered_args = jax_args
            dimout = ffi_state['dimout']
            first_shape = jax_args[0].shape
            batch_size = int(first_shape[0]) if len(first_shape) == 3 else 1

            # NEW: Compute scratch size needed for XLA allocator
            # Extract nx, ny from input shapes
            nx = 1
            ny = 1
            for i, cat in enumerate(var_cats):
                if cat == 0 and len(jax_args[i].shape) >= 2:  # Vi variable
                    nx = int(jax_args[i].shape[-2]) if len(jax_args[i].shape) == 3 else int(jax_args[i].shape[0])
                elif cat == 1 and len(jax_args[i].shape) >= 2:  # Vj variable
                    ny = int(jax_args[i].shape[-2]) if len(jax_args[i].shape) == 3 else int(jax_args[i].shape[0])

            # Get variable indices from kernel registration (need to access myconv_orig params)
            # For now, use conservative estimate
            max_scratch_bytes = _compute_max_scratch_size(
                nx=nx,
                ny=ny,
                batch_size=batch_size,
                nvi=len([c for c in var_cats if c == 0]),
                nvj=len([c for c in var_cats if c == 1]),
                nvp=len([c for c in var_cats if c == 2]),
                indsi=list(range(len([c for c in var_cats if c == 0]))),
                indsj=list(range(len([c for c in var_cats if c == 1]))),
                indsp=list(range(len([c for c in var_cats if c == 2])))
            )

            # OPTIMIZATION: Use cached shape computation
            output_shape = _compute_output_shape(jax_args, var_cats, axis, dimout)
            result = jax.ShapeDtypeStruct(shape=output_shape, dtype=jax_args[0].dtype)

            output = jax.ffi.ffi_call(
                target_name,
                result,
                vmap_method="broadcast_all",
                has_side_effect=False
            )(*reordered_args, kernel_id=int(kernel_id), batch_size=int(batch_size),
              max_scratch_bytes=int(max_scratch_bytes))  # NEW: Pass scratch size

            return output
        return ffi_wrapper(*args)

    if not enable_vjp: return keops_jax_op_impl

    @jax.custom_vjp
    def keops_jax_op(*args):
        return keops_jax_op_impl(*args)

    def keops_jax_op_fwd(*args):
        return keops_jax_op_impl(*args), args

    def keops_jax_op_bwd(residuals, g):
        args = residuals
        forward_output_cat = 0 if axis == 1 else 1
        grads = []

        for i, var_cat in enumerate(var_cats):
            var_name = var_names[i]
            input_shape = args[i].shape
            input_dim = var_dims[i]

            inner_formula = formula
            for red in ["Sum", "Max", "Min", "LogSumExp"]:
                if formula.startswith(f"{red}(") and formula.endswith(")"):
                    inner_formula = formula[len(red)+1:-1]
                    break

            current_alias = aliases[i]
            var_ref = current_alias if current_alias.startswith('Var(') else var_name
            grad_formula = f"Grad({inner_formula}, {var_ref}, eta)"

            grad_axis = 1 if var_cat != 1 else 0
            grad_eta_cat = forward_output_cat

            grad_aliases = list(aliases)
            cat_str = {0: "Vi", 1: "Vj", 2: "Pm"}[grad_eta_cat]
            grad_aliases.append(f"eta={cat_str}({g.shape[-1]})")

            grad_op = _make_keops_grad_op(grad_formula, tuple(grad_aliases), reduction_op,
                                          grad_axis, dtype_str, input_dim, var_cat)

            grad_args = list(args) + [g]
            grad_i = grad_op(*grad_args)

            if var_cat != 2:
                 zero_shim = jnp.array(0, dtype=args[i].dtype)
                 grad_i = grad_i + (zero_shim * args[i])

            if var_cat == 2:
                grad_i = jnp.sum(grad_i, axis=0, keepdims=True).reshape(input_shape)

            grads.append(grad_i)

        return tuple(grads)

    keops_jax_op.defvjp(keops_jax_op_fwd, keops_jax_op_bwd)
    return keops_jax_op

def keops_reduction(formula, aliases, reduction_op='Sum', axis=0, dtype='float32', enable_vjp=True):
    return make_keops_jax_op(formula, aliases, reduction_op, axis, dtype, enable_vjp)

def cleanup_registry():
    pass

def get_registry_info():
    return {'note': 'Thread-safe registration with optimized caching enabled'}