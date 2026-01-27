"""
KeOps JAX generic operations - FULLY OPTIMIZED

FIXES APPLIED:
1. Use C++ get_kernel_dimout to avoid redundant Python backend creation
2. Bounded gradient cache with LRU eviction
3. Thread-safe registration with proper error handling
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
from collections import OrderedDict

# Debug flag
DEBUG = os.environ.get('JAX_KEOPS_DEBUG', '0') == '1'

# Module-level sys.path setup
_JAX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Parent dir where .so lives
if _JAX_DIR not in sys.path:
    sys.path.insert(0, _JAX_DIR)

# Pre-compiled regex patterns
_ALIAS_PATTERN_1 = re.compile(r'(\w+)=(Vi|Vj|Pm)\((\d+)\)')
_ALIAS_PATTERN_2 = re.compile(r'Var\((\d+),(\d+),(\d+)\)')

from keopscore.formulas import Var
import keopscore.formulas.GetReduction as GetReductionModule

# Global lock for FFI registration
_registration_lock = threading.Lock()

# Track which FFI targets have been registered (JAX side)
# This is separate from C++ kernel registry
_registered_ffi_targets = set()

# Module-level extension cache
_keops_ext_cache = None
_keops_ext_lock = threading.Lock()


class BoundedLRUCache:
    """Thread-safe LRU cache with bounded size for gradient ops."""

    def __init__(self, maxsize=256):
        self.maxsize = maxsize
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def set(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.maxsize:
                    # Remove oldest item
                    self.cache.popitem(last=False)
            # Always set the value (both for new keys and updates to existing keys)
            self.cache[key] = value

    def clear(self):
        with self.lock:
            self.cache.clear()


# Global gradient cache with bounded size
_grad_cache = BoundedLRUCache(maxsize=512)


def _get_keops_ext():
    """Get KeOps extension module with caching."""
    global _keops_ext_cache

    if _keops_ext_cache is not None:
        return _keops_ext_cache

    with _keops_ext_lock:
        if _keops_ext_cache is not None:
            return _keops_ext_cache

        keops_jax_ext = None
        import_errors = []

        # Try: Import as pykeops.jax.keops_jax_ext
        try:
            keops_jax_ext = importlib.import_module('pykeops.jax.keops_jax_ext')
        except ImportError as e:
            import_errors.append(f"  1. import pykeops.jax.keops_jax_ext: {e}")

        # Try: Import as keops_jax_ext
        if keops_jax_ext is None:
            try:
                import keops_jax_ext as kext
                keops_jax_ext = kext
            except ImportError as e:
                import_errors.append(f"  2. import keops_jax_ext: {e}")

        # Try: Look for .so file in current directory
        if keops_jax_ext is None:
            so_files = [f for f in os.listdir(_JAX_DIR) if f.startswith('keops_jax_ext') and f.endswith('.so')]

            if so_files:
                so_path = os.path.join(_JAX_DIR, so_files[0])
                try:
                    spec = importlib.util.spec_from_file_location("keops_jax_ext", so_path)
                    if spec and spec.loader:
                        keops_jax_ext = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(keops_jax_ext)
                except Exception as e:
                    import_errors.append(f"  3. load {so_path}: {e}")
            else:
                import_errors.append(f"  3. No .so files found in {_JAX_DIR}")

        if keops_jax_ext is None:
            error_details = "\n".join(import_errors)
            raise ImportError(
                f"KeOps JAX extension not found. Tried:\n{error_details}\n\n"
                f"To build the extension, reinstall pykeops with JAX dependencies:\n"
                f"  pip install -e . --no-build-isolation\n"
                f"Or ensure jax, jaxlib, nanobind, cmake, and nvcc are available."
            )

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
    """Parse alias strings into variable names, dimensions, and categories."""
    var_names = []
    var_dims = []
    var_cats = []

    for i, alias in enumerate(aliases):
        alias = _canon_alias(alias)

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


@lru_cache(maxsize=1024)
def _compute_kernel_hash(formula: str, aliases: tuple, reduction_op: str,
                         axis: int, dtype_str: str, batch_key: str, is_grad: bool = False) -> Tuple[int, str]:
    """Compute deterministic kernel ID and target name."""
    suffix = "_grad" if is_grad else ""
    config_str = f"{formula}_{aliases}_{reduction_op}_{axis}_{dtype_str}_{batch_key}{suffix}"
    kernel_hash = hashlib.md5(config_str.encode()).hexdigest()
    kernel_id = int(kernel_hash[:8], 16) % (2 ** 31)
    target_prefix = "keops_jax_grad" if is_grad else "keops"
    target_name = f"{target_prefix}_{kernel_hash[:16]}"
    return kernel_id, target_name


@lru_cache(maxsize=2048)
def _compute_output_shape_cached(args_shapes: tuple, var_cats: tuple,
                                  axis: int, dimout: int, target_cat: int = None) -> Tuple[int, ...]:
    """Cached version of output shape computation."""
    if not args_shapes:
        raise ValueError(
            "[KeOps JAX] No input shapes provided for output shape computation. "
            "Ensure at least one input argument is passed to the operator."
        )

    if dimout <= 0:
        raise ValueError(
            f"[KeOps JAX] Invalid output dimension: {dimout}. "
            "This usually indicates the formula failed to parse correctly. "
            "Check formula syntax and variable dimensions."
        )

    first_shape = args_shapes[0]
    ndims = len(first_shape)
    num_args = len(args_shapes)

    if ndims == 3:
        batch_size = first_shape[0]
        # For Pm (parameter) gradients, we should NOT use target_cat to determine
        # the output size. The kernel still outputs over the non-reduced dimension.
        if target_cat is not None and target_cat != 2:
            # For Vi/Vj gradients, find the first variable with matching category
            for i, cat in enumerate(var_cats):
                if i < num_args and cat == target_cat:
                    if len(args_shapes[i]) < 2:
                        continue
                    return (batch_size, args_shapes[i][1], dimout)
            return (batch_size, args_shapes[0][1], dimout)

        # For Pm gradients (target_cat=2) or forward pass, use axis-based logic
        if axis == 0:
            for i, cat in enumerate(var_cats):
                if i < num_args and cat == 1:
                    if len(args_shapes[i]) < 2:
                        continue
                    return (batch_size, args_shapes[i][1], dimout)
            return (batch_size, args_shapes[0][1], dimout)
        else:
            for i, cat in enumerate(var_cats):
                if i < num_args and cat == 0:
                    if len(args_shapes[i]) < 2:
                        continue
                    return (batch_size, args_shapes[i][1], dimout)
            return (batch_size, args_shapes[0][1], dimout)
    else:
        # For Pm (parameter) gradients, we should NOT use target_cat to determine
        # the output size. The kernel still outputs over the non-reduced dimension
        # (Vi for axis=1, Vj for axis=0). The VJP code will sum this to get the
        # final Pm gradient.
        if target_cat is not None and target_cat != 2:
            # For Vi/Vj gradients, find the first variable with matching category
            for i, cat in enumerate(var_cats):
                if i < num_args and cat == target_cat:
                    if len(args_shapes[i]) < 1:
                        continue
                    return (args_shapes[i][0], dimout)
            return (args_shapes[0][0], dimout)

        # For Pm gradients (target_cat=2) or forward pass, use axis-based logic
        if axis == 0:
            for i, cat in enumerate(var_cats):
                if i < num_args and cat == 1:
                    if len(args_shapes[i]) < 1:
                        continue
                    return (args_shapes[i][0], dimout)
            return (args_shapes[0][0], dimout)
        else:
            for i, cat in enumerate(var_cats):
                if i < num_args and cat == 0:
                    if len(args_shapes[i]) < 1:
                        continue
                    return (args_shapes[i][0], dimout)
            return (args_shapes[0][0], dimout)


def _compute_output_shape(args, var_cats, axis, dimout, target_cat=None):
    """Wrapper for cached shape computation."""
    args_shapes = tuple(tuple(arg.shape) for arg in args)
    return _compute_output_shape_cached(args_shapes, tuple(var_cats), axis, dimout, target_cat)


def _make_keops_grad_op(grad_formula, grad_aliases, reduction_op, grad_axis, dtype_str, input_dim, var_cat, enable_vjp=True):
    """Create gradient operator."""

    def grad_op_impl(*args):
        first_shape = args[0].shape
        is_batched = len(first_shape) == 3
        batch_key = "3d" if is_batched else "2d"

        kernel_id, target_name = _compute_kernel_hash(
            grad_formula, grad_aliases, reduction_op, grad_axis, dtype_str, batch_key, is_grad=True
        )

        def ffi_wrapper(*jax_args):
            keops_jax_ext = _get_keops_ext()

            # Check both C++ registry AND FFI target registry
            cpp_registered = keops_jax_ext.is_kernel_registered(kernel_id)
            ffi_registered = target_name in _registered_ffi_targets

            if not cpp_registered or not ffi_registered:
                with _registration_lock:
                    # Re-check under lock
                    cpp_registered = keops_jax_ext.is_kernel_registered(kernel_id)
                    ffi_registered = target_name in _registered_ffi_targets

                    # Register C++ kernel if needed
                    if not cpp_registered:
                        try:
                            is_batched_inner = len(jax_args[0].shape) == 3
                            myconv_grad = _create_keops_backend(
                                grad_formula, list(grad_aliases), reduction_op, grad_axis, dtype_str,
                                jax_args, use_ranges=is_batched_inner
                            )
                            keops_jax_ext.register_keops_kernel(kernel_id, myconv_grad)
                        except Exception as e:
                            raise RuntimeError(
                                f"[KeOps JAX] Gradient C++ kernel registration failed.\n"
                                f"  Gradient formula: {grad_formula}\n"
                                f"  Gradient aliases: {grad_aliases}\n"
                                f"  Original error: {e}"
                            ) from e

                    # Register FFI target if needed (separate step)
                    if not ffi_registered:
                        try:
                            jax.ffi.register_ffi_target(target_name, keops_jax_ext.get_ffi_handler(), platform="CUDA")
                            _registered_ffi_targets.add(target_name)
                        except Exception as e:
                            # Check if it's already registered (idempotent)
                            if "already registered" in str(e).lower():
                                _registered_ffi_targets.add(target_name)
                            else:
                                raise RuntimeError(
                                    f"[KeOps JAX] Gradient FFI target registration failed.\n"
                                    f"  Target: {target_name}\n"
                                    f"  Original error: {e}"
                                ) from e

            reordered_args = jax_args
            dimout = input_dim
            var_names_grad, var_dims_grad, var_cats_grad = _parse_aliases(list(grad_aliases))

            first_shape = jax_args[0].shape
            batch_size = int(first_shape[0]) if len(first_shape) == 3 else 1

            output_shape = _compute_output_shape(jax_args, var_cats_grad, grad_axis, dimout, target_cat=var_cat)
            result = jax.ShapeDtypeStruct(shape=output_shape, dtype=jax_args[0].dtype)

            output = jax.ffi.ffi_call(
                target_name,
                result,
                vmap_method="broadcast_all",
                has_side_effect=False
            )(*reordered_args, kernel_id=int(kernel_id), batch_size=int(batch_size))

            return output

        return ffi_wrapper(*args)

    if not enable_vjp:
        return grad_op_impl

    @jax.custom_vjp
    def grad_op_with_vjp(*args):
        return grad_op_impl(*args)

    def grad_op_with_vjp_fwd(*args):
        output = grad_op_impl(*args)
        return output, args

    def grad_op_with_vjp_bwd(residuals, g):
        args = residuals
        args_stopped = tuple(jax.lax.stop_gradient(a) for a in args)
        return tuple(jnp.zeros_like(a) for a in args_stopped)

    grad_op_with_vjp.defvjp(grad_op_with_vjp_fwd, grad_op_with_vjp_bwd)

    return grad_op_with_vjp


def _patch_nvcc_flags():
    os.environ["PYKEOPS_JAX_MODE"] = "1"


def _create_keops_backend(formula, aliases, reduction_op, axis, dtype_str, jax_args, apply_axis_flip=True, use_ranges=False, enable_chunks=True, opt_arg=None, formula2=None):
    from pykeops.common.keops_io import keops_binder
    from pykeops.common.get_options import get_tag_backend

    var_names, var_dims, var_cats = _parse_aliases(aliases)

    cat = (axis + 1) % 2 if apply_axis_flip else axis
    
    # Build the reduction formula string
    # Format: ReductionOp_Reduction(formula[,opt_arg],cat[,formula2])
    str_opt_arg = f",{opt_arg}" if opt_arg is not None else ""
    str_formula2 = f",{formula2}" if formula2 else ""
    reduction_formula_str = f"{reduction_op}_Reduction({formula}{str_opt_arg},{cat}{str_formula2})"
    
    tagCPUGPU, tag1D2D, tagHostDevice = get_tag_backend("GPU", jax_args)
    
    # Determine sum_scheme based on reduction type
    # block_sum is only valid for sum-type reductions
    sum_type_reductions = ("Sum", "Max_SumShiftExp", "Max_SumShiftExpWeight")
    if reduction_op in sum_type_reductions:
        sum_scheme = 'block_sum'
    else:
        sum_scheme = 'direct_sum'

    return keops_binder["cpp"](
        tagCPUGPU, tag1D2D, tagHostDevice,
        use_ranges,
        -1,
        reduction_formula_str, aliases, len(jax_args), dtype_str, "jax",
        {
            'dtype_acc': dtype_str if reduction_op in sum_type_reductions else 'auto',
            'sum_scheme': sum_scheme,
            'enable_chunks': enable_chunks,
            'use_fast_math': True,
            'multVar_highdim': False
        }
    ).import_module()


def make_keops_jax_op(formula: str, aliases: Tuple[str, ...], reduction_op: str, axis: int, dtype_str: str, enable_vjp: bool = True, max_order: int = 2, opt_arg: int = None, formula2: str = None):
    """Creates a JAX op for KeOps kernel with FULLY OPTIMIZED gradient computation."""
    _patch_nvcc_flags()
    var_names, var_dims, var_cats = _parse_aliases(list(aliases))

    kernel_cache = {}

    precomputed_grads = []
    if enable_vjp:
        forward_output_cat = 0 if axis == 1 else 1
        inner_formula = formula
        for red in ["Sum", "Max", "Min", "LogSumExp", "Max_SumShiftExp", "Max_SumShiftExpWeight"]:
            if formula.startswith(f"{red}(") and formula.endswith(")"):
                inner_formula = formula[len(red)+1:-1]
                break

        for i, var_cat in enumerate(var_cats):
            var_name = var_names[i]
            var_dim = var_dims[i]
            current_alias = aliases[i]
            var_ref = current_alias if current_alias.startswith('Var(') else var_name
            grad_formula = f"Grad({inner_formula}, {var_ref}, eta)"
            grad_axis = 1 if var_cat != 1 else 0
            grad_eta_cat = forward_output_cat

            precomputed_grads.append({
                'grad_formula': grad_formula,
                'grad_aliases_base': list(aliases),
                'grad_axis': grad_axis,
                'grad_eta_cat': grad_eta_cat,
                'var_cat': var_cat,
                'input_dim': var_dim,
                'var_name': var_name
            })

    def keops_jax_op_impl(*args):
        first_shape = args[0].shape
        is_batched = len(first_shape) == 3
        batch_key = "3d" if is_batched else "2d"

        if batch_key not in kernel_cache:
            # Include opt_arg and formula2 in hash for uniqueness
            opt_arg_str = str(opt_arg) if opt_arg is not None else ""
            formula2_str = formula2 if formula2 else ""
            extended_batch_key = f"{batch_key}_{opt_arg_str}_{formula2_str}"

            kernel_id, target_name = _compute_kernel_hash(
                formula, aliases, reduction_op, axis, dtype_str, extended_batch_key
            )

            kernel_cache[batch_key] = {
                'kernel_id': kernel_id,
                'target_name': target_name,
                'registered': False,
                'dimout': None,
            }

        ffi_state = kernel_cache[batch_key]
        kernel_id = ffi_state['kernel_id']
        target_name = ffi_state['target_name']

        def ffi_wrapper(*jax_args):
            dimout = ffi_state.get('dimout')

            if not ffi_state['registered']:
                with _registration_lock:
                    if not ffi_state['registered']:
                        keops_jax_ext = _get_keops_ext()
                        is_batched_inner = len(jax_args[0].shape) == 3

                        already_registered = (
                            hasattr(keops_jax_ext, 'is_kernel_registered') and
                            keops_jax_ext.is_kernel_registered(kernel_id)
                        )

                        # Check if FFI target is already registered (separate from C++ registry)
                        ffi_target_registered = ffi_state.get('ffi_target_registered', False)

                        # OPTIMIZATION: Try to get dimout from C++ first
                        if already_registered and hasattr(keops_jax_ext, 'get_kernel_dimout'):
                            cached_dimout = keops_jax_ext.get_kernel_dimout(kernel_id)
                            if cached_dimout > 0:
                                dimout = cached_dimout
                                ffi_state['dimout'] = dimout
                                # Only skip if FFI target is also registered
                                if ffi_target_registered:
                                    ffi_state['registered'] = True

                        if not ffi_state['registered']:
                            # Need to create backend (either for registration or to get dimout)
                            if dimout is None:
                                myconv_orig = _create_keops_backend(
                                    formula, list(aliases), reduction_op, axis, dtype_str,
                                    jax_args, use_ranges=is_batched_inner,
                                    opt_arg=opt_arg, formula2=formula2
                                )
                                dimout = getattr(myconv_orig, 'dim', getattr(myconv_orig, 'dimout', 1))
                                ffi_state['dimout'] = dimout
                            else:
                                myconv_orig = None

                            # Register C++ kernel if needed
                            if not already_registered:
                                if myconv_orig is None:
                                    myconv_orig = _create_keops_backend(
                                        formula, list(aliases), reduction_op, axis, dtype_str,
                                        jax_args, use_ranges=is_batched_inner,
                                        opt_arg=opt_arg, formula2=formula2
                                    )
                                try:
                                    keops_jax_ext.register_keops_kernel(kernel_id, myconv_orig)
                                except Exception as e:
                                    raise RuntimeError(
                                        f"[KeOps JAX] C++ kernel registration failed.\n"
                                        f"  Formula: {formula}\n"
                                        f"  Aliases: {aliases}\n"
                                        f"  Original error: {e}"
                                    ) from e

                            # Register FFI target if needed (separate step)
                            if not ffi_target_registered:
                                try:
                                    jax.ffi.register_ffi_target(
                                        target_name,
                                        keops_jax_ext.get_ffi_handler(),
                                        platform="CUDA"
                                    )
                                    ffi_state['ffi_target_registered'] = True
                                except Exception as e:
                                    # Check if it's already registered (idempotent)
                                    if "already registered" in str(e).lower():
                                        ffi_state['ffi_target_registered'] = True
                                    else:
                                        raise RuntimeError(
                                            f"[KeOps JAX] FFI target registration failed.\n"
                                            f"  Target: {target_name}\n"
                                            f"  Original error: {e}"
                                        ) from e

                            ffi_state['registered'] = True

            if dimout is None:
                dimout = ffi_state.get('dimout', 1)

            first_shape = jax_args[0].shape
            batch_size = int(first_shape[0]) if len(first_shape) == 3 else 1

            output_shape = _compute_output_shape(jax_args, var_cats, axis, dimout)
            result = jax.ShapeDtypeStruct(shape=output_shape, dtype=jax_args[0].dtype)

            output = jax.ffi.ffi_call(
                target_name,
                result,
                vmap_method="broadcast_all",
                has_side_effect=False
            )(*jax_args,
              kernel_id=int(kernel_id),
              batch_size=int(batch_size))

            return output

        return ffi_wrapper(*args)

    if not enable_vjp:
        return keops_jax_op_impl

    @jax.custom_vjp
    def keops_jax_op(*args):
        return keops_jax_op_impl(*args)

    def keops_jax_op_fwd(*args):
        output = keops_jax_op_impl(*args)
        return output, args

    def keops_jax_op_bwd(residuals, g):
        args = residuals
        args_stopped = tuple(jax.lax.stop_gradient(a) for a in args)
        args_with_eta = args_stopped + (g,)

        eta_dim = g.shape[-1]
        # Use formula content as key instead of id() to avoid cache collisions
        # when function objects are garbage collected and addresses reused
        # Convert aliases to tuple to ensure hashability
        formula_key = (formula, tuple(aliases), reduction_op, axis, dtype_str)
        cache_key = (formula_key, eta_dim, tuple(tuple(arg.shape) for arg in args))

        # Use bounded global cache instead of unbounded function attribute
        cached = _grad_cache.get(cache_key)

        if cached is None:
            compiled_grads = []
            for grad_info in precomputed_grads:
                grad_aliases = grad_info['grad_aliases_base'].copy()
                cat_str = {0: "Vi", 1: "Vj", 2: "Pm"}[grad_info['grad_eta_cat']]
                grad_aliases.append(f"eta={cat_str}({eta_dim})")

                grad_op = _make_keops_grad_op(
                    grad_info['grad_formula'],
                    tuple(grad_aliases),
                    reduction_op,
                    grad_info['grad_axis'],
                    dtype_str,
                    grad_info['input_dim'],
                    grad_info['var_cat'],
                    enable_vjp=False
                )

                compiled_grads.append({
                    'grad_op': grad_op,
                    'var_cat': grad_info['var_cat']
                })

            _grad_cache.set(cache_key, compiled_grads)
            cached = compiled_grads

        results = []

        for i, grad_info in enumerate(cached):
            grad_op = grad_info['grad_op']
            var_cat = grad_info['var_cat']
            raw_grad = grad_op(*args_with_eta)

            if var_cat != 2:
                if raw_grad.shape != args[i].shape:
                    target_shape = args[i].shape
                    while len(raw_grad.shape) > len(target_shape):
                        raw_grad = jnp.sum(raw_grad, axis=0)
                    if raw_grad.shape != target_shape:
                        raw_grad = raw_grad.reshape(target_shape)
                grad_i = raw_grad + (jnp.zeros((), dtype=args[i].dtype) * args[i])
            else:
                # For Pm (parameter) variables, gradient needs to be summed/reduced
                # to match the original parameter shape.
                # The gradient formula may produce (batch, n_i, dim) but input was (1, dim)
                input_shape = args[i].shape
                
                if raw_grad.shape != input_shape:
                    # Sum over extra leading dimensions
                    while len(raw_grad.shape) > len(input_shape):
                        raw_grad = jnp.sum(raw_grad, axis=0)
                    
                    # Sum over axes where input has size 1 but grad has larger size
                    # This is the key fix: for Pm params, we sum gradients, not reshape
                    if raw_grad.shape != input_shape:
                        axes_to_sum = []
                        for ax in range(len(raw_grad.shape)):
                            if ax < len(input_shape):
                                if input_shape[ax] == 1 and raw_grad.shape[ax] > 1:
                                    axes_to_sum.append(ax)
                        
                        # Sum over all mismatched axes at once, keeping dims
                        if axes_to_sum:
                            raw_grad = jnp.sum(raw_grad, axis=tuple(axes_to_sum), keepdims=True)
                    
                    # Final reshape if shapes still don't match (should be rare)
                    if raw_grad.shape != input_shape:
                        raw_grad = jnp.broadcast_to(raw_grad, input_shape) if raw_grad.size == 1 else raw_grad.reshape(input_shape)
                
                grad_i = raw_grad

            results.append(grad_i)

        return tuple(results)

    keops_jax_op.defvjp(keops_jax_op_fwd, keops_jax_op_bwd)
    return keops_jax_op


def keops_reduction(formula, aliases, reduction_op='Sum', axis=0, dtype='float32', enable_vjp=True, opt_arg=None, formula2=None):
    return make_keops_jax_op(formula, aliases, reduction_op, axis, dtype, enable_vjp, opt_arg=opt_arg, formula2=formula2)


def cleanup_registry():
    try:
        keops_jax_ext = _get_keops_ext()
        keops_jax_ext.cleanup_all_kernels()
        _grad_cache.clear()
        _registered_ffi_targets.clear()
    except Exception:
        pass


def get_registry_info():
    try:
        keops_jax_ext = _get_keops_ext()
        return {
            'num_kernels': keops_jax_ext.get_registry_size(),
            'unique_kernels': keops_jax_ext.get_unique_kernel_count() if hasattr(keops_jax_ext, 'get_unique_kernel_count') else 'N/A',
            'max_kernels': 5000,
            'grad_cache_size': len(_grad_cache.cache),
            'ffi_targets_registered': len(_registered_ffi_targets),
            'note': 'Thread-safe registry with shared_ptr lifetime management'
        }
    except Exception:
        return {
            'num_kernels': 0,
            'max_kernels': 5000,
            'ffi_targets_registered': len(_registered_ffi_targets),
            'note': 'Extension not loaded'
        }