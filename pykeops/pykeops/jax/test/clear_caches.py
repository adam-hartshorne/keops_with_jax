#!/usr/bin/env python3
"""
Script to clear KeOps JAX function caches after applying bug fixes.

The @lru_cache decorator in generic_ops.py caches function results,
which means after fixing bugs, the OLD buggy version is still being used
until you restart Python or clear the cache explicitly.
"""

import os
os.environ['PYKEOPS_JAX_MODE'] = '1'

import sys
import jax

def clear_all_keops_caches():
    # 1. Clear JAX's internal JIT/Compilation cache
    jax.clear_caches()

    # 2. Clear Python LRU caches in your module
    from pykeops.jax.generic import generic_ops
    generic_ops._compute_kernel_hash.cache_clear()
    generic_ops._compute_output_shape_cached.cache_clear()

    # 3. Clear the C++ Kernel Registry
    try:
        ext = generic_ops._get_keops_ext()
        ext.cleanup_all_kernels()
    except:
        pass