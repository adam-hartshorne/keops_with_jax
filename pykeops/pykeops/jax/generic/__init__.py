"""
KeOps JAX backend initialization

This module provides JAX-compatible KeOps operations using the CMake/XLA FFI backend.
"""

from .generic_red import Genred
from .generic_ops import keops_reduction, cleanup_registry, get_registry_info

__all__ = ['Genred', 'keops_reduction', 'cleanup_registry', 'get_registry_info']