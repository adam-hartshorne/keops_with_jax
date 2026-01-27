"""
KeOps JAX backend initialization

This module provides JAX-compatible KeOps operations using the CMake/XLA FFI backend.
"""

from .generic_red import Genred
from .generic_ops import keops_reduction, cleanup_registry, get_registry_info
from .generic_helpers import (
    generic_sum,
    generic_logsumexp,
    generic_argmin,
    generic_argkmin,
    generic_min,
    generic_max,
)

__all__ = [
    'Genred',
    'keops_reduction',
    'cleanup_registry',
    'get_registry_info',
    'generic_sum',
    'generic_logsumexp',
    'generic_argmin',
    'generic_argkmin',
    'generic_min',
    'generic_max',
]