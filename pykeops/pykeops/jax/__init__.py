# /media/adam/shared_folder/PycharmProjects/keops/pykeops/pykeops/jax/__init__.py
# Use the explicit subfolder path
from .lazytensor.LazyTensor import LazyTensor, Vi, Vj, Pm
from .generic import Genred
from .generic import (
    generic_sum,
    generic_logsumexp,
    generic_argmin,
    generic_argkmin,
    generic_min,
    generic_max,
)
from .operations import KernelSolve

# This ensures the FFI targets are registered on import
from .generic import generic_ops

__all__ = [
    'LazyTensor', 'Vi', 'Vj', 'Pm', 'Genred', 'KernelSolve',
    'generic_sum', 'generic_logsumexp', 'generic_argmin',
    'generic_argkmin', 'generic_min', 'generic_max',
]
