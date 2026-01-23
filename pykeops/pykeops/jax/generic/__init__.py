"""
KeOps JAX backend initialization

This module provides JAX-compatible KeOps operations. The backend selection
(CUDA/CMake vs NVRTC) is now handled automatically at runtime based on the
array type, so no environment variable setup is required.
"""
import os

# =============================================================================
# Configuration
# =============================================================================

# Enable CUDA
os.environ["USE_CUDA"] = "1"

# =============================================================================
# Import pykeops to set config
# =============================================================================

import pykeops
import pykeops.config

# Ensure CMake backend is used for JAX
pykeops.config.compile_engine = "cmake"
pykeops.config.use_cuda = True

# =============================================================================
# Import JAX operations
# =============================================================================

from .generic_red import Genred
from .generic_ops import keops_reduction, cleanup_registry, get_registry_info

__all__ = ['Genred', 'keops_reduction', 'cleanup_registry', 'get_registry_info']