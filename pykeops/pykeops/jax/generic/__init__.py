"""
KeOps JAX backend initialization
Sets JAX mode to force CMake compilation backend
"""
import os

# =============================================================================
# CRITICAL: Check that JAX mode is set BEFORE importing
# =============================================================================

# Enable CUDA
os.environ["USE_CUDA"] = "1"

# Check that JAX mode was set by user
if os.environ.get("PYKEOPS_JAX_MODE") != "1":
    raise RuntimeError(
        "PYKEOPS_JAX_MODE must be set to '1' BEFORE importing pykeops.jax!\n"
        "Add this at the top of your script:\n"
        "    import os\n"
        "    os.environ['PYKEOPS_JAX_MODE'] = '1'\n"
        "    # Then import pykeops.jax\n"
    )

print("[KeOps JAX] JAX mode confirmed (PYKEOPS_JAX_MODE=1)")

# =============================================================================
# Import pykeops to set config
# =============================================================================

import pykeops
import pykeops.config

# Ensure CMake backend is used
pykeops.config.compile_engine = "cmake"
pykeops.config.use_cuda = True

print(f"[KeOps JAX] Backend configuration:")
print(f"  Compile engine: {pykeops.config.compile_engine}")
print(f"  CUDA enabled: {pykeops.config.use_cuda}")
print(f"  PYKEOPS_JAX_MODE: {os.environ.get('PYKEOPS_JAX_MODE')}")

# =============================================================================
# Import JAX operations
# =============================================================================

from .generic_red import Genred
from .generic_ops import keops_reduction, cleanup_registry, get_registry_info

__all__ = ['Genred', 'keops_reduction', 'cleanup_registry', 'get_registry_info']