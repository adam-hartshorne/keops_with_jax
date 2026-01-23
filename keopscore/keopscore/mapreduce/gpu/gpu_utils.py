"""
GPU MapReduce utilities.
"""

def use_cuda_backend(lang):
    """
    Determine if we should use CUDA/CMake backend instead of NVRTC.

    Args:
        lang: Language/frontend being used ("torch", "numpy", "jax", or None)

    Returns:
        True if CUDA/CMake backend should be used (for JAX),
        False for NVRTC backend (default for PyTorch/NumPy).
    """
    return lang == "jax"