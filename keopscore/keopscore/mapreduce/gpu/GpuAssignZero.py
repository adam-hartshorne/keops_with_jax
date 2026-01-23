"""
GpuAssignZero - GPU kernel for assigning zeros (bypass reduction).

This module supports both NVRTC (for PyTorch/NumPy) and CMake/CUDA (for JAX)
backends, selected at runtime based on the `lang` parameter.
"""

import os

# Import BOTH backends at module level for runtime selection
from keopscore.binders.cuda.Cuda_link_compile import Cuda_link_compile
from keopscore.binders.nvrtc.Gpu_link_compile import Gpu_link_compile as Nvrtc_link_compile

from keopscore.mapreduce.MapReduce import MapReduce
from keopscore.utils.code_gen_utils import (
    c_include,
    c_zero_float,
)


def _use_cuda_backend(lang):
    """
    Determine if we should use CUDA/CMake backend instead of NVRTC.
    """
    return (lang == "jax")


class GpuAssignZero_Cuda(MapReduce, Cuda_link_compile):
    """GpuAssignZero using CUDA/CMake backend (for JAX)."""

    def __init__(self, *args, lang=None):
        MapReduce.__init__(self, *args)
        Cuda_link_compile.__init__(self, lang=lang)
        self.dimy = self.varloader.dimy

    def get_code(self):
        super().get_code()
        _generate_gpu_assign_zero_code(self)


class GpuAssignZero_Nvrtc(MapReduce, Nvrtc_link_compile):
    """GpuAssignZero using NVRTC backend (for PyTorch/NumPy)."""

    def __init__(self, *args, lang=None):
        MapReduce.__init__(self, *args)
        Nvrtc_link_compile.__init__(self)
        self.dimy = self.varloader.dimy

    def get_code(self):
        super().get_code()
        _generate_gpu_assign_zero_code(self)


def _generate_gpu_assign_zero_code(self):
    """Shared code generation logic for GpuAssignZero."""
    outi = self.outi
    dtype = self.dtype
    arg = self.arg
    varloader = self.varloader

    if dtype == "half2":
        self.headers += c_include("cuda_fp16.h")

    self.code = f"""
                    {self.headers}

                    extern "C" __global__ void GpuConv1DOnDevice(signed long int nx, signed long int ny, {dtype} *out, {dtype} **{arg.id}) {{

                      // get the index of the current thread
                      signed long int i = blockIdx.x * blockDim.x + threadIdx.x;

                      if (i < nx) {{
                        {outi.assign(c_zero_float)}
                      }}

                    }}
                """


class GpuAssignZero:
    """
    Factory class for GPU assign zero with runtime backend selection.
    """

    def __new__(cls, *args, lang=None):
        if _use_cuda_backend(lang):
            return GpuAssignZero_Cuda(*args, lang=lang)
        else:
            return GpuAssignZero_Nvrtc(*args, lang=lang)
