"""
GpuReduc1D - GPU 1D reduction with runtime backend selection.

This module supports both NVRTC (for PyTorch/NumPy) and CMake/CUDA (for JAX)
backends, selected at runtime based on the `lang` parameter.
"""

import os

# Import BOTH backends at module level for runtime selection
from keopscore.binders.cuda.Cuda_link_compile import Cuda_link_compile
from keopscore.binders.nvrtc.Gpu_link_compile import Gpu_link_compile as Nvrtc_link_compile

from keopscore.mapreduce.gpu.GpuAssignZero import GpuAssignZero
from keopscore.mapreduce.MapReduce import MapReduce
from keopscore.utils.code_gen_utils import (
    c_variable,
    c_array,
)


def _use_cuda_backend(lang):
    """
    Determine if we should use CUDA/CMake backend instead of NVRTC.

    Args:
        lang: Language/frontend being used ("torch", "numpy", "jax", or None)

    Returns:
        True if CUDA/CMake backend should be used (for JAX or when env var is set),
        False for NVRTC backend (default for PyTorch/NumPy).
    """
    return (lang == "jax")


class GpuReduc1D_Cuda(MapReduce, Cuda_link_compile):
    """GpuReduc1D using CUDA/CMake backend (for JAX)."""

    AssignZero = GpuAssignZero

    def __init__(self, *args, lang=None):
        MapReduce.__init__(self, *args)
        Cuda_link_compile.__init__(self, lang=lang)
        self.dimy = self.varloader.dimy

    def get_code(self):
        super().get_code()
        _generate_gpu_reduc1d_code(self)


class GpuReduc1D_Nvrtc(MapReduce, Nvrtc_link_compile):
    """GpuReduc1D using NVRTC backend (for PyTorch/NumPy)."""

    AssignZero = GpuAssignZero

    def __init__(self, *args, lang=None):
        MapReduce.__init__(self, *args)
        Nvrtc_link_compile.__init__(self)
        self.dimy = self.varloader.dimy

    def get_code(self):
        super().get_code()
        _generate_gpu_reduc1d_code(self)


def _generate_gpu_reduc1d_code(self):
    """Shared code generation logic for GpuReduc1D."""
    red_formula = self.red_formula
    dtype = self.dtype
    varloader = self.varloader

    i = self.i
    j = self.j
    fout = self.fout
    outi = self.outi
    acc = self.acc
    arg = self.arg
    args = self.args
    sum_scheme = self.sum_scheme

    param_loc = self.param_loc
    xi = self.xi
    yjloc = c_array(dtype, varloader.dimy, f"(yj + threadIdx.x * {varloader.dimy})")
    yjrel = c_array(dtype, varloader.dimy, "yjrel")
    table = varloader.table(self.xi, yjrel, self.param_loc)
    jreltile = c_variable("signed long int", "(jrel + tile * blockDim.x)")

    print('GpuConv1DOnDevice')

    self.code = f"""

                    {self.headers}

                    extern "C" __global__ void GpuConv1DOnDevice(signed long int nx, signed long int ny, {dtype} *out, {dtype} **{arg.id}) {{

                      // get the index of the current thread
                      signed long int i = blockIdx.x * blockDim.x + threadIdx.x;

                      // declare shared mem
                      extern __shared__ {dtype} yj[];

                      // load parameters variables from global memory to local thread memory
                      {param_loc.declare()}
                      {varloader.load_vars("p", param_loc, args)}

                      {fout.declare()}
                      {xi.declare()}
                      {acc.declare()}
                      {sum_scheme.declare_temporary_accumulator()}

                      if (i < nx) {{
                        {red_formula.InitializeReduction(acc)} // acc = 0
                        {sum_scheme.initialize_temporary_accumulator_first_init()}
                        {varloader.load_vars('i', xi, args, row_index=i)} // load xi variables from global memory to local thread memory
                      }}

                      for (signed long int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++) {{

                        // get the current column
                        signed long int j = tile * blockDim.x + threadIdx.x;

                        if (j < ny) {{ // we load yj from device global memory only if j<ny
                          {varloader.load_vars("j", yjloc, args, row_index=j)} 
                        }}
                        __syncthreads();

                        if (i < nx) {{ // we compute x1i only if needed
                          {dtype} * yjrel = yj;
                          {sum_scheme.initialize_temporary_accumulator_block_init()}
                          for (signed long int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++, yjrel += {varloader.dimy}) {{
                            {red_formula.formula(fout, table)} // Call the function, which outputs results in fout
                            {sum_scheme.accumulate_result(acc, fout, jreltile)}
                          }}
                          {sum_scheme.final_operation(acc)}
                        }}
                        __syncthreads();
                      }}
                      if (i < nx) {{
                        {red_formula.FinalizeOutput(acc, outi, i)} 
                      }}

                    }}
                """


class GpuReduc1D:
    """
    Factory class for GPU 1D reduction with runtime backend selection.

    Returns either GpuReduc1D_Cuda or GpuReduc1D_Nvrtc based on lang parameter.
    """

    AssignZero = GpuAssignZero

    def __new__(cls, *args, lang=None):
        """
        Create appropriate backend instance based on lang parameter.

        Args:
            *args: Standard arguments for MapReduce
            lang: Language/frontend being used ("torch", "numpy", "jax", or None).
                  JAX requires CMake backend instead of NVRTC for multi-GPU support.
        """
        if _use_cuda_backend(lang):
            return GpuReduc1D_Cuda(*args, lang=lang)
        else:
            return GpuReduc1D_Nvrtc(*args, lang=lang)