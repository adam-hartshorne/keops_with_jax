"""
KeOps GPU Link Compile - CMake-based GPU compilation
Uses nvcc directly to compile .cu files to .so shared libraries

FIXES APPLIED:
1. Dtype-aware shared memory size calculation (float32/float64/float16)
2. Pinned memory for truly async cudaMemcpyAsync
3. Removed invalid scratch pointer cache assumption
4. Better error handling
"""

import os
import sysconfig
from os.path import join

from keopscore.binders.LinkCompile import LinkCompile
from keopscore.config import *
from keopscore.utils.misc_utils import KeOps_Error, KeOps_Message, KeOps_OS_Run

cuda_version = cuda_config.get_cuda_version()
cxx_compiler = config.get_cxx_compiler()
cuda_available = cuda_config.get_use_cuda()
build_folder = config.get_build_folder()
get_gpu_props = cuda_config.get_gpu_props()


class Cuda_link_compile(LinkCompile):
    """
    CMake-based GPU compilation (alternative to NVRTC)
    Uses nvcc directly to compile .cu files to .so shared libraries
    Includes a launcher function for GIL-free execution from C++
    """
    source_code_extension = "cu"
    ngpu, gpu_props_compile_flags = get_gpu_props

    def __init__(self, lang=None):
        if not (cuda_available and Cuda_link_compile.ngpu):
            KeOps_Error(
                "Trying to compile cuda code... but we detected that the system has no properly configured cuda lib."
            )

        # FIXED: Pass lang parameter to base class (was hardcoded to None)
        LinkCompile.__init__(self, lang=lang)

        self.low_level_code_file = "".encode("utf-8")

        self.so_file = os.path.join(
            build_folder,
            self.gencode_filename + sysconfig.get_config_var("SHLIB_SUFFIX"),
        )

        self.true_dllname = self.so_file
        self.file_to_check = self.so_file

    def save_info(self):
        """
        Save info file using base class's info_file path for consistency.

        FIXED: Previously wrote to self.so_file + ".info" but read_info()
        reads from self.info_file (.nfo). Now uses consistent path.
        Uses atomic write pattern for reliability.
        """
        import tempfile

        red_formula = getattr(self, 'red_formula_string', "Unknown")
        dim = getattr(self, 'dim', getattr(self, 'dimout', getattr(self, 'dimy', 0)))
        tagI = getattr(self, 'tagI', 0)
        dimy = getattr(self, 'dimy', 0)

        info_str = f"red_formula={red_formula}\ndim={dim}\ntagI={tagI}\ndimy={dimy}"

        # Use base class's info_file path for consistency with read_info()
        info_dir = os.path.dirname(self.info_file)

        # Atomic write: write to temp file then rename
        fd, temp_path = tempfile.mkstemp(dir=info_dir, prefix=".tmp_info_", suffix=".nfo")
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(info_str)
            # Atomic rename (on POSIX systems)
            os.replace(temp_path, self.info_file)
        except Exception as e:
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise e

    def generate_code(self):
        self.get_code()
        
        # Debug output for JAX chunked kernels (only when JAX_KEOPS_DEBUG=1)
        import os
        if getattr(self, 'jax_mode', False) and os.environ.get("JAX_KEOPS_DEBUG") == "1":
            print(f"[Cuda_link_compile DEBUG] dimy={getattr(self, 'dimy', 'N/A')}")
            print(f"[Cuda_link_compile DEBUG] blocksize_chunks={getattr(self, 'blocksize_chunks', 'N/A')}")
            if hasattr(self, 'chk'):
                print(f"[Cuda_link_compile DEBUG] chk.dimy={self.chk.dimy}")

        dtype_bytes = self._detect_dtype_bytes(self.code)
        self.code = self.add_launcher_wrapper(self.code, dtype_bytes=dtype_bytes)
        self.write_code()

        # print("\n")
        KeOps_Message("Compiling formula using nvcc ... ", flush=True, end="")

        cuda_include = cuda_config.get_cuda_include_path()

        compile_flags = [
            "nvcc",
            "-shared",
            "-Xcompiler", "-fPIC",
            "-O3",
            "--use_fast_math",
            f"-I{config.get_bindings_source_dir()}",
        ]

        if cuda_include:
            compile_flags.append(f"-I{cuda_include}")

        gpu_arch_flags = cuda_config.get_gpu_arch_flags()
        if gpu_arch_flags:
            compile_flags.append(gpu_arch_flags)

        if Cuda_link_compile.gpu_props_compile_flags:
            compile_flags.append(Cuda_link_compile.gpu_props_compile_flags)

        compile_flags.extend([
            self.gencode_file,
            "-o", self.so_file
        ])

        command = " ".join(compile_flags)

        KeOps_OS_Run(command)

        if not os.path.exists(self.so_file):
            KeOps_Error(f"CMake compilation succeeded but .so file not found: {self.so_file}")

        KeOps_Message("OK", use_tag=False, flush=True)

        self.tagI = getattr(self.red_formula, 'tagI', 0)
        self.dim = getattr(self.red_formula, 'dim', getattr(self.red_formula, 'dimout', 0))

    def _detect_dtype_bytes(self, kernel_code):
        """
        Get dtype size in bytes for shared memory calculation.

        Uses hybrid approach:
        1. Try explicit attributes first (cleaner, more reliable if set)
        2. Fall back to kernel code parsing (robust if attributes missing)

        Returns: 4 (float32), 8 (float64), or 2 (float16)
        """
        # Method 1: Use explicit attributes (preferred)
        dtype = getattr(self, "dtype", None)
        if dtype is not None:
            dtype_lower = str(dtype).lower()
            if dtype_lower in ("float64", "double"):
                return 8
            elif dtype_lower in ("float16", "half"):
                return 2
            elif dtype_lower == "float32":
                return 4

        # Check use_half flag
        if getattr(self, "use_half", False):
            return 2

        # Check use_double flag (if exists)
        if getattr(self, "use_double", False):
            return 8

        # Method 2: Fall back to kernel code parsing (robust)
        if '__TYPEACC__ double' in kernel_code or 'typedef double __TYPE__' in kernel_code:
            return 8
        elif '__half' in kernel_code or 'typedef __half __TYPE__' in kernel_code:
            return 2

        # Default to float32
        return 4

    def add_launcher_wrapper(self, kernel_code, dtype_bytes=4):
        """Add a C-compatible launcher function that can be called from C++ via dlopen."""
        is_ranges = "GpuConv1DOnDevice_ranges" in kernel_code

        # Map dtype_bytes to C++ type for proper pointer casting
        if dtype_bytes == 8:
            cpp_dtype = "double"
        elif dtype_bytes == 2:
            cpp_dtype = "__half"  # CUDA half-precision type
        else:
            cpp_dtype = "float"

        # Get the number of argument slots needed from varloader
        # nminargs is max(variable_index) + 1, which accounts for all variables including gaps
        # FIX: Use nminargs instead of len(all_indices) to handle sparse variable indices
        # This is crucial for gradient formulas where some variables drop out (e.g., d(K*v)/dv = K, v drops)
        indsi = list(getattr(self.varloader, 'indsi', []))
        indsj = list(getattr(self.varloader, 'indsj', []))
        indsp = list(getattr(self.varloader, 'indsp', []))

        all_indices = indsi + indsj + indsp

        if all_indices:
            # sparse_args_count = max(variable_index) + 1 to handle gaps in indices
            sparse_args_count = max(all_indices) + 1
        else:
            sparse_args_count = 0

        # FIX: Use identity mapping - FFI arguments are passed in alias index order [0, 1, 2, ...]
        # and the kernel expects arguments at those same indices. No reordering needed.
        # The old code incorrectly used len(all_indices) which caused misalignment when
        # some variables were not used in the formula (e.g., gradient formulas)

        # FIX: Compute compile-time variable counts to avoid runtime mismatch
        # These MUST match what the compiled kernel expects
        compile_time_nvi = len(indsi)
        compile_time_nvj = len(indsj)
        compile_time_nvp = len(indsp)

        preamble = '''
#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <algorithm>
#include <unordered_map>

using std::min;

#define KEOPS_DTYPE_BYTES ''' + str(dtype_bytes) + '''
#define PRECOMPUTED_SPARSE_ARGS_COUNT ''' + str(sparse_args_count) + '''
// FIX: Use compile-time variable counts to match kernel expectations
#define PRECOMPUTED_NVI ''' + str(compile_time_nvi) + '''
#define PRECOMPUTED_NVJ ''' + str(compile_time_nvj) + '''
#define PRECOMPUTED_NVP ''' + str(compile_time_nvp) + '''

#define CHECK_CUDA_LAUNCH(err) \\
    if (err != cudaSuccess) { \\
        printf("[LAUNCHER] CUDA Error: %s\\n", cudaGetErrorString(err)); \\
        return 1; \\
    }

struct PinnedBuffer {
    void* ptr = nullptr;
    size_t capacity = 0;
    
    ~PinnedBuffer() {
        if (ptr) {
            cudaFreeHost(ptr);
            ptr = nullptr;
        }
    }
    
    void* ensure(size_t needed) {
        if (needed > capacity) {
            if (ptr) cudaFreeHost(ptr);
            // Optimization: Grow geometrically (2x) to avoid frequent re-allocs
            // when sizes jitter slightly upwards between calls
            size_t new_cap = needed * 2;
            cudaError_t err = cudaMallocHost(&ptr, new_cap);
            if (err != cudaSuccess) {
                // Fallback to exact size if OOM on 2x allocation
                new_cap = needed;
                err = cudaMallocHost(&ptr, new_cap);
                if (err != cudaSuccess) {
                    // Last resort: use pageable memory (async becomes sync, but won't crash)
                    ptr = malloc(needed);
                    capacity = (ptr) ? needed : 0;
                    return ptr;
                }
            }
            capacity = new_cap;
        }
        return ptr;
    }
};
'''

        if is_ranges:
            launcher_code = preamble + '''
struct LaunchCache {
    int nx = -1;
    int ny = -1;
    int batch_size = -1;
    int cuda_block_size = -1;
    int nvi = -1;
    int nvj = -1;
    int nvp = -1;
    int tagI = -1;  // FIX: Include tagI in cache key to distinguish swapped kernels
    std::vector<char> cached_tables;
    size_t tables_size = 0;

    bool is_valid(int _nx, int _ny, int _batch_size, int _cuda_block_size, int _nvi, int _nvj, int _nvp, int _tagI) const {
        return nx == _nx && ny == _ny && batch_size == _batch_size &&
               cuda_block_size == _cuda_block_size &&
               nvi == _nvi && nvj == _nvj && nvp == _nvp &&
               tagI == _tagI &&  // FIX: Check tagI
               !cached_tables.empty();
    }

    void invalidate() {
        nx = ny = batch_size = cuda_block_size = nvi = nvj = nvp = tagI = -1;
        cached_tables.clear();
        tables_size = 0;
    }
};

static thread_local std::unordered_map<int, LaunchCache> g_launch_caches;
static thread_local PinnedBuffer g_pinned_buffer;

extern "C" int launch_keops_kernel(
    int tagHostDevice,
    int dimy,
    int nx,
    int ny,
    int tagI,
    int tagZero,
    int use_half,
    int tag1D2D,
    int dimred,
    int cuda_block_size,
    int use_chunk_mode,
    const int* indsi,
    const int* indsj,
    const int* indsp,
    int dimout,
    const int* dimsx,
    const int* dimsy,
    const int* dimsp,
    void* ranges,
    void* shapeout,
    void* out_ptr,
    void** args_ptr,
    void* argshapes,
    cudaStream_t stream,
    void* scratch_ptr
) {
    ''' + cpp_dtype + '''* out = (''' + cpp_dtype + '''*)out_ptr;
    ''' + cpp_dtype + '''** args = (''' + cpp_dtype + '''**)args_ptr;

    int batch_size = (int)(long long)ranges;
    if (batch_size == 0) batch_size = 1;

    // FIX: Use compile-time variable counts instead of runtime values
    // Runtime values from argshapes can mismatch when axis is swapped (tagI=1)
    // Compile-time values are always correct for this specific kernel
    constexpr int nvi = PRECOMPUTED_NVI;
    constexpr int nvj = PRECOMPUTED_NVJ;
    constexpr int nvp = PRECOMPUTED_NVP;

    // Dynamically adjust block size to fit shared memory constraints
    // Use 49152 bytes (48KB) to match SHAREDMEMPERBLOCK in CudaSizes.h
    int effective_block_size = std::min(
        cuda_block_size,
        (int)(49152 / std::max(1, dimy * KEOPS_DTYPE_BYTES))
    );

    int nbatchdims = 1;
    dim3 blockSize(effective_block_size);
    dim3 gridSize(((nx + blockSize.x - 1) / blockSize.x) * batch_size);
    size_t nblocks = gridSize.x;

    constexpr int total_offsets = (nvi + nvj + nvp > 0) ? (nvi + nvj + nvp) : 2;

    // Use precomputed value for argument array size
    constexpr int sparse_args_count = PRECOMPUTED_SPARSE_ARGS_COUNT;

    int device_id;
    cudaError_t cuda_err = cudaGetDevice(&device_id);
    if (cuda_err != cudaSuccess) {
        printf("[LAUNCHER] Failed to get CUDA device: %s\\n", cudaGetErrorString(cuda_err));
        return 1;
    }
    LaunchCache& cache = g_launch_caches[device_id];

    char* device_ptr = (char*)scratch_ptr;

    size_t size_offsets = sizeof(signed long int) * nblocks * total_offsets;
    size_t size_lookup  = sizeof(signed long int) * 3 * nblocks;
    size_t size_slices  = sizeof(signed long int) * batch_size;
    size_t size_ranges  = sizeof(signed long int) * 2 * batch_size;
    size_t size_args    = sizeof(''' + cpp_dtype + '''*) * sparse_args_count;
    size_t tables_size  = size_offsets + size_lookup + size_slices + size_ranges;
    size_t total_upload_size = tables_size + size_args;

    signed long int* offsets_d = (signed long int*)(device_ptr);
    signed long int* lookup_d  = (signed long int*)(device_ptr + size_offsets);
    signed long int* slices_x  = (signed long int*)(device_ptr + size_offsets + size_lookup);
    signed long int* ranges_y  = (signed long int*)(device_ptr + size_offsets + size_lookup + size_slices);
    ''' + cpp_dtype + '''** args_d = (''' + cpp_dtype + '''**)(device_ptr + tables_size);

    bool cache_hit = cache.is_valid(nx, ny, batch_size, cuda_block_size, nvi, nvj, nvp, tagI);

    void* pinned_ptr = g_pinned_buffer.ensure(total_upload_size);
    if (!pinned_ptr) {
        printf("[LAUNCHER] Failed to allocate pinned memory\\n");
        return 1;
    }
    char* h_ptr = (char*)pinned_ptr;

    if (!cache_hit) {
        int blocks_per_batch = (nx + blockSize.x - 1) / blockSize.x;

        signed long int* h_offsets = (signed long int*)h_ptr;
        for (int b = 0; b < batch_size; b++) {
            for (int block_in_batch = 0; block_in_batch < blocks_per_batch; block_in_batch++) {
                int block_id = b * blocks_per_batch + block_in_batch;
                int offset_idx = total_offsets * block_id;
                int block_offset = block_in_batch * blockSize.x;

                for (int i = 0; i < nvi; i++) h_offsets[offset_idx + i] = b * nx + block_offset;
                for (int j = 0; j < nvj; j++) h_offsets[offset_idx + nvi + j] = b * ny;
                for (int p = 0; p < nvp; p++) h_offsets[offset_idx + nvi + nvj + p] = 0;
            }
        }

        signed long int* h_lookup = (signed long int*)(h_ptr + size_offsets);
        for (int b = 0; b < batch_size; b++) {
            for (int block_in_batch = 0; block_in_batch < blocks_per_batch; block_in_batch++) {
                int block_id = b * blocks_per_batch + block_in_batch;
                h_lookup[3*block_id + 0] = b;
                h_lookup[3*block_id + 1] = b * nx + block_in_batch * blockSize.x;
                int end_within_batch = ((block_in_batch + 1) * blockSize.x < nx) ? 
                                       ((block_in_batch + 1) * blockSize.x) : nx;
                h_lookup[3*block_id + 2] = b * nx + end_within_batch;
            }
        }

        signed long int* h_slices = (signed long int*)(h_ptr + size_offsets + size_lookup);
        for (int b = 0; b < batch_size; b++) { h_slices[b] = b + 1; }

        signed long int* h_ranges = (signed long int*)(h_ptr + size_offsets + size_lookup + size_slices);
        for (int b = 0; b < batch_size; b++) { 
            h_ranges[2*b + 0] = b * ny;
            h_ranges[2*b + 1] = (b+1) * ny;
        }

        cache.nx = nx;
        cache.ny = ny;
        cache.batch_size = batch_size;
        cache.cuda_block_size = cuda_block_size;
        cache.nvi = nvi;
        cache.nvj = nvj;
        cache.nvp = nvp;
        cache.tagI = tagI;  // FIX: Store tagI for cache validation
        cache.tables_size = tables_size;
        
        cache.cached_tables.resize(tables_size);
        memcpy(cache.cached_tables.data(), h_ptr, tables_size);
    } else {
        memcpy(h_ptr, cache.cached_tables.data(), tables_size);
    }

    ''' + cpp_dtype + '''** h_args = (''' + cpp_dtype + '''**)(h_ptr + tables_size);

    // FIX: Use identity mapping - FFI passes arguments in order [0, 1, 2, ...],
    // and kernel expects them at the same indices. This correctly handles cases
    // where some variables are not used in the formula (e.g., gradient formulas).
    // Simply copy all arguments directly to h_args.
    memcpy(h_args, args, size_args);

    cudaMemcpyAsync(device_ptr, h_ptr, total_upload_size, cudaMemcpyHostToDevice, stream);

    size_t shared_mem = effective_block_size * dimy * KEOPS_DTYPE_BYTES;

    GpuConv1DOnDevice_ranges<<<gridSize, blockSize, shared_mem, stream>>>(
        nx, ny, nbatchdims, offsets_d, lookup_d, slices_x, ranges_y, out, args_d);

    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        printf("[LAUNCHER] Kernel launch error: %s\\n", cudaGetErrorString(cuda_err));
        return 1;
    }
    return 0;
}
'''
        else:
            launcher_code = preamble + '''
// Ultra-minimal launcher - all overhead removed
extern "C" int launch_keops_kernel(
    int tagHostDevice,
    int dimy,
    int nx,
    int ny,
    int tagI,
    int tagZero,
    int use_half,
    int tag1D2D,
    int dimred,
    int cuda_block_size,
    int use_chunk_mode,
    const int* indsi,
    const int* indsj,
    const int* indsp,
    int dimout,
    const int* dimsx,
    const int* dimsy,
    const int* dimsp,
    void* ranges,
    void* shapeout,
    void* out_ptr,
    void** args_ptr,
    void* argshapes,
    cudaStream_t stream,
    void* scratch_ptr
) {
    // Direct pointer casts - no validation
    ''' + cpp_dtype + '''* out = (''' + cpp_dtype + '''*)out_ptr;
    ''' + cpp_dtype + '''** args = (''' + cpp_dtype + '''**)args_ptr;
    ''' + cpp_dtype + '''** args_d = (''' + cpp_dtype + '''**)scratch_ptr;

    // Block size adjustment for shared memory
    // Use 49152 bytes (48KB) to match SHAREDMEMPERBLOCK in CudaSizes.h
    int effective_block_size = std::min(
        cuda_block_size,
        (int)(49152 / std::max(1, dimy * KEOPS_DTYPE_BYTES))
    );

    // FIX: Use identity mapping - copy args directly to device
    // FFI passes arguments in order [0, 1, 2, ...] and kernel expects same indices
    cudaMemcpyAsync(args_d, args,
                    sizeof(''' + cpp_dtype + '''*) * PRECOMPUTED_SPARSE_ARGS_COUNT,
                    cudaMemcpyHostToDevice, stream);

    // Launch kernel - no error checking in hot path
    dim3 blockSize(effective_block_size);
    dim3 gridSize((nx + effective_block_size - 1) / effective_block_size);
    size_t shared_mem = effective_block_size * dimy * KEOPS_DTYPE_BYTES;

    GpuConv1DOnDevice<<<gridSize, blockSize, shared_mem, stream>>>(nx, ny, out, args_d);

    return 0;
}
'''
        return kernel_code + launcher_code