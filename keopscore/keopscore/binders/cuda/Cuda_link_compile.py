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

        LinkCompile.__init__(self, lang=None)

        self.low_level_code_file = "".encode("utf-8")

        self.so_file = os.path.join(
            build_folder,
            self.gencode_filename + sysconfig.get_config_var("SHLIB_SUFFIX"),
        )

        self.true_dllname = self.so_file
        self.file_to_check = self.so_file

    def save_info(self):
        red_formula = getattr(self, 'red_formula_string', "Unknown")
        dim = getattr(self, 'dim', getattr(self, 'dimout', getattr(self, 'dimy', 0)))
        tagI = getattr(self, 'tagI', 0)
        dimy = getattr(self, 'dimy', 0)

        info_str = f"red_formula={red_formula}\ndim={dim}\ntagI={tagI}\ndimy={dimy}"

        info_file = self.so_file + ".info"
        with open(info_file, "w") as f:
            f.write(info_str)

    def generate_code(self):
        self.get_code()

        dtype_bytes = self._detect_dtype_bytes(self.code)
        self.code = self.add_launcher_wrapper(self.code, dtype_bytes=dtype_bytes)
        self.write_code()

        KeOps_Message("Compiling formula using CMake/nvcc ... ", flush=True, end="")

        cuda_include = cuda_config.get_cuda_include_path()

        compile_flags = [
            "nvcc",
            "-shared",
            "-Xcompiler", "-fPIC",
            "-O3",
            "--use_fast_math",
            "--ptxas-options=-v",
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

        preamble = '''
#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <algorithm>
#include <unordered_map>

using std::min;

#define KEOPS_DTYPE_BYTES ''' + str(dtype_bytes) + '''

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
    std::vector<char> cached_tables;
    size_t tables_size = 0;

    bool is_valid(int _nx, int _ny, int _batch_size, int _cuda_block_size) const {
        return nx == _nx && ny == _ny && batch_size == _batch_size && 
               cuda_block_size == _cuda_block_size && !cached_tables.empty();
    }

    void invalidate() {
        nx = ny = batch_size = cuda_block_size = -1;
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
    float* out = (float*)out_ptr;
    float** args = (float**)args_ptr;

    int batch_size = (int)(long long)ranges;
    if (batch_size == 0) batch_size = 1;

    int64_t var_counts = (int64_t)argshapes;
    int nvi = var_counts & 0xFF;
    int nvj = (var_counts >> 8) & 0xFF;
    int nvp = (var_counts >> 16) & 0xFF;

    int nbatchdims = 1;
    dim3 blockSize(cuda_block_size);
    dim3 gridSize(((nx + blockSize.x - 1) / blockSize.x) * batch_size);
    size_t nblocks = gridSize.x;

    int total_offsets = nvi + nvj + nvp;
    if (total_offsets == 0) total_offsets = 2;

    int max_var_idx = -1;
    if (nvi > 0 && indsi) {
        for (int i = 0; i < nvi; i++) if (indsi[i] > max_var_idx) max_var_idx = indsi[i];
    }
    if (nvj > 0 && indsj) {
        for (int j = 0; j < nvj; j++) if (indsj[j] > max_var_idx) max_var_idx = indsj[j];
    }
    if (nvp > 0 && indsp) {
        for (int p = 0; p < nvp; p++) if (indsp[p] > max_var_idx) max_var_idx = indsp[p];
    }
    int sparse_args_count = max_var_idx + 1;

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
    size_t size_args    = sizeof(float*) * sparse_args_count;
    size_t tables_size  = size_offsets + size_lookup + size_slices + size_ranges;
    size_t total_upload_size = tables_size + size_args;

    signed long int* offsets_d = (signed long int*)(device_ptr);
    signed long int* lookup_d  = (signed long int*)(device_ptr + size_offsets);
    signed long int* slices_x  = (signed long int*)(device_ptr + size_offsets + size_lookup);
    signed long int* ranges_y  = (signed long int*)(device_ptr + size_offsets + size_lookup + size_slices);
    float** args_d             = (float**)(device_ptr + tables_size);

    bool cache_hit = cache.is_valid(nx, ny, batch_size, cuda_block_size);

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
        cache.tables_size = tables_size;
        
        cache.cached_tables.resize(tables_size);
        memcpy(cache.cached_tables.data(), h_ptr, tables_size);
    } else {
        memcpy(h_ptr, cache.cached_tables.data(), tables_size);
    }

    float** h_args = (float**)(h_ptr + tables_size);

    float* sparse_buffer[256];
    float** sparse_args = (sparse_args_count <= 256) ? sparse_buffer : new float*[sparse_args_count];
    for (int i = 0; i < sparse_args_count; i++) sparse_args[i] = nullptr;

    int all_var_indices[256];
    int num_vars = nvi + nvj + nvp;
    int idx = 0;
    for (int i = 0; i < nvi; i++) all_var_indices[idx++] = indsi[i];
    for (int j = 0; j < nvj; j++) all_var_indices[idx++] = indsj[j];
    for (int p = 0; p < nvp; p++) all_var_indices[idx++] = indsp[p];
    std::sort(all_var_indices, all_var_indices + num_vars);

    for (int dense_idx = 0; dense_idx < num_vars; dense_idx++) {
        sparse_args[all_var_indices[dense_idx]] = args[dense_idx];
    }

    memcpy(h_args, sparse_args, size_args);
    if (sparse_args_count > 256) delete[] sparse_args;

    cudaMemcpyAsync(device_ptr, h_ptr, total_upload_size, cudaMemcpyHostToDevice, stream);

    size_t shared_mem = cuda_block_size * dimy * KEOPS_DTYPE_BYTES;
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
    float* out = (float*)out_ptr;
    float** args = (float**)args_ptr;
    dim3 blockSize(cuda_block_size);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x);

    int64_t var_counts = (int64_t)argshapes;
    int nvi = var_counts & 0xFF;
    int nvj = (var_counts >> 8) & 0xFF;
    int nvp = (var_counts >> 16) & 0xFF;
    int total_args = nvi + nvj + nvp;

    size_t shared_mem = cuda_block_size * dimy * KEOPS_DTYPE_BYTES;
    size_t args_size = sizeof(float*) * total_args;

    void* pinned_ptr = g_pinned_buffer.ensure(args_size);
    float** h_args;
    
    if (pinned_ptr) {
        h_args = (float**)pinned_ptr;
    } else {
        constexpr size_t FAST_PATH_MAX_ARGS = 32;
        static thread_local float* fallback_buffer[FAST_PATH_MAX_ARGS];
        if (total_args <= FAST_PATH_MAX_ARGS) {
            h_args = fallback_buffer;
        } else {
            static thread_local std::vector<float*> dynamic_args_buffer;
            dynamic_args_buffer.resize(total_args);
            h_args = dynamic_args_buffer.data();
        }
    }

    memcpy(h_args, args, args_size);

    char* device_ptr = (char*)scratch_ptr;
    float** args_d = (float**)device_ptr;

    cudaMemcpyAsync(args_d, h_args, args_size, cudaMemcpyHostToDevice, stream);
    GpuConv1DOnDevice<<<gridSize, blockSize, shared_mem, stream>>>(nx, ny, out, args_d);

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        printf("[LAUNCHER] Kernel launch error: %s\\n", cudaGetErrorString(cuda_err));
        return 1;
    }
    return 0;
}
'''
        return kernel_code + launcher_code