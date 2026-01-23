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


class Gpu_link_compile(LinkCompile):
    """
    CMake-based GPU compilation (alternative to NVRTC)
    Uses nvcc directly to compile .cu files to .so shared libraries
    Includes a launcher function for GIL-free execution from C++
    """
    source_code_extension = "cu"
    ngpu, gpu_props_compile_flags = get_gpu_props

    def __init__(self):
        # checking that the system has a Gpu :
        if not (cuda_available and Gpu_link_compile.ngpu):
            KeOps_Error(
                "Trying to compile cuda code... but we detected that the system has no properly configured cuda lib."
            )

        LinkCompile.__init__(self)

        # For CMake compilation, we compile directly to a .so file
        self.low_level_code_file = "".encode("utf-8")  # Not used in CMake mode

        # The actual .so file to be loaded
        self.so_file = os.path.join(
            build_folder,
            self.gencode_filename + sysconfig.get_config_var("SHLIB_SUFFIX"),
        )

        # actual dll to be called
        self.true_dllname = self.so_file
        # file to check for existence to detect compilation is needed
        self.file_to_check = self.so_file

    def save_info(self):
        """
        Override save_info to handle missing 'dim' attribute in JAX mode.
        """
        red_formula = getattr(self, 'red_formula_string', "Unknown")
        # Fallback logic for 'dim'
        dim = getattr(self, 'dim', getattr(self, 'dimout', getattr(self, 'dimy', 0)))
        tagI = getattr(self, 'tagI', 0)
        dimy = getattr(self, 'dimy', 0)

        info_str = f"red_formula={red_formula}\ndim={dim}\ntagI={tagI}\ndimy={dimy}"

        info_file = self.so_file + ".info"
        with open(info_file, "w") as f:
            f.write(info_str)

    def generate_code(self):
        # method to generate the code and compile it
        self.get_code()
        self.code = self.add_launcher_wrapper(self.code)
        self.write_code()

        # Compile using nvcc directly
        KeOps_Message("Compiling formula using CMake/nvcc ... ", flush=True, end="")

        cuda_include = cuda_config.get_cuda_include_path()

        # Build compile flags
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

        # Add GPU architecture flags
        gpu_arch_flags = cuda_config.get_gpu_arch_flags()
        if gpu_arch_flags:
            compile_flags.append(gpu_arch_flags)

        if Gpu_link_compile.gpu_props_compile_flags:
            compile_flags.append(Gpu_link_compile.gpu_props_compile_flags)

        # Add source and output
        compile_flags.extend([
            self.gencode_file,
            "-o", self.so_file
        ])

        command = " ".join(compile_flags)

        KeOps_OS_Run(command)

        if not os.path.exists(self.so_file):
            KeOps_Error(f"CMake compilation succeeded but .so file not found: {self.so_file}")

        KeOps_Message("OK", use_tag=False, flush=True)

        # retrieve parameters for info file
        self.tagI = getattr(self.red_formula, 'tagI', 0)
        self.dim = getattr(self.red_formula, 'dim', getattr(self.red_formula, 'dimout', 0))

    def add_launcher_wrapper(self, kernel_code):
        """
        Add a C-compatible launcher function that can be called from C++ via dlopen
        """
        is_ranges = "GpuConv1DOnDevice_ranges" in kernel_code

        preamble = """
#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <algorithm>
#include <unordered_map>

using std::min;

#define CHECK_CUDA_LAUNCH(err) \\
    if (err != cudaSuccess) { \\
        printf("[LAUNCHER] CUDA Error: %s\\n", cudaGetErrorString(err)); \\
        return 1; \\
    }
"""

        if is_ranges:
            launcher_code = preamble + """
// =============================================================================
// Launch Parameter Cache - Avoids redundant CPU computation and GPU uploads
// =============================================================================

struct LaunchCache {
    int nx = -1;
    int ny = -1;
    int batch_size = -1;
    int cuda_block_size = -1;

    // Cached GPU memory (persisted across calls)
    void* cached_scratch = nullptr;
    size_t cached_scratch_size = 0;

    // Cached host buffer for args (still need to update args each call)
    std::vector<char> host_args_buffer;

    // Precomputed sizes
    size_t offsets_size = 0;
    size_t lookup_size = 0;
    size_t slices_size = 0;
    size_t ranges_size = 0;
    size_t tables_total_size = 0;  // Everything except args

    bool is_valid(int _nx, int _ny, int _batch_size, int _cuda_block_size) const {
        return nx == _nx && ny == _ny && batch_size == _batch_size && 
               cuda_block_size == _cuda_block_size && cached_scratch != nullptr;
    }

    void invalidate() {
        nx = ny = batch_size = cuda_block_size = -1;
        // Note: We don't free cached_scratch here - it's managed by XLA's scratch allocator
        cached_scratch = nullptr;
    }
};

// Thread-local cache per device
static thread_local std::unordered_map<int, LaunchCache> g_launch_caches;

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

    // Unpack variable counts from argshapes parameter
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

    // Get device-specific cache
    int device_id;
    cudaGetDevice(&device_id);
    LaunchCache& cache = g_launch_caches[device_id];

    char* device_ptr = (char*)scratch_ptr;

    // Size calculations
    size_t size_offsets = sizeof(signed long int) * nblocks * total_offsets;
    size_t size_lookup  = sizeof(signed long int) * 3 * nblocks;
    size_t size_slices  = sizeof(signed long int) * batch_size;
    size_t size_ranges  = sizeof(signed long int) * 2 * batch_size;
    size_t size_args    = sizeof(float*) * sparse_args_count;
    size_t tables_size  = size_offsets + size_lookup + size_slices + size_ranges;

    // Device pointers
    signed long int* offsets_d = (signed long int*)(device_ptr);
    signed long int* lookup_d  = (signed long int*)(device_ptr + size_offsets);
    signed long int* slices_x  = (signed long int*)(device_ptr + size_offsets + size_lookup);
    signed long int* ranges_y  = (signed long int*)(device_ptr + size_offsets + size_lookup + size_slices);
    float** args_d             = (float**)(device_ptr + tables_size);

    // Check if we can use cached tables (dimensions unchanged)
    bool cache_hit = cache.is_valid(nx, ny, batch_size, cuda_block_size) && 
                     cache.cached_scratch == scratch_ptr;

    if (!cache_hit) {
        // Cache miss: need to rebuild and upload offset/lookup/slice/range tables

        // Ensure host buffer is large enough for tables
        static thread_local std::vector<char> host_tables_buffer;
        if (host_tables_buffer.size() < tables_size) {
            host_tables_buffer.resize(tables_size * 2);
        }
        char* h_ptr = host_tables_buffer.data();

        int blocks_per_batch = (nx + blockSize.x - 1) / blockSize.x;

        // Build offsets table
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

        // Build lookup table
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

        // Build slices
        signed long int* h_slices = (signed long int*)(h_ptr + size_offsets + size_lookup);
        for (int b = 0; b < batch_size; b++) { h_slices[b] = b + 1; }

        // Build ranges
        signed long int* h_ranges = (signed long int*)(h_ptr + size_offsets + size_lookup + size_slices);
        for (int b = 0; b < batch_size; b++) { 
            h_ranges[2*b + 0] = b * ny;
            h_ranges[2*b + 1] = (b+1) * ny;
        }

        // Upload tables to GPU
        cudaMemcpyAsync(device_ptr, h_ptr, tables_size, cudaMemcpyHostToDevice, stream);

        // Update cache
        cache.nx = nx;
        cache.ny = ny;
        cache.batch_size = batch_size;
        cache.cuda_block_size = cuda_block_size;
        cache.cached_scratch = scratch_ptr;
        cache.tables_total_size = tables_size;
    }
    // If cache_hit, tables are already on GPU at the right location - skip rebuild & upload!

    // Args ALWAYS need to be updated (data pointers change each call)
    // But this is much smaller than the full tables

    // Ensure host args buffer is large enough
    if (cache.host_args_buffer.size() < size_args) {
        cache.host_args_buffer.resize(size_args * 2);
    }
    float** h_args = (float**)cache.host_args_buffer.data();

    // Handle sparse variable indices
    std::vector<int> all_var_indices;
    all_var_indices.reserve(nvi + nvj + nvp);
    for (int i = 0; i < nvi; i++) all_var_indices.push_back(indsi[i]);
    for (int j = 0; j < nvj; j++) all_var_indices.push_back(indsj[j]);
    for (int p = 0; p < nvp; p++) all_var_indices.push_back(indsp[p]);
    std::sort(all_var_indices.begin(), all_var_indices.end());

    float* sparse_buffer[256];
    float** sparse_args = (sparse_args_count <= 256) ? sparse_buffer : new float*[sparse_args_count];
    for (int i = 0; i < sparse_args_count; i++) sparse_args[i] = nullptr;

    for (size_t dense_idx = 0; dense_idx < all_var_indices.size(); dense_idx++) {
        sparse_args[all_var_indices[dense_idx]] = args[dense_idx];
    }

    memcpy(h_args, sparse_args, size_args);
    if (sparse_args_count > 256) delete[] sparse_args;

    // Upload only args (small, always needed)
    cudaMemcpyAsync(args_d, h_args, size_args, cudaMemcpyHostToDevice, stream);

    size_t shared_mem = cuda_block_size * dimy * sizeof(float);
    GpuConv1DOnDevice_ranges<<<gridSize, blockSize, shared_mem, stream>>>(
        nx, ny, nbatchdims, offsets_d, lookup_d, slices_x, ranges_y, out, args_d);

    return (cudaGetLastError() == cudaSuccess) ? 0 : 1;
}
"""
        else:
            launcher_code = preamble + """
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

    size_t shared_mem = cuda_block_size * dimy * sizeof(float);
    size_t args_size = sizeof(float*) * total_args;

    // Fast path: use stack buffer for common case (most kernels have < 32 args)
    constexpr size_t FAST_PATH_MAX_ARGS = 32;
    float* fixed_args_buffer[FAST_PATH_MAX_ARGS];

    float** h_args;
    if (total_args <= FAST_PATH_MAX_ARGS) {
        h_args = fixed_args_buffer;
    } else {
        // Fallback for unusual kernels
        static thread_local std::vector<float*> dynamic_args_buffer;
        dynamic_args_buffer.resize(total_args);
        h_args = dynamic_args_buffer.data();
    }

    memcpy(h_args, args, args_size);

    char* device_ptr = (char*)scratch_ptr;
    float** args_d = (float**)device_ptr;

    cudaMemcpyAsync(args_d, h_args, args_size, cudaMemcpyHostToDevice, stream);
    GpuConv1DOnDevice<<<gridSize, blockSize, shared_mem, stream>>>(nx, ny, out, args_d);

    return (cudaGetLastError() == cudaSuccess) ? 0 : 1;
}
"""
        return kernel_code + launcher_code