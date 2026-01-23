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

    char* device_ptr = (char*)scratch_ptr;

    // KERNEL EXPECTS 64-BIT POINTERS (signed long*)
    size_t size_offsets = sizeof(signed long int) * nblocks * total_offsets;
    size_t size_lookup  = sizeof(signed long int) * 3 * nblocks;
    size_t size_slices  = sizeof(signed long int) * batch_size;
    size_t size_ranges  = sizeof(signed long int) * 2 * batch_size;

    // Cast to signed long* to match kernel signature
    signed long int* offsets_d = (signed long int*)(device_ptr);
    signed long int* lookup_d  = (signed long int*)(device_ptr + size_offsets);
    signed long int* slices_x  = (signed long int*)(device_ptr + size_offsets + size_lookup);
    signed long int* ranges_y  = (signed long int*)(device_ptr + size_offsets + size_lookup + size_slices);
    float** args_d             = (float**)(device_ptr + size_offsets + size_lookup + size_slices + size_ranges);

    // Host staging buffer
    struct PerDeviceHostBuffer { std::vector<char> buffer; };
    static thread_local std::unordered_map<int, PerDeviceHostBuffer> device_host_buffers;
    int device_id;
    cudaGetDevice(&device_id);
    auto& host_buffer = device_host_buffers[device_id].buffer;

    size_t total_size = size_offsets + size_lookup + size_slices + size_ranges + (sizeof(float*) * sparse_args_count);
    if (host_buffer.size() < total_size) host_buffer.resize(total_size * 2);
    char* h_ptr = host_buffer.data();

    int blocks_per_batch = (nx + blockSize.x - 1) / blockSize.x;

    // Build offsets table (using signed long int)
    signed long int* h_offsets = (signed long int*)h_ptr;
    for (int b = 0; b < batch_size; b++) {
        for (int block_in_batch = 0; block_in_batch < blocks_per_batch; block_in_batch++) {
            int block_id = b * blocks_per_batch + block_in_batch;
            int offset_idx = total_offsets * block_id;

            // CRITICAL FIX: Calculate the block's offset within the batch
            int block_offset = block_in_batch * blockSize.x;

            // For Vi (i-variables), we MUST add the block_offset because the kernel 
            // uses threadIdx.x (0..BlockSize) relative to this pointer!
            for (int i = 0; i < nvi; i++) h_offsets[offset_idx + i] = b * nx + block_offset;

            // For Vj (j-variables), we scan the whole axis, so no block offset needed
            for (int j = 0; j < nvj; j++) h_offsets[offset_idx + nvi + j] = b * ny;

            // Parameters are constant
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

    float** h_args = (float**)(h_ptr + size_offsets + size_lookup + size_slices + size_ranges);

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

    memcpy(h_args, sparse_args, sizeof(float*) * sparse_args_count);
    if (sparse_args_count > 256) delete[] sparse_args;

    cudaMemcpyAsync(device_ptr, h_ptr, total_size, cudaMemcpyHostToDevice, stream);

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

    size_t shared_mem = cuda_block_size * dimy * sizeof(float);

    struct PerDeviceHostBuffer { std::vector<char> buffer; };
    static thread_local std::unordered_map<int, PerDeviceHostBuffer> device_host_buffers;
    int device_id;
    cudaGetDevice(&device_id);
    auto& host_buffer = device_host_buffers[device_id].buffer;
    size_t args_size = sizeof(float*) * (nvi + nvj + nvp);
    if (host_buffer.size() < args_size) host_buffer.resize(args_size * 2);

    float** h_args = (float**)host_buffer.data();
    memcpy(h_args, args, args_size);

    char* device_ptr = (char*)scratch_ptr;
    float** args_d = (float**)device_ptr;

    cudaMemcpyAsync(args_d, h_args, args_size, cudaMemcpyHostToDevice, stream);
    GpuConv1DOnDevice<<<gridSize, blockSize, shared_mem, stream>>>(nx, ny, out, args_d);

    return (cudaGetLastError() == cudaSuccess) ? 0 : 1;
}
"""
        return kernel_code + launcher_code