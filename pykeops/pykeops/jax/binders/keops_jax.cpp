/*
 * KeOps JAX C++ Extension (using nanobind) - FULLY OPTIMIZED + DEBUG
 *
 * Optimizations Applied:
 * 1. XLA Scratch Space API (eliminates allocation overhead)
 * 2. C++ Scratch Size Calculation (eliminates Python→C++ overhead, +10-15μs)
 * 3. Pre-computed Static Values in KeOpsKernelInfo
 * 4. Thread-Local Storage with exponential growth
 * 5. Inline dimension extraction
 * 6. Cache-friendly data layouts
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <unordered_map>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>

namespace nb = nanobind;
namespace ffi = xla::ffi;

static const bool KEOPS_DEBUG = []() {
    const char* env = std::getenv("JAX_KEOPS_DEBUG");
    return env && std::string(env) == "1";
}();

#define DEBUG_PRINT(msg) \
    do { \
        if (KEOPS_DEBUG) { \
            std::cout << msg << std::endl; \
        } \
    } while(0)

// =============================================================================
// CUDA Kernel Function Signature
// =============================================================================

typedef int (*LaunchKeOpsKernel)(
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
    int* indsi,
    int* indsj,
    int* indsp,
    int dimout,
    int* dimsx,
    int* dimsy,
    int* dimsp,
    void* ranges,
    void* shapeout,
    void* out_ptr,
    void* args_ptr,
    void* argshapes,
    void* stream,
    void* scratch_ptr
);

// =============================================================================
// Kernel Registry with Pre-computed Values
// =============================================================================

struct KeOpsKernelInfo {
    void* kernel_lib;
    LaunchKeOpsKernel launch_fn;

    // Frequently accessed members (cache-friendly layout)
    int tagHostDevice;
    int dimy;
    int tagI;
    int tagZero;
    int use_half;
    int tag1D2D;
    int dimred;
    int cuda_block_size;
    int use_chunk_mode;
    int dimout;

    std::vector<int> indsi;
    std::vector<int> indsj;
    std::vector<int> indsp;
    std::vector<int> dimsx;
    std::vector<int> dimsy;
    std::vector<int> dimsp;

    // Pre-computed values
    int nvi_count;
    int nvj_count;
    int nvp_count;
    int max_var_idx;
    int64_t var_counts_packed;

    KeOpsKernelInfo() : kernel_lib(nullptr), launch_fn(nullptr),
                        nvi_count(0), nvj_count(0), nvp_count(0),
                        max_var_idx(-1), var_counts_packed(0) {}

    ~KeOpsKernelInfo() {
        if (kernel_lib) {
            dlclose(kernel_lib);
            kernel_lib = nullptr;
        }
    }

    KeOpsKernelInfo(const KeOpsKernelInfo&) = delete;
    KeOpsKernelInfo& operator=(const KeOpsKernelInfo&) = delete;

    KeOpsKernelInfo(KeOpsKernelInfo&& other) noexcept {
        *this = std::move(other);
    }

    KeOpsKernelInfo& operator=(KeOpsKernelInfo&& other) noexcept {
        if (this != &other) {
            if (kernel_lib) dlclose(kernel_lib);

            kernel_lib = other.kernel_lib;
            launch_fn = other.launch_fn;
            tagHostDevice = other.tagHostDevice;
            dimy = other.dimy;
            tagI = other.tagI;
            tagZero = other.tagZero;
            use_half = other.use_half;
            tag1D2D = other.tag1D2D;
            dimred = other.dimred;
            cuda_block_size = other.cuda_block_size;
            use_chunk_mode = other.use_chunk_mode;
            dimout = other.dimout;

            indsi = std::move(other.indsi);
            indsj = std::move(other.indsj);
            indsp = std::move(other.indsp);
            dimsx = std::move(other.dimsx);
            dimsy = std::move(other.dimsy);
            dimsp = std::move(other.dimsp);

            nvi_count = other.nvi_count;
            nvj_count = other.nvj_count;
            nvp_count = other.nvp_count;
            max_var_idx = other.max_var_idx;
            var_counts_packed = other.var_counts_packed;

            other.kernel_lib = nullptr;
        }
        return *this;
    }

    void finalize() {
        nvi_count = static_cast<int>(indsi.size());
        nvj_count = static_cast<int>(indsj.size());
        nvp_count = static_cast<int>(indsp.size());

        if (nvi_count > 255 || nvj_count > 255 || nvp_count > 255) {
            throw std::runtime_error("Too many variables (max 255 per category)");
        }

        max_var_idx = -1;
        for (int idx : indsi) {
            if (idx > max_var_idx) max_var_idx = idx;
        }
        for (int idx : indsj) {
            if (idx > max_var_idx) max_var_idx = idx;
        }
        for (int idx : indsp) {
            if (idx > max_var_idx) max_var_idx = idx;
        }

        var_counts_packed = (nvi_count & 0xFF) |
                           ((nvj_count & 0xFF) << 8) |
                           ((nvp_count & 0xFF) << 16);

        DEBUG_PRINT("[KERNEL INFO] Finalized: nvi=" << nvi_count <<
                   ", nvj=" << nvj_count <<
                   ", nvp=" << nvp_count <<
                   ", max_idx=" << max_var_idx);
    }
};

static std::unordered_map<uint64_t, KeOpsKernelInfo> g_kernel_registry;

// Maximum number of kernels to cache before clearing
// In typical DL workloads, models have fixed structure so this is rarely hit
// If exceeded, it usually means dynamic formula generation (e.g., changing constants)
static const size_t MAX_KERNEL_CACHE_SIZE = 5000;

void cleanup_all_kernels() {
    g_kernel_registry.clear();
}

bool is_kernel_registered(uint64_t kernel_id) {
    return g_kernel_registry.find(kernel_id) != g_kernel_registry.end();
}

size_t get_registry_size() {
    return g_kernel_registry.size();
}

void register_keops_kernel(uint64_t kernel_id, nb::object myconv) {
    if (g_kernel_registry.find(kernel_id) != g_kernel_registry.end()) return;

    // Safety cap to prevent memory leak from dynamic formula generation
    // In normal DL workloads with fixed model structure, this is never hit
    if (g_kernel_registry.size() >= MAX_KERNEL_CACHE_SIZE) {
        std::cerr << "[KeOps JAX] Warning: Kernel registry limit reached ("
                  << MAX_KERNEL_CACHE_SIZE << " kernels). "
                  << "Clearing cache to prevent memory leak. "
                  << "This may indicate dynamic formula generation." << std::endl;
        g_kernel_registry.clear();
    }

    KeOpsKernelInfo info;
    nb::gil_scoped_acquire gil;

    try {
        std::string kernel_path = nb::cast<std::string>(myconv.attr("kernel_so_path"));
        info.kernel_lib = dlopen(kernel_path.c_str(), RTLD_LAZY);
        if (!info.kernel_lib) throw std::runtime_error(std::string("Failed to load .so: ") + dlerror());

        info.launch_fn = (LaunchKeOpsKernel)dlsym(info.kernel_lib, "launch_keops_kernel");
        if (!info.launch_fn) throw std::runtime_error(std::string("Failed to find launch_keops_kernel: ") + dlerror());

        nb::object params = myconv.attr("params");
        info.tagHostDevice = nb::cast<int>(params.attr("tagHostDevice"));
        info.dimy = nb::cast<int>(params.attr("dimy"));
        info.tagI = nb::cast<int>(params.attr("tagI"));
        info.tagZero = nb::cast<int>(params.attr("tagZero"));
        info.use_half = nb::cast<int>(params.attr("use_half"));
        info.tag1D2D = nb::cast<int>(params.attr("tag1D2D"));
        info.dimred = nb::cast<int>(params.attr("dimred"));
        info.cuda_block_size = nb::cast<int>(params.attr("cuda_block_size"));
        info.use_chunk_mode = nb::cast<int>(params.attr("use_chunk_mode"));
        info.dimout = nb::cast<int>(params.attr("dim"));

        info.indsi = nb::cast<std::vector<int>>(params.attr("indsi"));
        info.indsj = nb::cast<std::vector<int>>(params.attr("indsj"));
        info.indsp = nb::cast<std::vector<int>>(params.attr("indsp"));
        info.dimsx = nb::cast<std::vector<int>>(params.attr("dimsx"));
        info.dimsy = nb::cast<std::vector<int>>(params.attr("dimsy"));
        info.dimsp = nb::cast<std::vector<int>>(params.attr("dimsp"));

        info.finalize();

    } catch (const nb::python_error& e) {
        if (info.kernel_lib) dlclose(info.kernel_lib);
        throw std::runtime_error(std::string("Python error during registration: ") + e.what());
    }

    g_kernel_registry[kernel_id] = std::move(info);
}

// =============================================================================
// OPTIMIZED FFI Handler with C++ Scratch Calculation + DEBUG
// =============================================================================

union RangesEncoding {
    void* as_ptr;
    int64_t as_int;
};

ffi::Error KeOpsKernelImpl(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    ffi::RemainingArgs inputs,
    ffi::Result<ffi::AnyBuffer> output,
    int64_t kernel_id,
    int64_t batch_size
) {
    // DEBUG: Function entry
    std::cerr << "\n[KEOPS DEBUG] KeOpsKernelImpl called" << std::endl;
    std::cerr << "  kernel_id: " << kernel_id << std::endl;
    std::cerr << "  batch_size: " << batch_size << std::endl;
    std::cerr.flush();

    auto it = g_kernel_registry.find(static_cast<uint64_t>(kernel_id));
    if (it == g_kernel_registry.end()) return ffi::Error::InvalidArgument("Kernel not found");

    KeOpsKernelInfo& kernel = it->second;
    size_t num_inputs = inputs.size();

    std::cerr << "  num_inputs: " << num_inputs << std::endl;
    std::cerr.flush();

    // PRIORITY 3: Multi-GPU Safety - Device-specific buffers
    // With JAX sharding, same thread can handle multiple devices on different streams
    // Each device needs its own buffer to prevent race conditions and data corruption

    // Get current CUDA device
    int device_id;
    cudaGetDevice(&device_id);

    // Per-device, per-thread storage
    struct PerDeviceBuffers {
        std::vector<void*> input_ptrs;
    };
    static thread_local std::unordered_map<int, PerDeviceBuffers> device_buffers;

    // Get buffer for this specific device
    auto& device_buffer = device_buffers[device_id];
    auto& tls_input_ptrs = device_buffer.input_ptrs;

    // Exponential growth strategy
    tls_input_ptrs.clear();
    if (tls_input_ptrs.capacity() < 16) tls_input_ptrs.reserve(16);
    if (tls_input_ptrs.capacity() < num_inputs) tls_input_ptrs.reserve(num_inputs * 2);

    for (size_t i = 0; i < num_inputs; ++i) {
        auto input_buffer = inputs.get<ffi::AnyBuffer>(i);
        if (!input_buffer.has_value()) return ffi::Error::InvalidArgument("Failed to get input buffer");
        tls_input_ptrs.push_back(const_cast<void*>(input_buffer.value().untyped_data()));
    }

    void* output_ptr = output->untyped_data();

    // DEBUG: Input buffer dimensions
    std::cerr << "\n[KEOPS DEBUG] Input buffer dimensions:" << std::endl;
    for (size_t i = 0; i < std::min(num_inputs, size_t(4)); ++i) {
        auto buf = inputs.get<ffi::AnyBuffer>(i);
        if (buf.has_value()) {
            auto dims = buf->dimensions();
            std::cerr << "  input[" << i << "]: [";
            for (size_t d = 0; d < dims.size(); ++d) {
                std::cerr << dims[d];
                if (d < dims.size() - 1) std::cerr << ", ";
            }
            std::cerr << "]" << std::endl;
        }
    }
    std::cerr.flush();

    // OPTIMIZATION: Extract dimensions directly in C++ (saves 10-15μs)
    int nx = 1, ny = 1;

    if (kernel.nvi_count > 0 && kernel.indsi[0] < num_inputs) {
        auto buf = inputs.get<ffi::AnyBuffer>(kernel.indsi[0]);
        if (buf.has_value()) {
            auto dims = buf->dimensions();
            nx = (dims.size() == 3) ? dims[1] : dims[0];
        }
    }

    if (kernel.nvj_count > 0 && kernel.indsj[0] < num_inputs) {
        auto buf = inputs.get<ffi::AnyBuffer>(kernel.indsj[0]);
        if (buf.has_value()) {
            auto dims = buf->dimensions();
            ny = (dims.size() == 3) ? dims[1] : dims[0];
        }
    }

    int nx_kernel = (kernel.tagI == 1) ? ny : nx;
    int ny_kernel = (kernel.tagI == 1) ? nx : ny;

    // DEBUG: Dimension extraction
    std::cerr << "\n[KEOPS DEBUG] Extracted dimensions:" << std::endl;
    std::cerr << "  nx (from buffer): " << nx << std::endl;
    std::cerr << "  ny (from buffer): " << ny << std::endl;
    std::cerr << "  tagI: " << kernel.tagI << std::endl;
    std::cerr << "  nx_kernel (passed to CUDA): " << nx_kernel << std::endl;
    std::cerr << "  ny_kernel (passed to CUDA): " << ny_kernel << std::endl;
    std::cerr << "  cuda_block_size: " << kernel.cuda_block_size << std::endl;
    std::cerr.flush();

    // OPTIMIZATION: Calculate scratch size in C++ (eliminates Python overhead)
    int blocks_per_batch = (nx_kernel + kernel.cuda_block_size - 1) / kernel.cuda_block_size;
    size_t nblocks = blocks_per_batch * batch_size;
    size_t total_offsets = kernel.nvi_count + kernel.nvj_count + kernel.nvp_count;
    if (total_offsets == 0) total_offsets = 2;

    // Calculate precise scratch buffer size needed
    size_t size_offsets = sizeof(int64_t) * nblocks * total_offsets;
    size_t size_lookup  = sizeof(int64_t) * 3 * nblocks;
    size_t size_slices  = sizeof(int64_t) * batch_size;
    size_t size_ranges  = sizeof(int64_t) * 2 * batch_size;
    size_t size_args    = sizeof(void*) * (kernel.max_var_idx + 1);

    size_t needed_bytes = size_offsets + size_lookup + size_slices + size_ranges + size_args;

    // DEBUG: Scratch buffer calculation
    std::cerr << "\n[KEOPS DEBUG] Scratch buffer calculation:" << std::endl;
    std::cerr << "  blocks_per_batch: " << blocks_per_batch << std::endl;
    std::cerr << "  nblocks (total): " << nblocks << std::endl;
    std::cerr << "  total_offsets: " << total_offsets << std::endl;
    std::cerr << "  needed_bytes: " << needed_bytes << " (" << (needed_bytes/1024.0) << " KB)" << std::endl;
    std::cerr << "    size_offsets: " << size_offsets << " bytes" << std::endl;
    std::cerr << "    size_lookup:  " << size_lookup << " bytes" << std::endl;
    std::cerr << "    size_slices:  " << size_slices << " bytes" << std::endl;
    std::cerr << "    size_ranges:  " << size_ranges << " bytes" << std::endl;
    std::cerr << "    size_args:    " << size_args << " bytes" << std::endl;
    std::cerr.flush();

    // Request scratch buffer from XLA
    auto scratch_result = scratch.Allocate(needed_bytes);
    if (!scratch_result.has_value()) {
        return ffi::Error::Internal("Scratch allocation failed");
    }
    void* scratch_ptr = scratch_result.value();

    std::cerr << "  scratch_ptr: " << scratch_ptr << std::endl;
    std::cerr.flush();

    // Debug output (only when enabled)
    if (KEOPS_DEBUG) {
        std::cout << "[FFI] XLA Scratch Space enabled" << std::endl;
        std::cout << "[FFI] Calculated scratch size: " << needed_bytes << " bytes" << std::endl;
        std::cout << "[FFI] nx=" << nx_kernel << ", ny=" << ny_kernel << ", batch=" << batch_size << std::endl;
        std::cout << "[FFI] Variable counts: nvi=" << kernel.nvi_count
                  << ", nvj=" << kernel.nvj_count
                  << ", nvp=" << kernel.nvp_count << std::endl;
    }

    RangesEncoding ranges_enc;
    ranges_enc.as_int = (batch_size > 1) ? batch_size : 0;
    if (batch_size <= 1) ranges_enc.as_ptr = nullptr;

    // DEBUG: Ranges encoding
    std::cerr << "\n[KEOPS DEBUG] Ranges encoding:" << std::endl;
    std::cerr << "  batch_size > 1: " << (batch_size > 1 ? "true" : "false") << std::endl;
    std::cerr << "  ranges_enc.as_int: " << ranges_enc.as_int << std::endl;
    std::cerr << "  ranges_enc.as_ptr: " << ranges_enc.as_ptr << std::endl;
    std::cerr.flush();

    void* argshapes_ptr = (void*)kernel.var_counts_packed;

    std::cerr << "\n[KEOPS DEBUG] Calling CUDA kernel..." << std::endl;
    std::cerr.flush();

    int result = kernel.launch_fn(
        kernel.tagHostDevice, kernel.dimy, nx_kernel, ny_kernel, kernel.tagI, kernel.tagZero,
        kernel.use_half, kernel.tag1D2D, kernel.dimred, kernel.cuda_block_size, kernel.use_chunk_mode,
        kernel.indsi.data(), kernel.indsj.data(), kernel.indsp.data(),
        kernel.dimout, kernel.dimsx.data(), kernel.dimsy.data(), kernel.dimsp.data(),
        ranges_enc.as_ptr, nullptr, output_ptr, tls_input_ptrs.data(), argshapes_ptr,
        (void*)stream, scratch_ptr
    );

    // DEBUG: Kernel result
    std::cerr << "\n[KEOPS DEBUG] Kernel result:" << std::endl;
    std::cerr << "  result code: " << result << std::endl;

    cudaError_t err = cudaGetLastError();
    std::cerr << "  CUDA error: " << cudaGetErrorString(err) << " (code " << err << ")" << std::endl;
    std::cerr << "========================================\n" << std::endl;
    std::cerr.flush();

    if (result != 0) return ffi::Error::Internal("Kernel launch failed: " + std::to_string(result));

    if (err != cudaSuccess) return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(err));

    return ffi::Error::Success();
}

// FFI Handler Symbol (version-compatible, no kCmdBufferCompatible)
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KeOpsKernel, KeOpsKernelImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .RemainingArgs()
        .Ret<ffi::AnyBuffer>()
        .Attr<int64_t>("kernel_id")
        .Attr<int64_t>("batch_size")
);

template <typename T> nb::capsule EncapsulateFfiCall(T *fn) {
    return nb::capsule(reinterpret_cast<void *>(fn));
}

NB_MODULE(keops_jax_ext, m) {
    m.def("register_keops_kernel", &register_keops_kernel);
    m.def("is_kernel_registered", &is_kernel_registered);
    m.def("cleanup_all_kernels", &cleanup_all_kernels);
    m.def("get_registry_size", &get_registry_size, "Get the current number of registered kernels");
    m.def("get_ffi_handler", []() { return EncapsulateFfiCall(KeOpsKernel); });
}