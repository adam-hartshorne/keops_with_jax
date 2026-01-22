/*
 * KeOps JAX C++ Extension (using nanobind)
 * XLA SCRATCH SPACE VERSION - January 2026
 *
 * Key Feature: Uses XLA-managed scratch buffers instead of cudaMallocAsync
 *
 * Optimizations:
 * 1. XLA Scratch Space API (NEW - eliminates allocation overhead)
 * 2. Pre-computed static values in KeOpsKernelInfo
 * 3. Debug prints wrapped in KEOPS_DEBUG checks
 * 4. Optimized dimension extraction
 * 5. Thread-Local Storage for speed
 * 6. Proper Resource Cleanup (dlclose)
 * 7. Safe Union-based Type Punning
 * 8. Vector Pre-allocation
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
// CUDA Kernel Function Signature (Updated for Scratch Buffer)
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
    void* scratch_ptr  // NEW: XLA-provided scratch buffer
);

// =============================================================================
// Kernel Registry with Pre-computed Values
// =============================================================================

struct KeOpsKernelInfo {
    void* kernel_lib;
    LaunchKeOpsKernel launch_fn;

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

    // OPTIMIZATION: Pre-computed values
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
            tagHostDevice = other.tagHostDevice; dimy = other.dimy; tagI = other.tagI;
            tagZero = other.tagZero; use_half = other.use_half; tag1D2D = other.tag1D2D;
            dimred = other.dimred; cuda_block_size = other.cuda_block_size;
            use_chunk_mode = other.use_chunk_mode; dimout = other.dimout;
            indsi = std::move(other.indsi); indsj = std::move(other.indsj); indsp = std::move(other.indsp);
            dimsx = std::move(other.dimsx); dimsy = std::move(other.dimsy); dimsp = std::move(other.dimsp);

            // Copy pre-computed values
            nvi_count = other.nvi_count;
            nvj_count = other.nvj_count;
            nvp_count = other.nvp_count;
            max_var_idx = other.max_var_idx;
            var_counts_packed = other.var_counts_packed;

            other.kernel_lib = nullptr;
        }
        return *this;
    }

    // OPTIMIZATION: Compute static values once during registration
    void finalize() {
        nvi_count = static_cast<int>(indsi.size());
        nvj_count = static_cast<int>(indsj.size());
        nvp_count = static_cast<int>(indsp.size());

        // Safety check before packing
        if (nvi_count > 255 || nvj_count > 255 || nvp_count > 255) {
            throw std::runtime_error("Too many variables (max 255 per category)");
        }

        // Compute max variable index
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

        // Pre-pack variable counts
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

void cleanup_all_kernels() {
    g_kernel_registry.clear();
}

bool is_kernel_registered(uint64_t kernel_id) {
    return g_kernel_registry.find(kernel_id) != g_kernel_registry.end();
}

void register_keops_kernel(uint64_t kernel_id, nb::object myconv) {
    if (g_kernel_registry.find(kernel_id) != g_kernel_registry.end()) return;

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

        // OPTIMIZATION: Pre-compute static values
        info.finalize();

    } catch (const nb::python_error& e) {
        if (info.kernel_lib) dlclose(info.kernel_lib);
        throw std::runtime_error(std::string("Python error during registration: ") + e.what());
    }

    g_kernel_registry[kernel_id] = std::move(info);
}

// =============================================================================
// FFI Handler with XLA Scratch Space
// =============================================================================

union RangesEncoding {
    void* as_ptr;
    int64_t as_int;
};

ffi::Error KeOpsKernelImpl(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,  // NEW: XLA scratch allocator
    ffi::RemainingArgs inputs,
    ffi::Result<ffi::AnyBuffer> output,
    int64_t kernel_id,
    int64_t batch_size,
    int64_t max_scratch_bytes  // NEW: Pre-computed max scratch size
) {
    auto it = g_kernel_registry.find(static_cast<uint64_t>(kernel_id));
    if (it == g_kernel_registry.end()) return ffi::Error::InvalidArgument("Kernel not found");

    KeOpsKernelInfo& kernel = it->second;
    size_t num_inputs = inputs.size();

    // OPTIMIZATION: Thread-local storage with pre-allocation
    static thread_local std::vector<void*> tls_input_ptrs;
    tls_input_ptrs.clear();
    if (tls_input_ptrs.capacity() < 16) tls_input_ptrs.reserve(16);
    if (tls_input_ptrs.capacity() < num_inputs) tls_input_ptrs.reserve(num_inputs * 2);

    for (size_t i = 0; i < num_inputs; ++i) {
        auto input_buffer = inputs.get<ffi::AnyBuffer>(i);
        if (!input_buffer.has_value()) return ffi::Error::InvalidArgument("Failed to get input buffer");
        tls_input_ptrs.push_back(const_cast<void*>(input_buffer.value().untyped_data()));
    }

    void* output_ptr = output->untyped_data();

    // OPTIMIZATION: Improved dimension extraction
    int nx = 1, ny = 1;
    if (!kernel.indsi.empty() && kernel.indsi[0] < num_inputs) {
        auto buf = inputs.get<ffi::AnyBuffer>(kernel.indsi[0]);
        if (buf.has_value()) {
            auto dims = buf->dimensions();
            nx = (dims.size() == 3) ? dims[1] : (dims.size() > 0 ? dims[0] : 1);
        }
    }
    if (!kernel.indsj.empty() && kernel.indsj[0] < num_inputs) {
        auto buf = inputs.get<ffi::AnyBuffer>(kernel.indsj[0]);
        if (buf.has_value()) {
            auto dims = buf->dimensions();
            ny = (dims.size() == 3) ? dims[1] : (dims.size() > 0 ? dims[0] : 1);
        }
    }

    int nx_kernel = (kernel.tagI == 1) ? ny : nx;
    int ny_kernel = (kernel.tagI == 1) ? nx : ny;

    // NEW: Request scratch buffer from XLA
    // Compatible with older JAX versions where Allocate returns std::optional<void*>
    auto scratch_result = scratch.Allocate(max_scratch_bytes);
    if (!scratch_result.has_value()) {
        return ffi::Error::Internal("Scratch allocation failed");
    }
    void* scratch_ptr = scratch_result.value();

    // OPTIMIZATION: Debug prints only when enabled
    if (KEOPS_DEBUG) {
        std::cout << "[FFI] XLA Scratch Space enabled" << std::endl;
        std::cout << "[FFI] Scratch size: " << max_scratch_bytes << " bytes" << std::endl;
        std::cout << "[FFI] Variable counts: nvi=" << kernel.nvi_count
                  << ", nvj=" << kernel.nvj_count
                  << ", nvp=" << kernel.nvp_count << std::endl;
    }

    RangesEncoding ranges_enc;
    ranges_enc.as_int = (batch_size > 1) ? batch_size : 0;
    if (batch_size <= 1) ranges_enc.as_ptr = nullptr;

    // OPTIMIZATION: Use pre-computed packed value
    void* argshapes_ptr = (void*)kernel.var_counts_packed;

    int result = kernel.launch_fn(
        kernel.tagHostDevice, kernel.dimy, nx_kernel, ny_kernel, kernel.tagI, kernel.tagZero,
        kernel.use_half, kernel.tag1D2D, kernel.dimred, kernel.cuda_block_size, kernel.use_chunk_mode,
        kernel.indsi.data(), kernel.indsj.data(), kernel.indsp.data(),
        kernel.dimout, kernel.dimsx.data(), kernel.dimsy.data(), kernel.dimsp.data(),
        ranges_enc.as_ptr, nullptr, output_ptr, tls_input_ptrs.data(), argshapes_ptr,
        (void*)stream, scratch_ptr  // NEW: Pass XLA scratch buffer
    );

    if (result != 0) return ffi::Error::Internal("Kernel launch failed: " + std::to_string(result));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(err));

    return ffi::Error::Success();
}

// FFI Handler Symbol with Scratch Space Support
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KeOpsKernel, KeOpsKernelImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()  // NEW: Request scratch allocator
        .RemainingArgs()
        .Ret<ffi::AnyBuffer>()
        .Attr<int64_t>("kernel_id")
        .Attr<int64_t>("batch_size")
        .Attr<int64_t>("max_scratch_bytes")  // NEW: Scratch size attribute
);

template <typename T> nb::capsule EncapsulateFfiCall(T *fn) {
    return nb::capsule(reinterpret_cast<void *>(fn));
}

NB_MODULE(keops_jax_ext, m) {
    m.def("register_keops_kernel", &register_keops_kernel);
    m.def("is_kernel_registered", &is_kernel_registered);
    m.def("cleanup_all_kernels", &cleanup_all_kernels);
    m.def("get_ffi_handler", []() { return EncapsulateFfiCall(KeOpsKernel); });
}