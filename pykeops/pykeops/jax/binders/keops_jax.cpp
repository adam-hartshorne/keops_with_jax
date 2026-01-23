/*
 * KeOps JAX C++ Extension (using nanobind) - FULLY OPTIMIZED
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <atomic>

namespace nb = nanobind;
namespace ffi = xla::ffi;

static const bool KEOPS_DEBUG = []() {
    const char* env = std::getenv("JAX_KEOPS_DEBUG");
    return env && std::string(env) == "1";
}();

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
// Kernel Registry
// =============================================================================

struct KeOpsKernelInfo {
    void* kernel_lib;
    LaunchKeOpsKernel launch_fn;

    int tagHostDevice, dimy, tagI, tagZero, use_half, tag1D2D, dimred, cuda_block_size, use_chunk_mode, dimout;
    std::vector<int> indsi, indsj, indsp, dimsx, dimsy, dimsp;
    int nvi_count, nvj_count, nvp_count, max_var_idx;
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

    // Move constructor/assignment (no copy)
    KeOpsKernelInfo(const KeOpsKernelInfo&) = delete;
    KeOpsKernelInfo& operator=(const KeOpsKernelInfo&) = delete;
    KeOpsKernelInfo(KeOpsKernelInfo&& other) noexcept { *this = std::move(other); }
    KeOpsKernelInfo& operator=(KeOpsKernelInfo&& other) noexcept {
        if (this != &other) {
            if (kernel_lib) dlclose(kernel_lib);
            kernel_lib = other.kernel_lib;
            launch_fn = other.launch_fn;
            tagHostDevice = other.tagHostDevice; dimy = other.dimy; tagI = other.tagI;
            tagZero = other.tagZero; use_half = other.use_half; tag1D2D = other.tag1D2D;
            dimred = other.dimred; cuda_block_size = other.cuda_block_size;
            use_chunk_mode = other.use_chunk_mode; dimout = other.dimout;
            indsi = std::move(other.indsi); indsj = std::move(other.indsj);
            indsp = std::move(other.indsp); dimsx = std::move(other.dimsx);
            dimsy = std::move(other.dimsy); dimsp = std::move(other.dimsp);
            nvi_count = other.nvi_count; nvj_count = other.nvj_count;
            nvp_count = other.nvp_count; max_var_idx = other.max_var_idx;
            var_counts_packed = other.var_counts_packed;
            other.kernel_lib = nullptr;
        }
        return *this;
    }

    void finalize() {
        nvi_count = static_cast<int>(indsi.size());
        nvj_count = static_cast<int>(indsj.size());
        nvp_count = static_cast<int>(indsp.size());
        max_var_idx = -1;
        for (int idx : indsi) if (idx > max_var_idx) max_var_idx = idx;
        for (int idx : indsj) if (idx > max_var_idx) max_var_idx = idx;
        for (int idx : indsp) if (idx > max_var_idx) max_var_idx = idx;
        var_counts_packed = (nvi_count & 0xFF) | ((nvj_count & 0xFF) << 8) | ((nvp_count & 0xFF) << 16);
    }
};

static std::unordered_map<uint64_t, KeOpsKernelInfo> g_kernel_registry;
static const size_t MAX_KERNEL_CACHE_SIZE = 5000;

// Global version counter - incremented on any registry modification
// Used to detect when thread-local caches become stale
static std::atomic<uint64_t> g_registry_version{0};

// Helper to create composite key from kernel_id and device_id
// This ensures kernels compiled for different GPUs don't conflict
inline uint64_t make_registry_key(uint64_t kernel_id, int device_id) {
    // Use upper 16 bits for device_id, lower 48 bits for kernel_id
    // Supports up to 65535 devices and 281 trillion unique kernel IDs
    return (static_cast<uint64_t>(device_id & 0xFFFF) << 48) | (kernel_id & 0x0000FFFFFFFFFFFF);
}

inline std::pair<uint64_t, int> split_registry_key(uint64_t key) {
    int device_id = static_cast<int>((key >> 48) & 0xFFFF);
    uint64_t kernel_id = key & 0x0000FFFFFFFFFFFF;
    return {kernel_id, device_id};
}

// Thread-local kernel cache for fast repeated lookups
// Avoids hash map overhead when the same kernel is called repeatedly
struct ThreadLocalKernelCache {
    uint64_t kernel_id = 0;
    int device_id = -1;
    uint64_t registry_version = 0;
    KeOpsKernelInfo* kernel_ptr = nullptr;

    void invalidate() {
        kernel_id = 0;
        device_id = -1;
        registry_version = 0;
        kernel_ptr = nullptr;
    }

    bool is_valid(uint64_t kid, int dev_id, uint64_t version) const {
        return kernel_id == kid && device_id == dev_id &&
               registry_version == version && kernel_ptr != nullptr;
    }
};

void cleanup_all_kernels() {
    g_kernel_registry.clear();
    g_registry_version++;  // Invalidate all thread-local caches
}

// Check if kernel is registered for the CURRENT device
bool is_kernel_registered(uint64_t kernel_id) {
    int device_id;
    cudaGetDevice(&device_id);
    uint64_t key = make_registry_key(kernel_id, device_id);
    return g_kernel_registry.find(key) != g_kernel_registry.end();
}

// Check if kernel is registered for a SPECIFIC device
bool is_kernel_registered_on_device(uint64_t kernel_id, int device_id) {
    uint64_t key = make_registry_key(kernel_id, device_id);
    return g_kernel_registry.find(key) != g_kernel_registry.end();
}

size_t get_registry_size() { return g_kernel_registry.size(); }

// Get number of unique kernels (across all devices)
size_t get_unique_kernel_count() {
    std::unordered_set<uint64_t> unique_kernels;
    for (const auto& pair : g_kernel_registry) {
        auto [kernel_id, device_id] = split_registry_key(pair.first);
        unique_kernels.insert(kernel_id);
    }
    return unique_kernels.size();
}

void register_keops_kernel(uint64_t kernel_id, nb::object myconv) {
    // Get current device - kernel will be registered for THIS device
    int device_id;
    cudaGetDevice(&device_id);

    uint64_t key = make_registry_key(kernel_id, device_id);

    if (g_kernel_registry.find(key) != g_kernel_registry.end()) return;
    if (g_kernel_registry.size() >= MAX_KERNEL_CACHE_SIZE) {
        g_kernel_registry.clear();
        g_registry_version++;  // Invalidate all thread-local caches
    }

    KeOpsKernelInfo info;
    nb::gil_scoped_acquire gil;

    try {
        std::string kernel_path = nb::cast<std::string>(myconv.attr("kernel_so_path"));
        info.kernel_lib = dlopen(kernel_path.c_str(), RTLD_LAZY);
        if (!info.kernel_lib) throw std::runtime_error("Failed to load .so: " + std::string(dlerror()));

        info.launch_fn = (LaunchKeOpsKernel)dlsym(info.kernel_lib, "launch_keops_kernel");
        if (!info.launch_fn) throw std::runtime_error("Failed to find launch_keops_kernel: " + std::string(dlerror()));

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
        throw std::runtime_error("Python error during registration: " + std::string(e.what()));
    }
    g_kernel_registry[key] = std::move(info);

    if (KEOPS_DEBUG) {
        std::cout << "[KeOps] Registered kernel " << kernel_id << " for device " << device_id << std::endl;
    }
}

// =============================================================================
// FFI Implementation
// =============================================================================

ffi::Error KeOpsKernelImpl(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    ffi::RemainingArgs inputs,
    ffi::Result<ffi::AnyBuffer> output,
    int64_t kernel_id,
    int64_t batch_size
) {
    // Thread-local kernel cache for fast repeated lookups
    static thread_local ThreadLocalKernelCache tls_kernel_cache;

    // Get current device - REQUIRED for per-device registry lookup
    int device_id;
    cudaGetDevice(&device_id);

    uint64_t kid = static_cast<uint64_t>(kernel_id);
    uint64_t current_version = g_registry_version.load(std::memory_order_acquire);
    KeOpsKernelInfo* kernel_ptr = nullptr;

    // Fast path: check thread-local cache first
    // Only valid if kernel_id AND device_id match AND registry version hasn't changed
    if (tls_kernel_cache.is_valid(kid, device_id, current_version)) {
        kernel_ptr = tls_kernel_cache.kernel_ptr;
    } else {
        // Slow path: hash map lookup with composite key
        uint64_t key = make_registry_key(kid, device_id);
        auto it = g_kernel_registry.find(key);
        if (it == g_kernel_registry.end()) {
            return ffi::Error::InvalidArgument(
                "Kernel " + std::to_string(kid) + " not found for device " + std::to_string(device_id)
            );
        }
        kernel_ptr = &(it->second);

        // Update thread-local cache
        tls_kernel_cache.kernel_id = kid;
        tls_kernel_cache.device_id = device_id;
        tls_kernel_cache.registry_version = current_version;
        tls_kernel_cache.kernel_ptr = kernel_ptr;
    }

    KeOpsKernelInfo& kernel = *kernel_ptr;
    size_t num_inputs = inputs.size();

    // Fast path: use stack-allocated array for common case (most kernels have < 16 inputs)
    // This avoids cudaGetDevice(), hash map lookup, and vector operations
    constexpr size_t FAST_PATH_MAX_INPUTS = 16;
    void* fixed_input_ptrs[FAST_PATH_MAX_INPUTS];

    // Fallback for unusual kernels with many inputs
    static thread_local std::vector<void*> dynamic_input_ptrs;

    void** input_ptrs;
    if (num_inputs <= FAST_PATH_MAX_INPUTS) {
        input_ptrs = fixed_input_ptrs;
    } else {
        dynamic_input_ptrs.resize(num_inputs);
        input_ptrs = dynamic_input_ptrs.data();
    }

    // Populate input pointers
    for (size_t i = 0; i < num_inputs; ++i) {
        input_ptrs[i] = const_cast<void*>(inputs.get<ffi::AnyBuffer>(i).value().untyped_data());
    }
    void* output_ptr = output->untyped_data();

    // Dimension extraction
    int nx = 1, ny = 1;
    if (kernel.nvi_count > 0 && kernel.indsi[0] < static_cast<int>(num_inputs)) {
        auto buf = inputs.get<ffi::AnyBuffer>(kernel.indsi[0]);
        if (buf.has_value()) {
            auto dims = buf->dimensions();
            nx = (dims.size() == 3) ? dims[1] : dims[0];
        }
    }
    if (kernel.nvj_count > 0 && kernel.indsj[0] < static_cast<int>(num_inputs)) {
        auto buf = inputs.get<ffi::AnyBuffer>(kernel.indsj[0]);
        if (buf.has_value()) {
            auto dims = buf->dimensions();
            ny = (dims.size() == 3) ? dims[1] : dims[0];
        }
    }

    int nx_kernel = (kernel.tagI == 1) ? ny : nx;
    int ny_kernel = (kernel.tagI == 1) ? nx : ny;

    int blocks_per_batch = (nx_kernel + kernel.cuda_block_size - 1) / kernel.cuda_block_size;
    size_t nblocks = blocks_per_batch * batch_size;
    size_t total_offsets = kernel.nvi_count + kernel.nvj_count + kernel.nvp_count;
    if (total_offsets == 0) total_offsets = 2;

    // USE int64_t (64-bit) to match kernel signature
    size_t size_offsets = sizeof(int64_t) * nblocks * total_offsets;
    size_t size_lookup  = sizeof(int64_t) * 3 * nblocks;
    size_t size_slices  = sizeof(int64_t) * batch_size;
    size_t size_ranges  = sizeof(int64_t) * 2 * batch_size;
    size_t size_args    = sizeof(void*) * (kernel.max_var_idx + 1);

    size_t needed_bytes = size_offsets + size_lookup + size_slices + size_ranges + size_args;

    auto scratch_result = scratch.Allocate(needed_bytes);
    if (!scratch_result.has_value()) return ffi::Error::Internal("Scratch allocation failed");
    void* scratch_ptr = scratch_result.value();

    // CRITICAL FIX: Pass actual batch_size, not just a flag
    // The previous upload hardcoded this to '2'
    int64_t ranges_enc_value = (batch_size > 1) ? batch_size : 0;
    void* ranges_enc_ptr = (void*)ranges_enc_value;

    void* argshapes_ptr = (void*)kernel.var_counts_packed;

    int result = kernel.launch_fn(
        kernel.tagHostDevice, kernel.dimy, nx_kernel, ny_kernel, kernel.tagI, kernel.tagZero,
        kernel.use_half, kernel.tag1D2D, kernel.dimred, kernel.cuda_block_size, kernel.use_chunk_mode,
        kernel.indsi.data(), kernel.indsj.data(), kernel.indsp.data(),
        kernel.dimout, kernel.dimsx.data(), kernel.dimsy.data(), kernel.dimsp.data(),
        ranges_enc_ptr, nullptr, output_ptr, input_ptrs, argshapes_ptr,
        (void*)stream, scratch_ptr
    );

    if (result != 0) return ffi::Error::Internal("Kernel launch failed: " + std::to_string(result));
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("CUDA error occurred");

    return ffi::Error::Success();
}

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
    m.def("register_keops_kernel", &register_keops_kernel,
          "Register a KeOps kernel for the current CUDA device");
    m.def("is_kernel_registered", &is_kernel_registered,
          "Check if kernel is registered for the current device");
    m.def("is_kernel_registered_on_device", &is_kernel_registered_on_device,
          "Check if kernel is registered for a specific device");
    m.def("cleanup_all_kernels", &cleanup_all_kernels,
          "Clear all registered kernels (all devices)");
    m.def("get_registry_size", &get_registry_size,
          "Get total number of registered kernel-device pairs");
    m.def("get_unique_kernel_count", &get_unique_kernel_count,
          "Get number of unique kernels (across all devices)");
    m.def("get_registry_version", []() { return g_registry_version.load(); },
          "Get current registry version (for cache invalidation)");
    m.def("get_ffi_handler", []() { return EncapsulateFfiCall(KeOpsKernel); },
          "Get the FFI handler capsule");
}