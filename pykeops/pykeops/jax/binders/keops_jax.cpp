/*
 * KeOps JAX C++ Extension (using nanobind) - FULLY OPTIMIZED
 * 
 * FIXES APPLIED:
 * 1. Thread-safe registry with std::shared_mutex (reader/writer locks)
 * 2. Safe pointer lifetime via std::shared_ptr in registry
 * 3. TLS cache stores shared_ptr to prevent use-after-free
 * 4. Added RTLD_DEEPBIND for multi-GPU symbol isolation
 * 5. Proper CUDA error checking
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
#include <shared_mutex>
#include <memory>

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
// Kernel Registry with Thread-Safe Access
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
            // SAFETY: Do not dlclose. Unloading code while GPU operations
            // might still reference it (even via driver internal state) is risky.
            // The GPU may still be executing kernels asynchronously, or the driver
            // may access symbol metadata after the CPU has released the shared_ptr.
            // Memory leak is negligible since kernel count stabilizes in practice.
            // dlclose(kernel_lib);
            kernel_lib = nullptr;
        }
    }

    // Move constructor/assignment (no copy)
    KeOpsKernelInfo(const KeOpsKernelInfo&) = delete;
    KeOpsKernelInfo& operator=(const KeOpsKernelInfo&) = delete;
    KeOpsKernelInfo(KeOpsKernelInfo&& other) noexcept { *this = std::move(other); }
    KeOpsKernelInfo& operator=(KeOpsKernelInfo&& other) noexcept {
        if (this != &other) {
            // Note: Don't dlclose here either - same safety concern as destructor
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

// Use shared_ptr for safe lifetime management across threads
static std::unordered_map<uint64_t, std::shared_ptr<KeOpsKernelInfo>> g_kernel_registry;
static std::shared_mutex g_registry_mutex;  // Reader/writer lock
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
// Stores shared_ptr to ensure kernel stays alive while in use
struct ThreadLocalKernelCache {
    uint64_t kernel_id = 0;
    int device_id = -1;
    uint64_t registry_version = 0;
    std::shared_ptr<KeOpsKernelInfo> kernel_ptr;  // shared_ptr for safe lifetime

    void invalidate() {
        kernel_id = 0;
        device_id = -1;
        registry_version = 0;
        kernel_ptr.reset();
    }

    bool is_valid(uint64_t kid, int dev_id, uint64_t version) const {
        return kernel_id == kid && device_id == dev_id &&
               registry_version == version && kernel_ptr != nullptr;
    }
};

void cleanup_all_kernels() {
    std::unique_lock lock(g_registry_mutex);  // Exclusive lock for modification
    g_kernel_registry.clear();
    g_registry_version.fetch_add(1, std::memory_order_release);  // Invalidate all TLS caches
}

// Check if kernel is registered for the CURRENT device
bool is_kernel_registered(uint64_t kernel_id) {
    int device_id;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess) {
        return false;  // CUDA not available or context issue
    }

    std::shared_lock lock(g_registry_mutex);  // Reader lock
    uint64_t key = make_registry_key(kernel_id, device_id);
    return g_kernel_registry.find(key) != g_kernel_registry.end();
}

// Check if kernel is registered on ALL available devices
bool is_kernel_registered_all_devices(uint64_t kernel_id) {
    int num_devices;
    cudaError_t err = cudaGetDeviceCount(&num_devices);
    if (err != cudaSuccess || num_devices == 0) {
        return false;
    }

    std::shared_lock lock(g_registry_mutex);
    for (int dev = 0; dev < num_devices; dev++) {
        uint64_t key = make_registry_key(kernel_id, dev);
        if (g_kernel_registry.find(key) == g_kernel_registry.end()) {
            return false;
        }
    }
    return true;
}

// Check if kernel is registered for a SPECIFIC device
bool is_kernel_registered_on_device(uint64_t kernel_id, int device_id) {
    std::shared_lock lock(g_registry_mutex);  // Reader lock
    uint64_t key = make_registry_key(kernel_id, device_id);
    return g_kernel_registry.find(key) != g_kernel_registry.end();
}

size_t get_registry_size() {
    std::shared_lock lock(g_registry_mutex);
    return g_kernel_registry.size();
}

// Get number of unique kernels (across all devices)
size_t get_unique_kernel_count() {
    std::shared_lock lock(g_registry_mutex);
    std::unordered_set<uint64_t> unique_kernels;
    for (const auto& pair : g_kernel_registry) {
        auto [kernel_id, device_id] = split_registry_key(pair.first);
        unique_kernels.insert(kernel_id);
    }
    return unique_kernels.size();
}

// Get dimout for a registered kernel (useful for avoiding Python backend creation)
int get_kernel_dimout(uint64_t kernel_id) {
    int device_id;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess) return -1;

    std::shared_lock lock(g_registry_mutex);
    uint64_t key = make_registry_key(kernel_id, device_id);
    auto it = g_kernel_registry.find(key);
    if (it != g_kernel_registry.end() && it->second) {
        return it->second->dimout;
    }
    return -1;
}

void register_keops_kernel(uint64_t kernel_id, nb::object myconv) {
    // Get number of available devices
    int num_devices;
    cudaError_t err = cudaGetDeviceCount(&num_devices);
    if (err != cudaSuccess || num_devices == 0) {
        throw std::runtime_error("No CUDA devices available");
    }

    // Save current device to restore later
    int original_device;
    cudaGetDevice(&original_device);

    // Check if already registered on ALL devices
    bool all_registered = true;
    for (int dev = 0; dev < num_devices; dev++) {
        uint64_t key = make_registry_key(kernel_id, dev);
        std::shared_lock read_lock(g_registry_mutex);
        if (g_kernel_registry.find(key) == g_kernel_registry.end()) {
            all_registered = false;
            break;
        }
    }

    if (all_registered) {
        return;  // Already registered on all devices
    }

    // Need to register - prepare info (shared across all devices)
    auto info = std::make_shared<KeOpsKernelInfo>();

    {
        nb::gil_scoped_acquire gil;

        try {
            std::string kernel_path = nb::cast<std::string>(myconv.attr("kernel_so_path"));

            // Use RTLD_DEEPBIND for multi-GPU symbol isolation
            // Use RTLD_NODELETE to prevent unloading (GPU may reference code asynchronously)
            info->kernel_lib = dlopen(kernel_path.c_str(), RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND | RTLD_NODELETE);
            if (!info->kernel_lib) {
                // Fallback without RTLD_DEEPBIND/RTLD_NODELETE (some systems don't support them)
                info->kernel_lib = dlopen(kernel_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
            }
            if (!info->kernel_lib) {
                throw std::runtime_error("Failed to load .so: " + std::string(dlerror()));
            }

            info->launch_fn = (LaunchKeOpsKernel)dlsym(info->kernel_lib, "launch_keops_kernel");
            if (!info->launch_fn) {
                throw std::runtime_error("Failed to find launch_keops_kernel: " + std::string(dlerror()));
            }

            nb::object params = myconv.attr("params");
            info->tagHostDevice = nb::cast<int>(params.attr("tagHostDevice"));
            info->dimy = nb::cast<int>(params.attr("dimy"));
            info->tagI = nb::cast<int>(params.attr("tagI"));
            info->tagZero = nb::cast<int>(params.attr("tagZero"));
            info->use_half = nb::cast<int>(params.attr("use_half"));
            info->tag1D2D = nb::cast<int>(params.attr("tag1D2D"));
            info->dimred = nb::cast<int>(params.attr("dimred"));
            info->cuda_block_size = nb::cast<int>(params.attr("cuda_block_size"));
            info->use_chunk_mode = nb::cast<int>(params.attr("use_chunk_mode"));
            info->dimout = nb::cast<int>(params.attr("dim"));

            info->indsi = nb::cast<std::vector<int>>(params.attr("indsi"));
            info->indsj = nb::cast<std::vector<int>>(params.attr("indsj"));
            info->indsp = nb::cast<std::vector<int>>(params.attr("indsp"));
            info->dimsx = nb::cast<std::vector<int>>(params.attr("dimsx"));
            info->dimsy = nb::cast<std::vector<int>>(params.attr("dimsy"));
            info->dimsp = nb::cast<std::vector<int>>(params.attr("dimsp"));

            info->finalize();
        } catch (const nb::python_error& e) {
            throw std::runtime_error("Python error during registration: " + std::string(e.what()));
        }
    }

    // Register on ALL devices (shared_ptr allows sharing the same info)
    {
        std::unique_lock write_lock(g_registry_mutex);

        // Check cache size and evict if needed
        if (g_kernel_registry.size() >= MAX_KERNEL_CACHE_SIZE) {
            g_kernel_registry.clear();
            g_registry_version.fetch_add(1, std::memory_order_release);
            if (KEOPS_DEBUG) {
                std::cout << "[KeOps] Registry cache full, cleared all kernels" << std::endl;
            }
        }

        // Register for each device
        for (int dev = 0; dev < num_devices; dev++) {
            uint64_t key = make_registry_key(kernel_id, dev);
            if (g_kernel_registry.find(key) == g_kernel_registry.end()) {
                g_kernel_registry[key] = info;  // Shared ptr - same info for all devices
                if (KEOPS_DEBUG) {
                    std::cout << "[KeOps] Registered kernel " << kernel_id << " for device " << dev << std::endl;
                }
            }
        }

        g_registry_version.fetch_add(1, std::memory_order_release);
    }

    // Restore original device
    cudaSetDevice(original_device);
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
    // Uses shared_ptr to keep kernel alive even if registry is modified
    static thread_local ThreadLocalKernelCache tls_kernel_cache;

    // Get current device - REQUIRED for per-device registry lookup
    int device_id;
    cudaError_t cuda_err = cudaGetDevice(&device_id);
    if (cuda_err != cudaSuccess) {
        return ffi::Error::Internal("Failed to get CUDA device");
    }

    uint64_t kid = static_cast<uint64_t>(kernel_id);
    uint64_t current_version = g_registry_version.load(std::memory_order_acquire);
    std::shared_ptr<KeOpsKernelInfo> kernel_ptr;

    // Fast path: check thread-local cache first
    // Only valid if kernel_id AND device_id match AND registry version hasn't changed
    if (tls_kernel_cache.is_valid(kid, device_id, current_version)) {
        kernel_ptr = tls_kernel_cache.kernel_ptr;
    } else {
        // Slow path: hash map lookup with reader lock
        std::shared_lock lock(g_registry_mutex);

        uint64_t key = make_registry_key(kid, device_id);
        auto it = g_kernel_registry.find(key);
        if (it == g_kernel_registry.end()) {
            return ffi::Error::InvalidArgument(
                "Kernel " + std::to_string(kid) + " not found for device " + std::to_string(device_id)
            );
        }
        kernel_ptr = it->second;  // Copy shared_ptr (increases ref count)

        // Update thread-local cache (still under lock to ensure consistency)
        tls_kernel_cache.kernel_id = kid;
        tls_kernel_cache.device_id = device_id;
        tls_kernel_cache.registry_version = current_version;
        tls_kernel_cache.kernel_ptr = kernel_ptr;
    }

    // At this point, kernel_ptr is a shared_ptr that keeps the kernel alive
    // even if the registry is cleared by another thread
    KeOpsKernelInfo& kernel = *kernel_ptr;
    size_t num_inputs = inputs.size();

    // Fast path: use stack-allocated array for common case (most kernels have < 16 inputs)
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

    // Additional validation
    if (scratch_ptr == nullptr) {
        return ffi::Error::Internal("Scratch allocation returned null pointer");
    }

    // Verify all input pointers are non-null
    for (size_t i = 0; i < num_inputs; ++i) {
        if (input_ptrs[i] == nullptr) {
            return ffi::Error::Internal("Input pointer " + std::to_string(i) + " is null");
        }
    }
    if (output_ptr == nullptr) {
        return ffi::Error::Internal("Output pointer is null");
    }

    // For 3D tensors, always enable ranges by passing batch_size
    // batch_size >= 1 for 3D tensors (is_batched=True in Python)
    // batch_size = 1 for 2D tensors (is_batched=False in Python)
    // The kernel was compiled with use_ranges=(len(shape)==3), so we must
    // pass batch_size consistently to match the compiled kernel's expectations
    int64_t ranges_enc_value = batch_size;
    void* ranges_enc_ptr = (void*)ranges_enc_value;

    void* argshapes_ptr = (void*)kernel.var_counts_packed;

    // DEBUG: Print launch parameters
    if (KEOPS_DEBUG) {
        std::cout << "[KeOps FFI DEBUG] Launch params:" << std::endl;
        std::cout << "  num_inputs=" << num_inputs << std::endl;
        std::cout << "  nx_kernel=" << nx_kernel << ", ny_kernel=" << ny_kernel << std::endl;
        std::cout << "  batch_size=" << batch_size << std::endl;
        std::cout << "  nvi=" << kernel.nvi_count << ", nvj=" << kernel.nvj_count << ", nvp=" << kernel.nvp_count << std::endl;
        std::cout << "  max_var_idx=" << kernel.max_var_idx << std::endl;
        std::cout << "  indsi=["; for(int i=0; i<kernel.nvi_count; i++) std::cout << kernel.indsi[i] << " "; std::cout << "]" << std::endl;
        std::cout << "  indsj=["; for(int i=0; i<kernel.nvj_count; i++) std::cout << kernel.indsj[i] << " "; std::cout << "]" << std::endl;
        std::cout << "  indsp=["; for(int i=0; i<kernel.nvp_count; i++) std::cout << kernel.indsp[i] << " "; std::cout << "]" << std::endl;
        std::cout << "  scratch_bytes=" << needed_bytes << std::endl;
    }

    int result = kernel.launch_fn(
        kernel.tagHostDevice, kernel.dimy, nx_kernel, ny_kernel, kernel.tagI, kernel.tagZero,
        kernel.use_half, kernel.tag1D2D, kernel.dimred, kernel.cuda_block_size, kernel.use_chunk_mode,
        kernel.indsi.data(), kernel.indsj.data(), kernel.indsp.data(),
        kernel.dimout, kernel.dimsx.data(), kernel.dimsy.data(), kernel.dimsp.data(),
        ranges_enc_ptr, nullptr, output_ptr, input_ptrs, argshapes_ptr,
        (void*)stream, scratch_ptr
    );

    if (result != 0) return ffi::Error::Internal("Kernel launch failed: " + std::to_string(result));

    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        return ffi::Error::Internal("CUDA error: " + std::string(cudaGetErrorString(cuda_err)));
    }

    // Synchronize stream to ensure kernel completes before returning
    // This is critical for correct behavior when multiple FFI calls
    // are executed in sequence by XLA
    cuda_err = cudaStreamSynchronize(stream);
    if (cuda_err != cudaSuccess) {
        return ffi::Error::Internal("CUDA stream sync error: " + std::string(cudaGetErrorString(cuda_err)));
    }

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
          "Register a KeOps kernel for ALL available CUDA devices");
    m.def("is_kernel_registered", &is_kernel_registered,
          "Check if kernel is registered for the current device");
    m.def("is_kernel_registered_all_devices", &is_kernel_registered_all_devices,
          "Check if kernel is registered on all available devices");
    m.def("is_kernel_registered_on_device", &is_kernel_registered_on_device,
          "Check if kernel is registered for a specific device");
    m.def("cleanup_all_kernels", &cleanup_all_kernels,
          "Clear all registered kernels (all devices)");
    m.def("get_registry_size", &get_registry_size,
          "Get total number of registered kernel-device pairs");
    m.def("get_unique_kernel_count", &get_unique_kernel_count,
          "Get number of unique kernels (across all devices)");
    m.def("get_kernel_dimout", &get_kernel_dimout,
          "Get dimout for a registered kernel (-1 if not found)");
    m.def("get_registry_version", []() { return g_registry_version.load(); },
          "Get current registry version (for cache invalidation)");
    m.def("get_ffi_handler", []() { return EncapsulateFfiCall(KeOpsKernel); },
          "Get the FFI handler capsule");
}
