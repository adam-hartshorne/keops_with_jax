cmake_minimum_required(VERSION 3.18)
project(keops_jax_ext LANGUAGES CXX CUDA)

# Find Python and nanobind
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Get XLA FFI include directory from JAX
# Try new API first (jax.extend.ffi), then fall back to old API (jax.ffi)
execute_process(
    COMMAND "${Python_EXECUTABLE}" "-c"
    "try:\n    from jax.extend import ffi\nexcept ImportError:\n    import jax.ffi as ffi\nprint(ffi.include_dir())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE XLA_INCLUDE_DIR
    RESULT_VARIABLE XLA_INCLUDE_RESULT
)

if(NOT XLA_INCLUDE_RESULT EQUAL 0)
    # Fallback: try jaxlib include directory directly
    execute_process(
        COMMAND "${Python_EXECUTABLE}" "-c" "import jaxlib; import os; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))"
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE XLA_INCLUDE_DIR
        RESULT_VARIABLE XLA_INCLUDE_RESULT
    )

    if(NOT XLA_INCLUDE_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to get XLA include directory from JAX/jaxlib")
    endif()
endif()

message(STATUS "XLA include directory: ${XLA_INCLUDE_DIR}")

# Create the extension module
nanobind_add_module(
    keops_jax_ext
    keops_jax.cpp
)

# Include XLA FFI headers
target_include_directories(keops_jax_ext PRIVATE ${XLA_INCLUDE_DIR})

# Link CUDA runtime
target_link_libraries(keops_jax_ext PRIVATE CUDA::cudart)

# Set C++ standard
target_compile_features(keops_jax_ext PRIVATE cxx_std_17)

# Install the module
install(TARGETS keops_jax_ext LIBRARY DESTINATION .)
