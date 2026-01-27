# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KeOps (Kernel Operations) is a library for computing reductions of large arrays defined by mathematical formulas, with automatic differentiation and without memory overflows. It performs symbolic matrix-vector products where entries are computed on-the-fly rather than stored in memory.

**Two-package structure:**
- **keopscore** - Meta-programming engine that generates C++/CUDA code from formulas
- **pykeops** - Python bindings for NumPy, PyTorch, and JAX backends

## Build Commands

```bash
# Development install (editable)
pip install -e ./keopscore --no-build-isolation --no-deps
pip install -e ./pykeops --no-build-isolation --no-deps

# With JAX support
pip install ./pykeops[jax]

# Release packaging
./pybuild.sh           # Build distributions
./pybuild.sh -l        # Local build (no hard-coded versions)
```

## Testing

```bash
# Run JAX tests (from pykeops/pykeops/jax/test/)
python run_tests.py              # All tests
python run_tests.py quick        # Quick sanity check
python run_tests.py api          # API tests only
python run_tests.py correctness  # Cross-validation vs PyTorch

# Using pytest
pytest pykeops/pykeops/jax/test/test_api.py -v
pytest -m "not slow"             # Skip slow tests
pytest -m pytorch                # Only PyTorch comparison tests

# Full test suite
./pytest.sh
```

## Architecture

### Formula System (keopscore/formulas/)

Symbolic algebra system with:
- **variables/** - `Vi` (indexed by i), `Vj` (indexed by j), `Pm` (parameters)
- **maths/** - Mathematical operations (sin, cos, exp, SqDist, etc.)
- **reductions/** - Sum, LogSumExp, Min, Max, ArgMin, ArgMax, KMin
- **autodiff/** - Automatic differentiation engine
- **complex/** - Complex number support

### MapReduce Engine (keopscore/mapreduce/)

- **cpu/** - OpenMP parallel execution
- **gpu/** - CUDA kernel execution via NVRTC JIT compilation

### Python Backends (pykeops/)

Three backends sharing common interfaces:
- **torch/** - Production-ready PyTorch bindings
- **jax/** - JAX bindings using XLA FFI + nanobind (active development on `jax_api` branch)
- **numpy/** - NumPy bindings

### Key Interfaces

**LazyTensor** (symbolic operations):
```python
from pykeops.jax import LazyTensor, Vi, Vj, Pm

x_i = Vi(x)              # Shape: (N, 1, D) - rows indexed by i
y_j = Vj(y)              # Shape: (1, M, D) - cols indexed by j
D_ij = ((x_i - y_j)**2).sum(dim=2)
K_ij = (-D_ij).exp()
result = K_ij.sum(dim=1) # Reduction over j
```

**Genred** (formula-based):
```python
from pykeops.jax import Genred

formula = "Sum_j(Exp(-SqDist(x,y)))"
aliases = ["x=Vi(3)", "y=Vj(3)"]
genred = Genred(formula, aliases, reduction_op='Sum', axis=1)
result = genred(x, y)
```

**KernelSolve** (conjugate gradient):
```python
from pykeops.jax import KernelSolve

solver = KernelSolve(formula, aliases, varinvalias="a", axis=1)
a_star = solver(x, x, b, alpha=0.1)  # Solves (αI + K)a = b
```

## JAX Backend Details

The JAX backend (pykeops/jax/) uses:
- **CMake + nanobind** for C++ extension building
- **XLA FFI** for GPU kernel dispatch (not pybind11)
- Thread-safe kernel registry with shared mutex
- RTLD_DEEPBIND for multi-GPU symbol isolation

Critical implementation details:
- Uses monotonic counter (not Python `id()`) for variable tracking to prevent reuse bugs
- LazyTensor `_var_ids` tuple tracks unique variable IDs
- `fixvariables()` converts id-based to index-based formulas before compilation
- All JAX tests validate against PyTorch as ground truth

## Environment Variables

- `PYKEOPS_VERBOSE` / `KEOPS_VERBOSE` - Control verbosity (set to "0" to silence)
- `JAX_KEOPS_DEBUG` - Enable debug output in JAX extension
- `KEOPS_CACHE_FOLDER` - Custom cache location for compiled binaries
