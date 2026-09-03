# KeOps JAX Test Suite

Comprehensive test suite for the JAX backend of KeOps.

## Overview

| Test File | Suite name | Description |
|-----------|-----------|-------------|
| `test_edge_cases.py` | `edge` | Bugs found during development; also the `quick` suite |
| `test_api.py` | `api` | Genred, LazyTensor, Vi/Vj/Pm, dtypes, JIT, gradients |
| `test_correctness.py` | `correctness` | Cross-validation against PyTorch KeOps |
| `test_advanced.py` | `advanced` | LogSumExp, KMin, exotic math, refused transformations |
| `test_batched_gradients.py` | `batched` | 3D gradients, block-boundary cases |
| `test_batch_broadcasting.py` | `broadcast` | Size-one batch axes against a batch |
| `test_helpers.py` | `helpers` | generic_sum, generic_logsumexp and friends |
| `test_benchmark_single_gpu.py` | `benchmark` | Single-GPU timings, not in `all` |
| `test_benchmark_multi_gpu.py` | `benchmark-multi` | Multi-GPU scaling, not in `all` |

Every one of these compares against PyTorch KeOps as ground truth and needs a GPU. Without
PyTorch with CUDA they exit rather than run: the comparison is the test.

## Quick Start

```bash
# Run all tests
python run_tests.py

# Run specific test suite
python run_tests.py edge          # Edge cases only
python run_tests.py api           # API tests only
python run_tests.py correctness   # Cross-validation (needs PyTorch)
python run_tests.py advanced      # Advanced features

# Run benchmarks
python run_tests.py benchmark       # Single GPU
python run_tests.py benchmark-multi # Multi-GPU

# Quick sanity check
python run_tests.py quick
```

## Using pytest

Run from inside this directory; it collects 67 tests.

```bash
pytest -q                          # everything, in one process
pytest test_edge_cases.py          # one file
pytest test_api.py -k gradient     # select by name
pytest -m pytorch                  # every test here is marked gpu and pytorch
pytest -m "not slow"               # nothing is marked slow yet, so this selects everything
```

Two differences from `run_tests.py`:

- Helpers that take arguments and are driven by a file's `main()` are named `check_*`, not
  `test_*`, so pytest does not collect them as tests with missing fixtures. Follow that when
  adding one.
- pytest runs every file in a single process, where `run_tests.py` forks one per file. The
  memory-hungry `test_high_dim_gradient` can therefore run out of GPU memory under pytest on a
  busy card while passing on its own.

`old/` is excluded from collection: it holds superseded tests that no longer import.

## Test Descriptions

### test_edge_cases.py

Tests for bugs that were discovered and fixed during development:

1. **Scalar multiplication variable ordering** - `2.0 * (x_i - y_j)` gave wrong results due to `_var_ids` tracking issue
2. **Batched vs non-batched operations** - Shape handling for 2D vs 3D inputs
3. **Complex varifold kernels** - Multi-variable formulas with 6+ variables
4. **Pm parameter gradients** - Gradient computation for scalar parameters
5. **High-dimensional chunking** - Large dimension handling
6. **JIT compilation** - Ensure JIT works correctly

### test_api.py

Core API functionality:

- **Genred interface**: Various formulas, reductions, axes
- **LazyTensor interface**: Symbolic operations
- **Vi, Vj, Pm helpers**: Convenience functions
- **Data types**: float32, float64
- **Gradients**: Automatic differentiation
- **JIT**: Compilation compatibility

### test_correctness.py

Cross-validation against PyTorch KeOps (ground truth):

- Forward pass comparison for all formula types
- Gradient comparison (Vi, Vj, Pm variables)
- Multiple problem sizes
- Reduction operations (Sum, Min, Max)

### test_advanced.py

Advanced features and operations:

- **LogSumExp reduction** - Critical for stable softmax, tests numerical stability
- **KMin/ArgKMin reductions** - K-nearest neighbor operations
- **Exotic math** - Trig functions (sin, cos), Step, Sign, Abs, Clamp
- **Complex broadcasting** - Mixed batch dimensions (B,N,D) vs (M,D)
- **Higher-order gradients** - Hessian-vector products, double gradients

### test_benchmark_single_gpu.py

Performance measurements:

- Forward pass timing (JAX vs PyTorch)
- Gradient computation timing
- Various problem sizes (1K to 100K points)
- Different formulas (SqDist, Gaussian)

### test_benchmark_multi_gpu.py

Multi-GPU scaling tests:

- Single GPU baseline
- Scaling with 2, 4, 8 GPUs
- Efficiency metrics
- Memory distribution

## Requirements

### Core Requirements
- Python 3.8+
- JAX with CUDA support
- NumPy
- KeOps with JAX backend

### Optional (for comparison tests)
- PyTorch with CUDA
- KeOps with PyTorch backend

### Optional (for enhanced output)
- Rich library (`pip install rich`)

## Output

The tests produce colorful, formatted output with:
- Progress indicators
- ASCII tables for results
- Color-coded pass/fail status
- Summary statistics

If the Rich library is installed, output is enhanced with:
- Beautiful tables
- Progress bars
- Syntax highlighting

## Exit Codes

- `0`: All tests passed
- `1`: Some tests failed
- `2`: Configuration error

## Continuous Integration

Example GitHub Actions workflow:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install jax[cuda] pykeops numpy rich
      - name: Run tests
        run: python run_tests.py quick
```

## Contributing

When adding new tests:

1. Add edge cases to `test_edge_cases.py`
2. Add API tests to `test_api.py`
3. Use `test_utils.py` for consistent formatting
4. Include both JAX-only and PyTorch comparison versions
5. Document what bug/feature the test covers
