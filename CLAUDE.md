# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

KeOps computes reductions of large arrays defined by mathematical formulas, with automatic
differentiation and without storing the full matrix. Entries are computed on the fly inside a
generated CUDA kernel.

Two packages:

- `keopscore` turns a formula string into C++/CUDA source and compiles it.
- `pykeops` binds that to NumPy, PyTorch and JAX.

This checkout is a fork. The work is the JAX backend on the `jax_api` branch, about 50 commits
ahead of `main`. The NumPy and PyTorch paths are upstream code and must keep working unchanged;
`pykeops.torch` is also the correctness reference the JAX tests compare against.

## Do not run Python with the repo root as the working directory

`keops/keopscore/__init__.py` is a 0-byte file sitting next to `keopscore/setup.py`. Whenever the
repo root is on `sys.path`, which is any `python` invocation started from there, it shadows the
installed `keopscore` package and every pykeops import dies at
`keopscore/keopscore/config/base_config.py:82` with:

```
AttributeError: module 'keopscore' has no attribute '__version__'
```

Run from somewhere else. `cd pykeops/pykeops/jax/test` for the test suite, or a scratch directory
for one-off scripts. This also breaks `pytest` and `python -c` launched at the root.

## Environment

Conda env `jax_latest` (Python 3.12). `keopscore` and `pykeops` are installed editable and already
point at this checkout. The JAX backend is CUDA only, with no CPU fallback, so it needs a GPU and a
CUDA toolkit with `nvcc`.

## Build

```bash
pip install -e ./keopscore --no-build-isolation --no-deps
pip install -e ./pykeops   --no-build-isolation --no-deps
```

Install keopscore first, and keep `--no-deps`: `pykeops/setup.py` imports keopscore, and without
the flag pip replaces the editable checkout with a release wheel. setup.py prints a warning when it
detects this.

Installing pykeops also builds the JAX C++ extension. `pykeops/setup.py` runs CMake over
`pykeops/pykeops/jax/binders/` from both its `build` and `egg_info` commands and writes
`keops_jax_ext.cpython-*.so` into `pykeops/pykeops/jax/`. `*.so` is gitignored, so a fresh clone has
no extension until you install. `CMAKE_CUDA_ARCHITECTURES` comes from
`nvidia-smi --query-gpu=compute_cap`, falling back to `70;75;80;86;89;90`. If JAX, nanobind, nvcc or
cmake is missing, setup.py prints why, skips the extension and installs only the Python side, so
watch the install log rather than the exit code.

After editing `keops_jax.cpp` or its `CMakeLists.txt`, rebuild. The configured tree persists, so

```bash
cmake --build pykeops/build/jax_ext_build -j
```

is enough. Reinstalling also works but reconfigures from scratch.

Release packaging is `./pybuild.sh`, or `./pybuild.sh -l` for a local build with no hard-coded
versions.

## Testing

The JAX suite lives in `pykeops/pykeops/jax/test/` and has to run from that directory, because the
test files import `test_utils` as a top-level module.

```bash
cd pykeops/pykeops/jax/test
python run_tests.py              # edge, api, correctness, advanced, batched, helpers
python run_tests.py quick        # edge only
python run_tests.py api          # one suite
python run_tests.py correctness  # cross-check against pykeops.torch
python run_tests.py --float64    # sets KEOPS_TEST_FLOAT64=1 and JAX_ENABLE_X64=1
python test_api.py               # one file directly; run_tests.py just shells out to this
```

Suite names are api, correctness, edge, advanced, batched, helpers, benchmark, benchmark-multi.
Benchmarks are excluded from `all`.

Every JAX test compares against `pykeops.torch` as ground truth and calls `sys.exit(1)` when
PyTorch with CUDA is absent.

pytest works from inside that directory too, and collects 67 tests:

```bash
pytest -q                          # the whole suite in one process
pytest test_api.py -k gradient     # select by name
pytest -m pytorch                  # every test here is marked gpu and pytorch
```

Two things to know. Helpers that take arguments and are driven by each file's `main()` are named
`check_*`, not `test_*`, so pytest does not mistake them for tests with missing fixtures; if you
add one, follow that. And pytest runs every file in one process where `run_tests.py` forks per
file, so the memory-hungry `test_high_dim_gradient` can fail under pytest on a busy card while
passing on its own. `run_tests.py` remains the runner the suites are written for.

`./pytest.sh` is the upstream harness. It builds a throwaway venv, installs both packages, clears
the cache, then runs `keopscore/keopscore/test/` and `pykeops/pykeops/test/`, which are the
PyTorch and NumPy suites. It never touches the JAX backend.
`.github/workflows/cuda_test.yml` runs it on a self-hosted GPU runner.

## Lint

`.github/workflows/black.yml` runs psf/black on every push and pull request. The JAX backend files
predate that job and are not black-formatted (single quotes, `if cond: stmt` on one line), so a
repo-wide `black .` would produce an enormous diff. Format what you touch, not the tree.

## Architecture

### Formula system (keopscore/keopscore/formulas/)

`variables/` holds Vi (indexed by i), Vj (indexed by j) and Pm (parameters). `maths/` holds the
operations, `reductions/` holds Sum, LogSumExp, Min, Max, ArgMin, ArgMax and KMin, and `complex/`
and `factorization/` extend both. `autodiff/` differentiates formulas symbolically, which is why a
gradient is another KeOps kernel rather than a tape replay.

A string such as `Sum_Reduction(Exp(-Sum(Square(Var(0,3,0)-Var(1,3,1)))),0)` is parsed,
differentiated if needed, and turned into CUDA source by `mapreduce/`. `mapreduce/cpu/` emits
OpenMP code, `mapreduce/gpu/` emits CUDA.

### Two compile backends, selected by `lang`

This is the main structural change in the fork, and it is spread across several files.

`pykeops.torch` and `pykeops.numpy` compile with NVRTC and call the kernel through a pybind11
wrapper. `pykeops.jax` compiles with nvcc into a standalone `.so` and calls it through XLA FFI. A
single `lang` string threads that choice through:

1. `pykeops/pykeops/jax/generic/generic_ops.py::_create_keops_backend` calls
   `keops_binder["cpp"](..., "jax", ...)`. Note that the JAX path goes through the `cpp` binder,
   not the `nvrtc` one.
2. `LoadKeOps_cpp` sees `params.lang == "jax"` and skips both phases of the pybind11 wrapper,
   recording `kernel_so_path` instead.
3. `keopscore/get_keops_dll.py` forwards `lang` to the map-reduce class.
4. Every `GpuReduc*` class in `keopscore/mapreduce/gpu/` is a factory. Its `__new__` asks
   `gpu_utils.use_cuda_backend(lang)` and returns either the `*_Cuda` variant, which mixes in
   `binders/cuda/Cuda_link_compile.py` and shells out to nvcc, or the `*_Nvrtc` variant, which
   mixes in `binders/nvrtc/Gpu_link_compile.py`.
5. `LinkCompile.__init__` folds `lang` into the cache hash and appends a suffix in JAX mode, so the
   two backends never load each other's cached artifacts.

The nvcc-built `.so` exports `extern "C" int launch_keops_kernel(...)`, emitted by the launcher
template in `Cuda_link_compile.py`. `keops_jax.cpp` `dlopen`s it with
`RTLD_LAZY|RTLD_LOCAL|RTLD_DEEPBIND|RTLD_NODELETE`, falling back without the last two, and `dlsym`s
that symbol.

`Cuda_link_compile.py` resolves nvcc to an absolute path, trying `shutil.which` and then a list of
common CUDA bin directories, because IDE run configurations often start Python without CUDA on
PATH; the failure otherwise surfaces as "CMake compilation succeeded but .so file not found".

### JAX runtime path

`Genred.__init__` calls `make_keops_jax_op`, which builds a closure and compiles nothing. The first
`__call__` does the work:

- selects an entry in `kernel_cache`, keyed `"2d"` or `"3d"` plus opt_arg and formula2, so a
  batched and an unbatched call to the same `Genred` compile separate kernels
- compiles through `_create_keops_backend`
- calls `register_keops_kernel(kernel_id, myconv)`, which loads the `.so` on every visible CUDA
  device
- calls `jax.ffi.register_ffi_target(name, get_ffi_handler(), platform="CUDA")` under
  `_registration_lock`, treating "already registered" as success
- calls `jax.ffi.ffi_call(...)` with `kernel_id` and `batch_size` passed as FFI attributes

Registration is deferred to the first call on purpose. Doing it at construction time makes JAX's
trace-time validation calls fire the kernel.

On the C++ side `g_kernel_registry` maps `(kernel_id, device_id)` to a
`shared_ptr<KeOpsKernelInfo>` under a `shared_mutex`, fronted by a thread-local single-entry cache
that is validated against an atomic `g_registry_version`. The handler ends with
`cudaStreamSynchronize`.

Gradients use `jax.custom_vjp`. `make_keops_jax_op` precomputes a `Grad(formula, var, eta)` string
per input variable up front; the backward pass compiles those on demand and stores them in
`_grad_cache`, a bounded LRU keyed by formula content rather than `id()`.

Batched (3D) inputs reuse the KeOps ranges mechanism, `use_ranges=True`, not a separate kernel.

### Variable identity in LazyTensor

`pykeops/pykeops/jax/lazytensor/LazyTensor.py` gives each variable an id from a lock-protected
monotonic counter offset by 1000000 instead of `id()`, because CPython reuses the addresses of
collected objects and that produced wrong answers for expressions such as `2.0 * (x_i - y_j)`.
`_var_ids` is a tuple carried next to `variables` and merged on binary operations; `fixvariables()`
maps those ids to positional indices before the formula is compiled.

## Key interfaces

All three take formulas over `Vi`, `Vj` and `Pm` variables. Verified against NumPy on 2026-09-03.

LazyTensor, the symbolic front end:

```python
from pykeops.jax import LazyTensor, Vi, Vj, Pm

x_i = Vi(x)                     # (N, D) array seen as (N, 1, D)
y_j = Vj(y)                     # (M, D) array seen as (1, M, D)
D_ij = ((x_i - y_j) ** 2).sum(dim=2)
K_ij = (-D_ij).exp()
result = K_ij.sum(dim=1)        # (N, 1)
```

`LazyTensor(x[:, None, :])` and `LazyTensor(y[None, :, :])` with `.sum(-1)` is the equivalent
explicit form.

Genred, the formula front end. The `formula` argument is the summand only; `reduction_op` and
`axis` supply the reduction. Writing the reduction into the formula string
(`"Sum_j(Exp(-SqDist(x,y)))"`) raises `NameError: name 'Sum_j' is not defined`.

```python
from pykeops.jax import Genred

genred = Genred("Exp(-SqDist(x,y))", ["x=Vi(3)", "y=Vj(3)"], reduction_op="Sum", axis=1)
result = genred(x, y)           # (N, 1)
```

KernelSolve, conjugate gradient against the kernel matrix:

```python
from pykeops.jax import KernelSolve

solver = KernelSolve("Exp(-SqDist(x,y)) * a", ["x=Vi(3)", "y=Vj(3)", "a=Vj(3)"], "a", axis=1)
a_star = solver(x, x, b, alpha=0.1)      # solves (alpha I + K) a = b
```

Shorthands `generic_sum`, `generic_logsumexp`, `generic_argmin`, `generic_argkmin`, `generic_min`
and `generic_max` wrap Genred and are exported from `pykeops.jax`.

## Environment variables

- `PYKEOPS_VERBOSE` / `KEOPS_VERBOSE`, set to `"0"` to silence. Set `PYKEOPS_VERBOSE` before
  importing pykeops; `pykeops/__init__.py` propagates it to `KEOPS_VERBOSE`.
- `JAX_KEOPS_DEBUG=1`, debug output from the C++ handler and from the chunking decision in
  `get_keops_dll.py`.
- `KEOPS_CACHE_FOLDER`, compiled-artifact cache. Defaults to `~/.cache/keops2.3`.
- `KEOPS_TEST_FLOAT64=1` together with `JAX_ENABLE_X64=1` for float64 test mode.
  `run_tests.py --float64` sets both.
- `CUDA_PATH`, `CUDA_ARCH`, `CXX`, `CXXFLAGS` feed keopscore's config detection.

`pykeops.jax` sets `PYKEOPS_JAX_MODE=1` on first use, but nothing reads it.

## Known bugs, in the order they matter

### 1. Batch axis of size one (fixed 2026-09-03 on `fix-jax-batch-broadcasting`)

A side of shape `(1, M, D)` used against `(B, N, D)` is now reused for every one
of the B samples, the way NumPy and `pykeops.torch` do it.

What was wrong. The binder took the batch size from `args[0]` alone, and the
generated ranges launcher in `Cuda_link_compile.py` gives every "i" variable a
batch stride of `nx` and every "j" variable a stride of `ny`. Upstream KeOps
instead computes a per-argument offset in `keopscore/include/ranges_utils.h`,
where `broadcast_index()` contributes zero stride for a batch dim of size one.
So `(1, M)` against `(B, N)` returned one sample in the shape `(1, M)`, and
`(B, M)` against `(1, N)` returned the expected shape with out-of-bounds reads on
every row past the first.

The fix, in `pykeops/pykeops/jax/generic/generic_ops.py`: `_batch_info` derives
the batch size from every "i" and "j" argument rather than the first one, and
`_broadcast_batch_dims` expands size-one batch axes with `jnp.broadcast_to`
before the FFI call, which makes the launcher's uniform-stride assumption true.
Both run outside the `custom_vjp`, so JAX differentiates the expansion itself and
sums each cotangent back to the caller's own shape. `test_batch_broadcasting.py`
(suite name `broadcast`) covers it against `pykeops.torch`.

The cost is memory: the expanded argument is materialized, where the torch path
uses a zero stride and copies nothing. For the usual shapes (one template against
a batch) that is small. Teaching the launcher per-argument strides would remove
it, and would also fix the batched `Pm` below, but it changes the launcher
template without changing the kernel cache hash, so stale kernels in
`~/.cache/keops2.3` would silently keep the old behaviour. Bump the hash if you
go that way.

Three neighbouring shapes now raise instead of computing something wrong:

| shapes | before | now |
|---|---|---|
| `(1, M)` vs `(B, N)` | one sample, shape `(1, M)` | correct, shape `(B, M)` |
| `(B, M)` vs `(1, N)` | right shape, rows past 0 read out of bounds | correct |
| `(2, M)` vs `(3, N)` | silently returned `(2, M)` | `ValueError` |
| 3D against 2D | silently wrong | `ValueError` |
| more than one batch axis | treated as unbatched | `NotImplementedError` |

Note on the older note in this file: only the LazyTensor front end raised
`Incompatible batch dimensions` for the `(2,)` vs `(3,)` case, on either backend.
`check_broadcasting` lives in `pykeops/common/lazy_tensor.py:424` and Genred never
called it, and `do_checks` is 0 in `keopscore/include/Sizes.h`, so the torch
Genred path silently accepted that pair too.

### 1b. A parameter that varies along the batch (still open)

`Pm` of shape `(B, 1, d)` with B > 1 is read at offset 0 for every sample, so only
the first sample's value is used. The launcher sets every parameter offset to
zero (`Cuda_link_compile.py`, the `h_offsets` loop), and no Python-side expansion
can fix that, unlike the case above. `pykeops.torch` handles it. It now raises
`NotImplementedError` rather than returning a wrong answer; the real fix is
per-argument strides in the launcher.

### 2. jax.vmap is refused (decided 2026-09-03)

`jax.vmap` over a KeOps reduction used to return an array of the right shape holding the wrong
numbers: max error 4.9 against numpy on a 3x5x7 Gaussian, and GSED's `gsed/varifold.py` records
"~7000x wrong and negative" on a quantity that cannot be negative. The FFI call has no batching
rule, so JAX mapped over a handler that knows nothing about the extra axis.

It is refused in two places, and it needs both:

- `_reject_vmap` (`generic_ops.py:259`) raises `NotImplementedError` naming KeOps batch dimensions
  as the route. It spots the vmap tracer by class name, since `BatchTracer` moved to `jax._src` in
  JAX 0.11 and is no longer re-exported from `jax.interpreters.batching`.
- Both `ffi_call` sites pass `vmap_method=None` (`generic_ops.py:483` and `:730`), which makes JAX
  itself refuse to batch the primitive.

The guard alone is not enough. It only sees the arguments handed to the Python wrapper, so under
`vmap(jit(op))` those are jit tracers, the batching happens outside on the compiled jaxpr, and with
the previous `vmap_method="broadcast_all"` that route returned wrong numbers with max error 5.05.
`vmap_method=None` closes it. Measured across `vmap(Genred)`, `vmap(jit(...))`, `jit(vmap(...))`,
LazyTensor, KernelSolve and `jacrev`: all refuse.

Refusing rather than fixing is a decision. `vmap_method="sequential"` makes vmap correct in one
word, measured, but calls the kernel once per sample: 3.24 ms against 0.17 ms for the same work
through batch dimensions at B=32, N=M=4000. That trades a wrong answer for a quiet 19x slowdown,
and batch dimensions are both correct and faster. `jax.jacrev` maps internally and now raises where
it was silently wrong; `jax.jacfwd` already failed on `custom_vjp` for a separate reason. Covered
by `test_advanced.py`, section 4.

A scan of every project importing `pykeops.jax` on 2026-09-03 found exactly one caller that
vmaps a KeOps kernel, `toy_neural_process/erwin_varifold_weight_learner.py:472`, whose RKHS
errors have therefore always been wrong. It is deliberately left alone: it is a different
project, and failing loudly there is the point of this change. Everything else (GSED,
gmtools, flow_to_glow, pptf) already uses batch dimensions and carries its own notes saying
so.

### 3. Async dispatch (measured 2026-09-03, shelved)

The idea was to make the handler's closing `cudaStreamSynchronize` optional behind
`JAX_KEOPS_ASYNC=1`, replacing its one real job (fencing the ranges launcher's thread-local pinned
staging buffer) with a per-device CUDA event. The gate on it was: measure how much host-side Python
gsed could hide, and drop the idea if the answer is around 10%. That measurement has now been run
and the answer is **0.95%**, so it is not being done.

Over 197 steps of `stage1_varifold.yaml` at one card's share of its 4-GPU batch: blocking step call
518.01 ms, hideable Python 4.99 ms median (p95 5.69). gsed's step is 518 ms where the pairwise
template fit's is 62.5 ms, and the Python per step is roughly constant at 5-11 ms either way, so the
17% measured there does not carry over.

The split matters more than the number. gsed uses the ranges path, so the pinned staging buffer and
its fence are live there, and it runs on 4 GPUs, which is the one rung that cannot be tested on this
machine: that is where the risk of silently wrong numbers sits, and it buys 1%. The pairwise fit has
the 17%, but every one of its KeOps calls has `batch_size = 1` and takes the plain launcher, which
has no pinned buffer and needs no fence at all. If it is ever revisited for that consumer, the flag
and the sync skip alone would do it, with no fence and no ranges path involved.

The measurement was made by wrapping `eqx.filter_jit` so every jitted call is timed, changing
nothing in gsed's source. `stage1_varifold.yaml`'s full 80-mesh batch wants a single 17.87 GiB
allocation and OOMs on one card, so it ran at `groups: 2`, one card's share of the 4-GPU recipe.
That biases the result in the safe direction: at the real per-card load the step is longer and the
share smaller.

### 4. Both frameworks get imported whenever KeOps is used

Measured in fresh processes on 2026-09-03: after `import pykeops.jax`, `torch` and `pykeops.torch`
are in `sys.modules`; after `import pykeops.torch` alone `jax` is not, but the first torch
reduction imports it through the guarded `from jax import Array` in
`pykeops/pykeops/common/get_options.py::_find_mem` (line 98). The site of the torch import on the
JAX side is not yet traced (`pykeops/config.py:10` only probes with `find_spec`, which does not
import; a finder hook broke KeOps's own import, so use
`python -X importtime -c "import pykeops.jax"` instead). The cost is import time and resident
memory, not correctness. It may be unavoidable, since KeOps was built around torch and numpy and
the JAX binder reuses that plumbing; the things to look at are whether the JAX side can avoid the
torch import, and whether `_find_mem` can classify by `type(var).__module__` instead of importing
both frameworks.

One caution before removing it: on this machine the torch import is what makes JAX GPU linear
algebra correct in a JAX-only process, because torch loads its pip CUDA libraries by explicit path
ahead of the system CUDA that `~/.bashrc` puts on `LD_LIBRARY_PATH`. See
`geometric_measure_theory_functions/docs/precision.md`. Removing the import without fixing that
machine's `.bashrc` re-exposes every JAX-only script there.
