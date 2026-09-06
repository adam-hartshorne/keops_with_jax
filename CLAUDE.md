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
`pykeops.torch` is also the correctness reference the JAX tests compare against. `rkeops/` (the R
binder), `doc/` (Sphinx) and `benchmarks/` are upstream too and nothing here touches them.

## Do not run Python with the repo root as the working directory

The repo root holds a directory called `keopscore/` and one called `pykeops/`, and both shadow the
installed packages whenever the root is on `sys.path`, which is any `python` started from there.
Neither has an `__init__.py`, so each resolves to an empty namespace package: `pykeops.__file__` is
None and `__path__` is `['<root>/pykeops']`. That makes the failure quiet. `import pykeops` at the
root succeeds, and nothing raises until something asks for an attribute, a long way from the cause.

Run from somewhere else. `cd pykeops/pykeops/jax/test` for the test suite, or a scratch directory
for one-off scripts. `python -c` and `python -m pytest` fail the same way at the root, because `-m`
puts the working directory on `sys.path`. The bare `pytest` console script does not, so since the
shim below was deleted it collects fine from the root; `python -m pytest` still does not.

There used to be a second, worse version of this. A tracked 0-byte `keopscore/__init__.py` sat next
to `keopscore/setup.py`, added on this branch by `9f596f38` and never present on `main`. Because
pytest walks up from a test file through every directory holding an `__init__.py`, it walked
through that one and put the repo root on `sys.path`, so `pytest keopscore/keopscore/test/` errored
during collection from *any* working directory, `--import-mode=importlib` included. That is the
keopscore half of `./pytest.sh`, which runs under `set +e` and exits with the status of the pykeops
run, so `cuda_test.yml` reported green with 67 tests never running, from 2026-01-23 until the file
was deleted on 2026-09-06. Do not reintroduce it; nothing imports it and `keopscore/setup.py` reads
its data by path.

## Ten tracked paths are symlinks; keep them that way

Upstream keeps one copy of the version string, the licence and the readme at the repo root and
symlinks them into each package, so git tracks ten paths at mode `120000` whose content is just a
relative target:

```
keopscore/keopscore/keops_version -> ../../keops_version    pykeops/pykeops/keops_version -> ../../keops_version
keopscore/keopscore/licence.txt   -> ../../licence.txt      pykeops/pykeops/licence.txt   -> ../../licence.txt
keopscore/keopscore/readme.md     -> ../readme.md           pykeops/pykeops/readme.md     -> ../readme.md
keopscore/licence.txt             -> ../licence.txt         pykeops/licence.txt           -> ../licence.txt
rkeops/LICENSE.md                 -> ../licence.txt         rkeops/version                -> ../keops_version
```

This checkout had arrived with all ten flattened into regular files holding the dereferenced
content, so `git status` showed ten permanent `T` (typechange) entries. That is a trap rather than
cosmetic: `git add -A` or `git commit -a` would have committed *symlinks converted to regular
files* into KeOps history, an upstream-breaking change that looks like background noise in the
status output. The filesystem is not the cause on either box: `dior` is ext4, the 5090 box is NTFS
through ntfs-3g, and both create symlinks fine (`ln -s` in the work tree succeeds on both, and
`core.symlinks` is at its default). It is the copy that created the tree that flattened them, so it
can happen again on any machine if that copy is repeated.

They were restored on 2026-09-06 with `git checkout --` on the ten paths, after confirming each
flattened file was byte-identical to its target, and `keopscore.__version__` still reads `2.3`
through the links. If those ten `T` entries ever reappear, the tree was copied again without `-a`;
verify equality and restore the same way rather than committing them.

## Environment: several machines, one CUDA major version

This checkout is worked on from more than one box, and they differ in almost everything that a
setup instruction would normally hard-code: card model, card count, compute capability, conda
distribution, environment name, even the filesystem. **Do not assume which one you are on. Ask.**

```bash
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv,noheader
ls -d ~/anaconda3 ~/miniconda3 2>/dev/null      # which conda
conda env list                                   # which env has KeOps
python -c "import jax, torch; print(jax.__version__, jax.devices(), torch.__version__)"
```

The ones seen so far:

| | `dior` | the 5090 box |
|---|---|---|
| GPUs | **ten** RTX A5000, `sm_86`, 24 GB | **one** RTX 5090, `sm_120`, 32 GB |
| driver | 580.167.08 | 610.43.02 |
| conda | miniconda, env `jax_torch_latest` | anaconda3, env `jax_latest` |
| python / jax / torch | 3.12.13 / 0.11.1 / 2.12.1+cu130 | 3.12.12 / 0.11.1 / 2.12.0+cu130 |
| filesystem | ext4 | NTFS through ntfs-3g (`fuseblk`) |
| `/usr/local/cuda` | -> 13.3 (12.5.1 and 12.6.3 also installed) | -> 13.0, the only toolkit |
| `nvcc` on PATH | 12.6.85 (`/etc/profile` puts 12.6 first) | 13.0.48 |
| multi-GPU tests | run, with the cap below | skip: `test_sharding.py` needs two devices |

A third configuration, five RTX PRO 6000 Blackwell cards at `sm_120` on anaconda / `jax_latest`,
appears in older revisions of this file and in measurements below. Treat every measured number as
belonging to the box named beside it; none of them transfer.

**What is the same everywhere, and what the fork targets: CUDA 13.** Every box runs a driver that
advertises 13.x, JAX on the `jax_cuda13_plugin` with the `nvidia/cu13` wheels, and torch built for
`cu130`. So the JAX backend is a CUDA 13 target: the `.so` KeOps builds is `dlopen`ed into the JAX
process and shares its context, and both ends are 13. An environment carrying CUDA 12 JAX is not
this project's, whatever else is installed on the machine (`dior` has `jax_torch_3_12` and
`kernel_compiler` on cuda12, with no KeOps in them).

That invariant is what makes the toolkit resolution below safe: picking the environment's own
`nvcc` rather than the machine's is only sensible because the environment is always 13, even where
`/usr/local/cuda` is 13.0 on one box and 13.3 on another, and where `PATH` offers 12.6.

Activate deliberately on every box; do not trust the ambient `python`:

```bash
# dior
source /home/adam/miniconda3/etc/profile.d/conda.sh && conda activate jax_torch_latest
# the 5090 box
source /home/adam/anaconda3/etc/profile.d/conda.sh && conda activate jax_latest
```

`~/.bashrc` runs `conda init` but never `conda activate`, so a fresh interactive terminal lands in
`base`, whose python has no JAX (`ModuleNotFoundError: No module named 'jax'`). A Claude Code
session may nevertheless *look* correct, because it inherits whatever environment its launcher had.
That is a property of the session, not the machine: a clean login shell (`env -i bash -lc`) has no
python on `PATH` at all, since the `conda init` block only runs for interactive shells.

`keopscore` and `pykeops` are installed editable in the project's env on each box and already point
at that box's checkout. The JAX backend is CUDA only, with no CPU fallback, so it needs a GPU and a
CUDA toolkit with `nvcc`.

### One box, several CUDA versions: check the layer, not the machine (`dior`)

The major version is 13 everywhere (see above), but the *minor* version differs by layer, and on
`dior` three toolkits are installed at once. Nothing is broken -- the whole JAX suite and the
upstream NumPy suite pass -- but do not answer "what CUDA is this" from one command. This table is
`dior`'s; on the 5090 box there is a single toolkit, 13.0, and `nvcc` on `PATH` is 13.0.48 while
the environment's wheel is 13.2.78, so the same split exists in a milder form:

| layer | version | how it is chosen |
|---|---|---|
| driver | 580.167.08, advertises CUDA 13.0 | -- |
| toolkits under `/usr/local` | 12.5.1, 12.6.3, 13.3.0 | deb installs |
| `/usr/local/cuda` | -> `/usr/local/cuda-13.3` | `/etc/alternatives` |
| `nvcc` on `PATH` | 12.6.85 | `/etc/profile` puts `cuda-12.6/bin` first |
| `LD_LIBRARY_PATH` | `/usr/local/cuda-12.6/lib64` | `/etc/profile:32` |
| **the JAX backend's `nvcc`** | **13.3.73** | **the env's `nvidia-cuda-nvcc` wheel; see below** |
| keopscore's NVRTC | 13.3.33 | `/usr/local/cuda/targets/x86_64-linux/lib`, so the alternatives link |
| JAX | cuda13 wheels | `jax_cuda13_plugin`, `nvidia/cu13/lib` |
| PyTorch | built for CUDA 13.0 | `2.12.1+cu130` |

### The JAX backend builds against the environment's CUDA, not the machine's

The `.so` KeOps builds is `dlopen`ed into the JAX process and shares its CUDA context and driver,
so the CUDA it has to agree with is **JAX's**, which lives in the conda env and moves with it --
not whatever `/usr/local` happens to hold. pip installs a complete toolkit next to JAX for exactly
this: `nvidia-cuda-nvcc 13.3.73` puts an `nvcc` and matching headers under
`site-packages/nvidia/cu13/`.

`Cuda_link_compile.py` therefore resolves **nvcc and the `-I` headers from one root**, highest
priority first:

1. `CUDA_PATH` / `CUDA_HOME`, an explicit override;
2. the `nvidia/*` wheels of the running Python environment (`_find_env_cuda_toolkit()`), which is
   what normally wins and what keeps KeOps aligned with JAX;
3. `nvcc` on `PATH`, the historical behaviour;
4. a list of common install dirs, for launchers that start with no CUDA on `PATH`.

Whatever wins, the headers are taken from that same root when it ships them, so compiler and
headers can no longer disagree. `JAX_KEOPS_DEBUG=1` prints the pair it chose.

Until 2026-09-06 step 3 was the only rule, so on `dior` kernels were built by **nvcc 12.6**
against **CUDA 13.3 headers** -- `shutil.which` found `/etc/profile`'s 12.6 while the include path
came from the `/usr/local/cuda` alternatives link. It worked, but nothing made those two agree.
Two wheel layouts are handled: CUDA 13 consolidates into `nvidia/cu13/{bin,include}`, CUDA 12
splits per component (`nvidia/cuda_nvcc/bin`, `nvidia/cuda_runtime/include`). A machine with no
CUDA wheels falls through to the old behaviour unchanged.

The kernel cache hashes the formula, **not the compiler**, so switching toolkits leaves stale
`.so`s that are silently reused. `rm -rf ~/.cache/keops2.3` after any such change. To confirm what
built a kernel: `strings <cached>.so | grep -oE 'V1[23]\.[0-9.]+'`.

The NVRTC backend behind `pykeops.torch` and `pykeops.numpy` is untouched by this and still loads
NVRTC 13.3 from `/usr/local/cuda`; `keopscore/binders/cuda/` is fork-only and not on `main`.

### Never let JAX see more than eight GPUs (`dior` only)

CUDA allows at most eight peers per device and `dior` has ten cards, so **cap the device list**
there. It is unnecessary and harmless on any box with eight or fewer:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

Everything below assumes it is set. Without it, JAX's initialization tries to enable peer access
between every pair, GPU 9 fails against all nine others with `CUDA_ERROR_TOO_MANY_PEERS: peer
mapping resources exhausted`, and the allocator then fails every allocation on device 0 in turn,
walking 17.66 GiB down to 297 MiB without recovering. It is loud rather than silent, but it buries
the test output under thousands of lines. With the cap, `jax.devices()` returns 8 and the log is
clean. `test_sharding.py` needs at least two devices: it runs on `dior` either way, and skips its
per-device tests entirely on a single-card box.

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
`nvidia-smi --query-gpu=compute_cap` (`86` on `dior`, `120` on the 5090 box), falling back to
`70;75;80;86;89;90`. Should that
list ever not cover the card, it is harmless, and worth knowing so nobody chases it:
`keops_jax.cpp` is a dlopen shim and an FFI handler with no device code (no `__global__`, no
`<<<`), and the kernels that do run get their `-arch` from keopscore's own detection at runtime
(`compute capability 8.6 (arch=sm_86)` on `dior`, `12.0 (arch=sm_120)` on the 5090 box).
If JAX, nanobind, nvcc or cmake is missing, setup.py prints why, skips the extension and installs
only the Python side, so watch the install log rather than the exit code.

After editing `keops_jax.cpp` or its `CMakeLists.txt`, rebuild. The configured tree persists, so

```bash
cmake --build pykeops/build/jax_ext_build -j
```

is enough. Reinstalling also works but reconfigures from scratch.

Three layers, and what refreshes each: the Python binding is the editable checkout, so a pull is
its update; the C++ extension is rebuilt as above, or by reinstalling; the JIT-compiled kernels in
`~/.cache/keops2.3` (`KEOPS_CACHE_FOLDER`) are stale only after a launcher-template change that
leaves the kernel hash alone, and then `rm -rf` the folder. `pykeops/pykeops/jax/README.md` is the
install and update guide written for another machine, with the flags explained; keep it current.

Release packaging is `./pybuild.sh`, or `./pybuild.sh -l` for a local build with no hard-coded
versions.

## Testing

The JAX suite lives in `pykeops/pykeops/jax/test/` and has to run from that directory, because the
test files import `test_utils` as a top-level module.

```bash
cd pykeops/pykeops/jax/test
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # `dior` only, it has ten cards; see Environment
python run_tests.py              # all but the benchmarks; see the suite list below
python run_tests.py quick        # edge only
python run_tests.py api          # one suite
python run_tests.py correctness  # cross-check against pykeops.torch
python run_tests.py --float64    # sets KEOPS_TEST_FLOAT64=1 and JAX_ENABLE_X64=1; exits 1 on
                                 # the two complex tests of known bug 6, everything else passes
python test_api.py               # one file directly; run_tests.py just shells out to this
```

Suite names are api, correctness, edge, advanced, batched, broadcast, helpers, kernelsolve,
sharding, varifold, benchmark, benchmark-multi. Benchmarks are excluded from `all`; varifold runs
last in `all` because it is the slowest and the likeliest to OOM on a shared card.

Last full run on `dior`, 2026-09-06: all ten suites PASSED, exit 0 -- edge 18, api 22, correctness
49, advanced 36, batched 21, broadcast 11, helpers 11, kernelsolve 8, sharding 7, varifold 5. Re-run
green from a cold cache after the switch to the environment's nvcc 13.3.73, with all 174 cached
`*_jax.so` confirmed built by it (`strings ... | grep -oE 'V1[23]\.[0-9.]+'`) and none by 12.6. Note
these are check counts, not the pytest test counts below; the two runners count differently. The
first run of the day is slow because every kernel is compiled by nvcc from cold; a warm
`~/.cache/keops2.3` makes it minutes rather than tens of minutes. The A5000s have 24 GB against the
other box's 96, so varifold and the larger `check_at_scale` cases have far less headroom here --
run the suite on an idle card.

Every JAX test compares against `pykeops.torch` as ground truth and calls `sys.exit(1)` when
PyTorch with CUDA is absent, except `test_sharding.py` (known bug 5): it compares the multi-device
call against the single-device one, needs no torch, and skips its per-device tests below two GPUs.

pytest works from inside that directory too, and collects 87 tests in about 4 s:

```bash
pytest -q                          # the whole collection in one process
pytest test_api.py -k gradient     # select by name
pytest -m pytorch                  # every test here is marked gpu and pytorch
```

Know what those 87 cover before you trust a green run.

The tests come from nine files: `test_edge_cases` 18, `test_api` 13, `test_helpers` 11,
`test_batch_broadcasting` 11, `test_advanced` 10, `test_kernelsolve` 8, `test_sharding` 7,
`test_varifold_batched_grad` 5, `test_batched_gradients` 4.
`test_correctness.py` contributes none of them, because every function in it is a `check_*` driven
by `main()`. A passing `pytest` run has therefore cross-checked nothing against `pykeops.torch`;
`python run_tests.py correctness`, or `python test_correctness.py`, is what runs that. Helpers that
take arguments are named `check_*` rather than `test_*` on purpose, so pytest does not
mistake them for tests with missing fixtures, and you should follow that when adding one.

Keep test bodies inside functions. Until 2026-09-06 `test_kernelsolve.py` and
`test_varifold_batched_grad.py` did their work at module level, which meant pytest ran both while
importing them and collected no tests from either. Collection alone took 12 s and executed
`check_at_scale(21, 98776, 25000)`, whose own comment says it might OOM; had it, collection would
have errored and killed the run before a single test started. Both are ordinary test functions now,
wired into `run_tests.py` as the `kernelsolve` and `varifold` suites, and collection is down to
4.3 s with nothing executing.

pytest still runs every file in one process where `run_tests.py` forks per file, so
`test_high_dim_gradient` can fail under pytest on a busy card while passing on its own.
`run_tests.py` remains the runner the suites are written for.

`./pytest.sh` is the upstream harness. It builds a throwaway venv, installs both packages, clears
the cache, then runs `keopscore/keopscore/test/` and `pykeops/pykeops/test/`, which are the
PyTorch and NumPy suites. It never touches the JAX backend.
`.github/workflows/cuda_test.yml` runs it on a self-hosted GPU runner.

You do not need that venv to run one upstream test. From any directory except the repo root:

```bash
python -m pytest ~/keops/pykeops/pykeops/test/test_numpy.py           # 8 tests, ~2 s
python -m pytest ~/keops/pykeops/pykeops/test/test_lazytensor_grad.py
```

The keopscore half collects again now that the 0-byte `keopscore/__init__.py` is gone: 67 tests in
1.6 s, from any directory except the repo root, and from the root too under the bare `pytest`
script. Only `python -m pytest` from the root still fails, for the ordinary working-directory
reason and not that file -- `ImportError: cannot import name 'default_device_id' from 'pykeops'
(unknown location)`, where `(unknown location)` is the namespace-package tell.

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
- calls `_partitioned_ffi_call`, which wraps `jax.ffi.ffi_call(...)` (`kernel_id` and
  `batch_size` passed as FFI attributes) in a `custom_partitioning`, so a multi-device jit launches
  on each device's own samples or rows instead of replicating the call (known bug 5); inside a
  shard_map body, or for an operand pattern the rule does not describe, it is the bare `ffi_call`

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
- `JAX_KEOPS_DEBUG=1`, debug output from the C++ handler, from the chunking decision in
  `get_keops_dll.py`, and the `nvcc=... includes=...` pair the JAX backend resolved.
- `KEOPS_CACHE_FOLDER`, compiled-artifact cache. Defaults to `~/.cache/keops2.3`.
- `KEOPS_TEST_FLOAT64=1` together with `JAX_ENABLE_X64=1` for float64 test mode.
  `run_tests.py --float64` sets both.
- `CUDA_PATH`, `CUDA_ARCH`, `CXX`, `CXXFLAGS` feed keopscore's config detection. `CUDA_PATH` (or
  `CUDA_HOME`) is also the top-priority override for the JAX backend's nvcc *and* its headers, both
  taken from `$CUDA_PATH/{bin/nvcc,include}`; leave it unset to get the environment's own CUDA
  wheels, which is what keeps KeOps aligned with JAX.

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

- `_reject_vmap` (`generic_ops.py:264`) raises `NotImplementedError` naming KeOps batch dimensions
  as the route. It spots the vmap tracer by class name, since `BatchTracer` moved to `jax._src` in
  JAX 0.11 and is no longer re-exported from `jax.interpreters.batching`.
- The `ffi_call` passes `vmap_method=None` (`generic_ops.py:540`), which makes JAX itself refuse to
  batch the primitive. This used to be two call sites; the custom_partitioning work of known bug 5
  routed the forward and gradient launches through the one `_partitioned_ffi_call`, so there is a
  single place to keep it.

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

### 3. Async dispatch (measured 2026-09-03 on the five-card box, shelved)

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

### 5. Under XLA's SPMD partitioner the FFI call was REPLICATED (found 2026-09-06 in gsed, fixed the same day)

A JAX program `jit`ted over a multi-device mesh with batch-sharded inputs ran every KeOps
reduction on the WHOLE global batch on EVERY device. XLA has no partitioning rule for a custom
call it cannot see into, so it did the one safe thing: all-gathered the operands onto every
card, ran the reduction on all of them, and dynamic-sliced each card's own rows back out.
Measured in gsed's compiled training step on the five-RTX-PRO-6000 box, not `dior`: the varifold's `keops_*` and
`keops_jax_grad_*` calls took `f32[25,24778,3]` operands assembled by all-gather from the
cards' `[5,24778,3]` slices, with 16 partition-id slices after them -- each card doing the
work of all five, the largest term in the step at 5x its needed cost, the collectives
another 8% of GPU time. Nothing raised: the answer was right, five times over.

**The fix is `generic_ops._partitioned_ffi_call`**, which both launch sites (forward and
gradient) now go through. It wraps the FFI launch in `jax.experimental.custom_partitioning`
with a Shardy sharding rule built from what the binding already knows: the alias categories
(Vi / Vj / Pm) and whether a batch axis is present. Every output row of a KeOps reduction is a
complete reduction over the other index, so the rule frees exactly two factors, the batch
axis and the surviving row axis (`Vi`'s for `axis=1`, `Vj`'s for `axis=0`; a gradient launch
follows the same convention through its own axis), and marks the reduced row axis, every
feature axis and every parameter `need_replication`. The per-device computation is the same
`ffi_call` on the local shapes, whose batch size and output shape the binding derives at
trace time as before; `kernel_id` and the C++/FFI registration are untouched. Three details
that cost a round each: Shardy numbers factors by first appearance and requires the special
factors listed in that order (the verifier says "indices of special factors must be sorted"
otherwise); the JAX backend's first, registering call accepts JAX arrays or tracers only,
never numpy (`common/get_options.py::_find_mem`, pre-existing); and the partitioning
primitive has no differentiation rule, so a second-order derivative now raised a
NotImplementedError where the bare launch raised JAX's ValueError "cannot be differentiated"
-- the same limitation, which `test_advanced.py::test_hessian` matches on, so the launch
re-raises it in the original form.

What it does and does not do, pinned by `jax/test/test_sharding.py` -- originally on five GPUs,
and re-run on eight here (7 tests; the per-device ones skip below two GPUs, the rule-string one
runs anywhere):

* batch-sharded operands: each device launches on its own samples, value and gradients
  bitwise the single-device call's, no all-gather in the compiled program;
* a row shard of an unbatched call: each device reduces its own rows over all of j;
* a shard over the REDUCED axis is gathered before the launch and the numbers are right,
  never a partial sum -- the rule refuses to split j rather than combine it (a `psum` per
  reduction type would be the extension, not needed by any caller yet);
* a caller-side `shard_map` (gsed's fix before this rule, `check_vma=False`) still works:
  inside a shard_map body the mesh axes in context are manual and the launch is made as is;
* the LazyTensor path goes through the same launch and is covered.

The vmap refusal (#2) is intact: custom_partitioning's per-shard tracing produces no
BatchTracer. Ragged batches are a separate matter -- the binding batches by block-diagonal
ranges over a flattened axis, which is also KeOps' native way to hold meshes of different
sizes; a rule cutting on block boundaries would cover both, and exposing ragged ranges through
the JAX binding is the work that would come first.

### 6. Complex kernels do not work in float64 (open, and the only red in `--float64`)

Re-verified on `dior` 2026-09-06: exactly two failures, both in `advanced`, both complex, every
other suite green -- the count below still holds.

`run_tests.py --float64` passes 185 checks across 8 of the 10 suites and fails exactly two, both
in `test_advanced.py` section 6: `Complex: NUDFT (exp)` and `Complex: Real*Complex Mixed`, each a
`ValueError: Incompatible`. The other three complex tests in that section pass, as does every
non-complex suite.

This is not a regression. `3906bf1e` (2026-01-27) says so in its own subject line when it added
64-bit mode: "Doesn't support 64-bit complex kernels yet". It is a to-do, and until it is done
`--float64` exits 1 with those two errors and nothing else. Anyone reading a red float64 run should
check the count before assuming something broke; three failures, not two, means something new.

Deliberately left failing rather than skipped, so the gap stays visible. A `skipif` on
`is_float64_mode()` in `test_advanced.py` would make the suite green in one line if that is ever
preferred.

### 4. Importing pykeops.jax no longer pulls in torch (fixed 2026-09-03)

`pykeops/__init__.py` used to run `from .torch.test_install import test_torch_bindings` at import,
guarded only by a `find_spec` probe, so `import pykeops.jax` pulled `pykeops.torch` and therefore
torch. Both backend diagnostics are now resolved on first attribute access (PEP 562 module
`__getattr__`), so `pykeops.test_torch_bindings()` and `from pykeops import test_torch_bindings`
still work and nothing else changes. Measured over three runs each:

| `import pykeops.jax` | before | after |
|---|---|---|
| time | 1.08-1.13 s | 0.48-0.49 s |
| peak RSS | 658-660 MB | 212-214 MB |
| modules | 1776 | 929 |

That import site was the only one: with it removed, `torch` is absent from `sys.modules` after
`import pykeops.jax`.

**What the torch import was accidentally hiding.** This paragraph is the other machine's story;
what it means for `dior` is at the end. There, `~/.bashrc` put
`/usr/local/cuda/lib64` on `LD_LIBRARY_PATH`, which the loader searches before a binary's
`DT_RUNPATH`. The system CUDA was 13.0 (cuBLAS 13.0.0.19) and the pip wheels ship cuBLAS 13.1.0.3,
so a JAX-only process ended up with the wheel's `libcublas.so.13` and `libcublasLt.so.13` *and* the
system's `libcublasLt.so.13.0.0.19` mapped together, because the system cuBLAS needs its own
fully-versioned cuBLASLt. `jnp.linalg.cho_solve` then failed to launch outright ("Unable to launch
triangular solve"). Importing torch first hid it: `torch/__init__.py::_load_global_deps()` loads its
CUDA libraries by absolute path with `RTLD_GLOBAL` before anything else asks. KeOps does the same
via `CDLL(found_path, mode=RTLD_GLOBAL)` in `keopscore/config/cuda.py`, which is why importing
pykeops hid it too. JAX has the same mechanism
(`jax_plugins/xla_cuda13/__init__.py::_load_nvidia_libraries`) and it is not sufficient, because it
cannot stop the system cuBLAS dragging in its own cuBLASLt.

There, the line was removed from `~/.bashrc` on 2026-09-03 (backup at `~/.bashrc.bak-2026-09-03`); the
`PATH` export above it stays, since that is how `nvcc` is found. It should never have been there:
NVIDIA's installation guide scopes that step to the runfile installer ("when using the runfile
installation method, the LD_LIBRARY_PATH variable needs to contain /usr/local/cuda-X.Y/lib64"),
while the `PATH` export is unconditional. This box was installed from the deb local repo, and
package installs register the directory with ldconfig instead. The line came from a third-party
guide that carried the runfile instruction into a deb workflow. Nothing needed the library path:
the system CUDA is still discoverable through 69 entries in the ldconfig cache, an `nvcc`-built
CUDA program compiles and runs correctly without it, and both KeOps backends compile fresh kernels.
A torch-free JAX process now reports `lu 1.26e-06, cholesky 1.02e-06`.

**On `dior` the fault is dormant, not absent, and the cause is subtler than a clean path.**
`/etc/profile:32` sets `LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64` for every login on the box, so
unlike the other machine it is not yours to remove without root. It is harmless only because of a
soname mismatch: that directory ships `libcublas.so.12`, while JAX here runs the **cuda13** plugin
and asks for `libcublas.so.13`. Checked against a live process's `/proc/self/maps`, JAX maps the
wheel's `nvidia/cu13/lib/libcublas.so.13` and `libcublasLt.so.13` and nothing else, and
`jnp.linalg.solve` / `cho_solve` return `lu 8.34e-07, cholesky 7.15e-07`.

CUDA 13.3 is *also* installed here, at `/usr/local/cuda-13.3/lib64`, and it does ship
`libcublas.so.13`. Put that directory on `LD_LIBRARY_PATH`, or switch this env to cuda12 wheels
while 12.6 stays on the path, and the fault comes straight back. The rule is general: the fault
needs a system CUDA lib directory ahead of the wheels **at the same major version they use**.
Strip it for the process you launch rather than globally, and keep `gmtools.jax`'s
`check_linear_algebra()` as the tripwire: it raises at import with the full diagnosis.
