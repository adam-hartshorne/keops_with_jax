# KeOps JAX backend (`pykeops.jax`)

This fork (branch `jax_api`) adds a JAX backend to KeOps: `Genred`, `LazyTensor` with `Vi`, `Vj`,
`Pm`, the `generic_sum` / `generic_logsumexp` / `generic_argmin` / `generic_argkmin` / `generic_min` /
`generic_max` helpers, and `KernelSolve`. A reduction is compiled by KeOps' own engine with `nvcc`
into a shared library and called from JAX through XLA's FFI. It is CUDA only, with no CPU fallback.

**The PyPI wheel does not contain it.** `pip install pykeops` installs cleanly and `import pykeops`
succeeds, and the first `import pykeops.jax` then fails with `No module named 'pykeops.jax'`. The
backend exists only as an editable install of this checkout, so `import pykeops.jax` is the one check
that means anything.

## Requirements

- A CUDA GPU, and a CUDA toolkit with `nvcc` on `PATH`.
- `cmake` and `nanobind`, for the C++ FFI extension.
- JAX with CUDA support. This branch tracks jax 0.11.1.
- Python 3.12. On Adam's machines that is the conda env `jax_latest`.
- PyTorch with CUDA only for the tests that cross-check against `pykeops.torch`. `pykeops.jax`
  itself does not import torch.

## Install, from the fork root

```bash
cd ~/keops                                                   # branch jax_api
pip install -e ./keopscore --no-build-isolation --no-deps
pip install -e ./pykeops   --no-build-isolation --no-deps
cd ~ && python -c "import pykeops.jax; print('ok')"          # from any directory EXCEPT the fork root
```

Each part of that matters:

- `keopscore` first, because `pykeops/setup.py` imports it.
- `--no-deps`, or pip replaces the editable checkout with the release wheel, which has no JAX
  backend. `setup.py` prints a warning when it sees that happen.
- `--no-build-isolation`, so the build sees the environment's JAX and nanobind.
- Installing `pykeops` also builds the C++ FFI extension: CMake runs over
  `pykeops/pykeops/jax/binders/` and writes `keops_jax_ext.cpython-*.so` into `pykeops/pykeops/jax/`.
  The `.so` is gitignored, so a fresh clone has none until you install. `CMAKE_CUDA_ARCHITECTURES`
  is read from `nvidia-smi --query-gpu=compute_cap`, falling back to `70;75;80;86;89;90`.
- Read the install log, not the exit code. If JAX, nanobind, `nvcc` or `cmake` is missing,
  `setup.py` says why, skips the extension and installs only the Python side, and exits 0.
- Never run Python at the fork root. The source directories there shadow the installed packages,
  and the failure reads `module 'keopscore' has no attribute '__version__'`.

## The three layers, and what refreshes each

| layer | where it lives | when it needs redoing |
|---|---|---|
| the Python binding (`pykeops/pykeops/jax/`) | this checkout, through the editable install | never once installed editable: a `git pull` is the update |
| the C++ FFI extension | `pykeops/pykeops/jax/keops_jax_ext.cpython-*.so`, gitignored, built by CMake at install | after editing `binders/keops_jax.cpp` or its `CMakeLists.txt`, or on a fresh clone |
| the JIT-compiled CUDA kernels | `~/.cache/keops2.3/` (or `KEOPS_CACHE_FOLDER`), one per formula, batch layout and dtype | only after a change to the launcher template that does not change the kernel hash |

Most commits touch the first layer only. The 2026-09-06 partitioning fix (below) is one of them:
a pull is the whole update.

## Updating on another machine

```bash
cd ~/keops && git pull                       # branch jax_api
pip show pykeops | grep Editable             # must name this checkout; if it does not, install as above
cd ~ && python -c "import pykeops.jax"
```

Then, only if the pulled commits touched the layer:

```bash
cmake --build ~/keops/pykeops/build/jax_ext_build -j    # the C++ extension changed (or just reinstall)
rm -rf ~/.cache/keops2.3                                # the launcher template changed under an unchanged hash
```

Reinstalling with the two `pip install -e` lines above always works and overwrites the extension in
place; it reconfigures CMake from scratch, which is the only cost.

## Tests

The JAX suite lives in `pykeops/pykeops/jax/test/` and runs from inside that directory (see the
last install point). `python run_tests.py` runs the suites `edge`, `api`, `correctness`,
`advanced`, `batched`, `broadcast`, `helpers` and `sharding`; the two `benchmark*` suites are
excluded from `all`. `pytest -q` from the same directory collects 74 tests. Every suite except
`sharding` compares against `pykeops.torch` as ground truth and exits rather than runs without
PyTorch with CUDA; `sharding` compares the multi-device call against the single-device one and
skips its per-device tests below two GPUs.

## Several GPUs: a `jit` over a mesh

Since commit d579e35d (2026-09-06) the launch partitions itself. Under a multi-device `jit` with
batch-sharded inputs each device reduces its own samples; a row-sharded unbatched call has each
device reduce its own rows over all of `j`. Operands sharded over the reduced axis, over a feature
axis or over a parameter are gathered before the launch, so the numbers are always the plain
call's, never a partial sum. Before that commit XLA, having no rule for the custom call, replicated
it: every device all-gathered the whole batch and reduced all of it. A caller-side `shard_map`
over the batch (with `check_vma=False`, since `ffi_call` drops the varying-axes type) works too
and is bit for bit the same computation. The rule is `generic_ops._partitioned_ffi_call` and its
record is `test/test_sharding.py`; CLAUDE.md, known bug 5, has the details.

## Limits

- No `jax.vmap` over a reduction: the FFI call has no batching rule and the binding refuses rather
  than answer wrongly. Pass `(B, N, D)` arrays and let KeOps batch. `jax.jacfwd` and `jax.jacrev`
  map internally and reach the same refusal.
- No second-order derivatives: a `jvp` of a `grad` raises JAX's "cannot be differentiated".
- One batch dimension, and every batched `i` / `j` variable carries it (size 1 broadcasts).
- A `Pm` parameter cannot vary along the batch.
- The first, registering call of a reduction takes JAX arrays or tracers, not numpy arrays.
