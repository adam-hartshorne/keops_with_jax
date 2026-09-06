"""The FFI launch under XLA's SPMD partitioner (CLAUDE.md known bug 5, fixed 2026-09-06).

A multi-device `jit` with sharded inputs used to REPLICATE every KeOps reduction: XLA had no
partitioning rule for the custom call, so it all-gathered the operands onto every device and ran the
whole reduction everywhere. `generic_ops._partitioned_ffi_call` now declares the batch axis and the
surviving row axis independent, so each device reduces its own rows. Pinned here: the numbers equal
the single-device call's, the compiled program launches KeOps on PER-DEVICE operands with no
all-gather feeding them, a shard over the reduced axis is gathered rather than answered wrongly, and
the caller-side `shard_map` gsed used before this rule still works with it.

Run from this directory:  python -m pytest test_sharding.py -q
The per-device checks need two or more GPUs and skip otherwise; the rule-string test runs anywhere.
Precision follows KEOPS_TEST_FLOAT64 like the other suites, so `run_tests.py --float64` exercises
this file too.
"""
import re
import sys

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from test_utils import get_dtype_str, get_np_dtype


def _gpus():
    try:
        return jax.devices("gpu")
    except RuntimeError:
        return []


needs_gpu = pytest.mark.skipif(not _gpus(), reason="KeOps is CUDA")
needs_multi = pytest.mark.skipif(len(_gpus()) < 2, reason="the per-device checks need two or more GPUs")

FORMULA = "Exp(-g * SqDist(x, y)) * b"
ALIASES = ["x = Vi(3)", "y = Vj(3)", "b = Vj(2)", "g = Pm(1)"]


def _op(axis=1):
    from pykeops.jax import Genred
    return Genred(FORMULA, ALIASES, reduction_op="Sum", axis=axis, dtype=get_dtype_str())


def _data(B, N, M, seed=0):
    r = np.random.default_rng(seed)
    dt = get_np_dtype()
    x = r.normal(size=(B, N, 3)).astype(dt)
    y = r.normal(size=(B, M, 3)).astype(dt)
    b = r.normal(size=(B, M, 2)).astype(dt)
    g = np.array([0.5], dt)
    # JAX arrays, not numpy: the first (registering) call of the JAX backend reads the device off its
    # arguments and accepts only JAX arrays or tracers (`pykeops/common/get_options.py::_find_mem`).
    return tuple(jnp.asarray(a) for a in (x, y, b, g))


def _dense(x, y, b, g):
    x, y, b, g = (np.asarray(a) for a in (x, y, b, g))
    d2 = ((x[:, :, None, :] - y[:, None, :, :]) ** 2).sum(-1)
    return (np.exp(-g[0] * d2)[..., None] * b[:, None, :, :]).sum(2)


def _loss(op):
    return lambda x, y, b, g: jnp.sum(op(x, y, b, g) ** 2)


def _mesh():
    devs = _gpus()
    return Mesh(np.array(devs), axis_names=("b",)), len(devs)


def _launches(hlo):
    """Every KeOps launch in a compiled program as (result shape, launch text); the compiled text does
    not print operand shapes, so the result shape is what a check reads. The element type follows the
    test precision now that the data does, so the pattern has to as well; every caller asserts the
    match is non-empty, so a stale pattern fails loudly rather than skipping the checks."""
    et = "f64" if get_dtype_str() == "float64" else "f32"
    return [(tuple(int(v) for v in res.split(",")), line)
            for res, line in re.findall(
                r'= ' + et + r'\[([\d,]+)\]([^\n]*custom_call_target="keops[^\n]*)', hlo)]


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def test_the_sharding_rule_describes_every_launch_pattern():
    """Runs anywhere. The rule strings parse (with the parser JAX applies at lowering, a private path
    used only here) and free exactly the batch and the surviving rows."""
    from jax._src.custom_partitioning_sharding_rule import str_to_sdy_sharding_rule
    from pykeops.jax.generic.generic_ops import _sharding_rule
    cases = [   # var_cats, ranks, axis, out_rank -> the rule
        ((0, 1, 1, 2), (3, 3, 3, 1), 1, 3, "b i f0, b j f1, b j f2, p3_0 -> b i o"),
        ((0, 1, 1, 2), (2, 2, 2, 1), 1, 2, "i f0, j f1, j f2, p3_0 -> i o"),
        ((0, 1, 2), (3, 3, 2), 0, 3, "b j f0, b i f1, p2_0 p2_1 -> b i o"),                     # axis 0 keeps the Vj rows
        ((0, 1, 1, 2, 0), (3, 3, 3, 1, 3), 1, 3, "b i f0, b j f1, b j f2, p3_0, b i f4 -> b i o"),  # a gradient launch: eta rides with the surviving rows
    ]
    for var_cats, ranks, axis, out_rank, want in cases:
        rule, rep = _sharding_rule(var_cats, ranks, axis, out_rank)
        assert rule == want, rule
        assert "j" in rep and "o" in rep and "b" not in rep and "i" not in rep, rep
        order = list(dict.fromkeys(rule.replace(",", " ").replace("->", " ").split()))     # first appearance
        assert list(rep) == [f for f in order if f in rep], "replicated factors must be listed in order of first appearance"
        str_to_sdy_sharding_rule(rule, need_replication_factors=rep)      # raises on a malformed rule
    with pytest.raises(ValueError):
        _sharding_rule((1, 2), (3, 1), 1, 3)        # nothing carries the surviving rows
    with pytest.raises(ValueError):
        _sharding_rule((0, 1), (3, 2), 1, 3)        # batched and unbatched variables mixed


@needs_gpu
def test_on_one_device_the_launch_is_unchanged():
    op = _op()
    x, y, b, g = _data(2, 50, 60)
    eager = op(x, y, b, g)
    jitted = jax.jit(op)(x, y, b, g)
    assert np.array_equal(np.asarray(eager), np.asarray(jitted))
    assert _rel(jitted, _dense(x, y, b, g)) < 1e-4


@needs_gpu
@needs_multi
def test_a_batch_shard_runs_each_device_on_its_own_samples():
    mesh, n = _mesh()
    op = _op()
    x, y, b, g = _data(2 * n, 300, 400)
    vg = jax.jit(jax.value_and_grad(_loss(op), argnums=(0, 1, 2, 3)))
    ref_v, ref_g = vg(x, y, b, g)                                                  # one device
    sh, rep = NamedSharding(mesh, P("b")), NamedSharding(mesh, P())
    args = (jax.device_put(x, sh), jax.device_put(y, sh), jax.device_put(b, sh), jax.device_put(g, rep))
    v, grads = vg(*args)
    hlo = vg.lower(*args).compile().as_text()
    assert abs(float(v) - float(ref_v)) <= 1e-5 * abs(float(ref_v)), (float(v), float(ref_v))
    worst = max(_rel(a, r) for a, r in zip(grads, ref_g))
    assert worst < 1e-5, worst
    launches = _launches(hlo)
    assert launches, "no KeOps launch in the compiled program"
    bad = [res for res, _ in launches if res[0] != 2]
    assert not bad, f"launches on a batch other than the per-device 2: {bad} (the global batch is {2 * n})"
    assert "all-gather" not in hlo, "an all-gather feeds a launch: the call is still replicated"
    print(f"{len(launches)} launches at batch 2 on {n} GPUs, value and gradients to {worst:.1e}, no all-gather")


@needs_gpu
@needs_multi
def test_a_row_shard_of_an_unbatched_call_runs_each_device_on_its_own_rows():
    mesh, n = _mesh()
    op = _op()
    x, y, b, g = _data(1, 40 * n, 300)
    x, y, b = x[0], y[0], b[0]
    f = jax.jit(op)
    ref = f(x, y, b, g)
    rep = NamedSharding(mesh, P())
    args = (jax.device_put(x, NamedSharding(mesh, P("b"))), jax.device_put(y, rep), jax.device_put(b, rep), jax.device_put(g, rep))
    out = f(*args)
    hlo = f.lower(*args).compile().as_text()
    assert _rel(out, ref) < 1e-6
    launches = _launches(hlo)
    assert launches and all(res[0] == 40 for res, _ in launches), launches
    assert "all-gather" not in hlo
    assert out.sharding.spec[0] == "b", out.sharding


@needs_gpu
@needs_multi
def test_a_shard_over_the_reduced_axis_is_gathered_not_answered_wrongly():
    mesh, n = _mesh()
    op = _op()
    x, y, b, g = _data(2, 100, 60 * n)
    f = jax.jit(op)
    ref = f(x, y, b, g)
    rep, jsh = NamedSharding(mesh, P()), NamedSharding(mesh, P(None, "b"))        # the j rows split across devices
    args = (jax.device_put(x, rep), jax.device_put(y, jsh), jax.device_put(b, jsh), jax.device_put(g, rep))
    out = f(*args)
    hlo = f.lower(*args).compile().as_text()
    assert _rel(out, ref) < 1e-6
    launches = _launches(hlo)
    # nothing the rule frees is sharded, so every device runs the whole reduction on gathered operands
    assert launches and all(res == (2, 100, 2) for res, _ in launches), launches
    assert "all-gather" in hlo, "the j-sharded operands were not gathered before the launch"


@needs_gpu
@needs_multi
def test_inside_a_caller_shard_map_the_launch_is_made_as_is():
    """gsed's caller-side fix, a shard_map over the batch with check_vma=False, must keep working."""
    mesh, n = _mesh()
    op = _op()
    x, y, b, g = _data(2 * n, 200, 250)
    ref = jax.jit(op)(x, y, b, g)
    f = jax.jit(jax.shard_map(lambda x, y, b, g: op(x, y, b, g), mesh=mesh,
                              in_specs=(P("b"), P("b"), P("b"), P()), out_specs=P("b"), check_vma=False))
    sh, rep = NamedSharding(mesh, P("b")), NamedSharding(mesh, P())
    out = f(jax.device_put(x, sh), jax.device_put(y, sh), jax.device_put(b, sh), jax.device_put(g, rep))
    assert _rel(out, ref) < 1e-6


@needs_gpu
@needs_multi
def test_the_lazytensor_path_is_partitioned_too():
    from pykeops.jax import LazyTensor
    mesh, n = _mesh()

    def f(x, y, b):
        xi, yj, bj = LazyTensor(x[:, :, None, :]), LazyTensor(y[:, None, :, :]), LazyTensor(b[:, None, :, :])
        return ((-0.5 * ((xi - yj) ** 2).sum(-1)).exp() * bj).sum(dim=2)

    x, y, b, _ = _data(2 * n, 120, 130)
    jf = jax.jit(f)
    ref = jf(x, y, b)
    sh = NamedSharding(mesh, P("b"))
    args = (jax.device_put(x, sh), jax.device_put(y, sh), jax.device_put(b, sh))
    out = jf(*args)
    hlo = jf.lower(*args).compile().as_text()
    assert _rel(out, ref) < 1e-6
    launches = _launches(hlo)
    assert launches and all(res[0] == 2 for res, _ in launches), launches
    assert "all-gather" not in hlo


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
