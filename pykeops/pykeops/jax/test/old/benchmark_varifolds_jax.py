#!/usr/bin/env python3
import os

import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad
from jax.tree_util import tree_map
from pykeops.jax import LazyTensor, Genred


@dataclass
class BenchmarkResult:
    name: str
    forward_time_ms: float
    forward_time_std_ms: float
    grad_time_ms: float
    grad_time_std_ms: float
    output_shape: Tuple[int, ...]
    output_mean: float
    output_std: float
    grad_norm: float
    num_runs: int
    device: str
    dtype: str


@dataclass
class ComparisonResult:
    name: str
    max_diff: float
    mean_diff: float
    relative_diff: float
    jax_time_ms: float
    pytorch_time_ms: float
    speedup: float
    passed: bool


def get_devices() -> List[jax.Device]:
    try:
        return jax.devices('gpu')
    except RuntimeError:
        return jax.devices('cpu')


def create_test_data(N: int, M: int, D: int, batch_size: int = 1, dtype: jnp.dtype = jnp.float32, seed: int = 42) -> \
Dict[str, jnp.ndarray]:
    np.random.seed(seed)

    if batch_size > 1:
        x_np = np.random.randn(batch_size, N, D).astype(np.float32)
        y_np = np.random.randn(batch_size, M, D).astype(np.float32)
        u_np = np.random.randn(batch_size, N, D).astype(np.float32)
        v_np = np.random.randn(batch_size, M, D).astype(np.float32)
        u_np = u_np / np.linalg.norm(u_np, axis=-1, keepdims=True)
        v_np = v_np / np.linalg.norm(v_np, axis=-1, keepdims=True)
    else:
        x_np = np.random.randn(N, D).astype(np.float32)
        y_np = np.random.randn(M, D).astype(np.float32)
        u_np = np.random.randn(N, D).astype(np.float32)
        v_np = np.random.randn(M, D).astype(np.float32)
        u_np = u_np / np.linalg.norm(u_np, axis=-1, keepdims=True)
        v_np = v_np / np.linalg.norm(v_np, axis=-1, keepdims=True)

    x = jnp.array(x_np, dtype=dtype)
    y = jnp.array(y_np, dtype=dtype)
    u = jnp.array(u_np, dtype=dtype)
    v = jnp.array(v_np, dtype=dtype)

    sigma_spatial = 0.5
    sigma_normal = 0.75
    gamma_spatial = jnp.array(1.0 / (2 * sigma_spatial ** 2), dtype=dtype)
    gamma_normal = jnp.array(1.0 / (sigma_normal ** 2), dtype=dtype)

    return {
        'x': x, 'y': y, 'u': u, 'v': v,
        'gamma_spatial': gamma_spatial,
        'gamma_normal': gamma_normal
    }


# --- LazyTensor Implementations ---

def varifold_kernel_lazytensor(xx, yy, uu, vv, gamma, gamma_1):
    if xx.ndim == 3:
        x = LazyTensor(xx[:, :, None, :])
        y = LazyTensor(yy[:, None, :, :])
        u = LazyTensor(uu[:, :, None, :])
        v = LazyTensor(vv[:, None, :, :])
        reduction_axis = 2
    else:
        x = LazyTensor(xx[:, None, :])
        y = LazyTensor(yy[None, :, :])
        u = LazyTensor(uu[:, None, :])
        v = LazyTensor(vv[None, :, :])
        reduction_axis = 1

    D2 = x.sqdist(y)
    ss = (u * v).sum(-1)
    K_spatial = (-D2 * gamma).exp()
    K_normal = (ss * gamma_1).exp()
    return (K_spatial * K_normal).sum(axis=reduction_axis)


def gaussian_kernel_lazytensor(x, y, gamma):
    if x.ndim == 3:
        x_i = LazyTensor(x[:, :, None, :])
        y_j = LazyTensor(y[:, None, :, :])
        reduction_axis = 2
    else:
        x_i = LazyTensor(x[:, None, :])
        y_j = LazyTensor(y[None, :, :])
        reduction_axis = 1
    D2 = x_i.sqdist(y_j)
    K = (-D2 * gamma).exp()
    return K.sum(axis=reduction_axis)


# --- Genred Implementations ---

def varifold_kernel_genred(xx, yy, uu, vv, gamma, gamma_1):
    D = xx.shape[-1]
    formula = "Exp(-g * SqDist(x, y)) * Exp(g1 * (u | v))"
    aliases = [
        f"x = Vi({D})", f"y = Vj({D})",
        f"u = Vi({D})", f"v = Vj({D})",
        "g = Pm(1)", "g1 = Pm(1)"
    ]
    my_genred = Genred(formula, aliases, reduction_op='Sum', axis=1)
    g = jnp.atleast_1d(gamma).reshape(-1)
    g1 = jnp.atleast_1d(gamma_1).reshape(-1)
    return my_genred(xx, yy, uu, vv, g, g1)


def gaussian_kernel_genred(x, y, gamma):
    D = x.shape[-1]
    formula = "Exp(-g * SqDist(x, y))"
    aliases = [f"x = Vi({D})", f"y = Vj({D})", "g = Pm(1)"]
    my_genred = Genred(formula, aliases, reduction_op='Sum', axis=1)
    g = jnp.atleast_1d(gamma).reshape(-1)
    return my_genred(x, y, g)


def benchmark_forward(kernel_fn, inputs, num_warmup=5, num_runs=20, use_jit=True):
    if use_jit:
        fn = jit(kernel_fn)
    else:
        fn = kernel_fn

    tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else None, inputs)

    for _ in range(num_warmup):
        result = fn(**inputs)
        result.block_until_ready()

    times = []
    for _ in range(num_runs):
        if 'result' in locals():
            result.block_until_ready()

        start = time.perf_counter()
        result = fn(**inputs)
        result.block_until_ready()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    # Standardize result dimensions
    # FIX: Use first available tensor to check dimension
    first_input = next(iter(inputs.values()))

    if result.ndim == 3 and result.shape[-1] == 1:
        result = result.squeeze(-1)
    elif result.ndim == 2 and result.shape[-1] == 1 and first_input.ndim == 2:
        result = result.squeeze(-1)

    return result, np.mean(times), np.std(times)


def benchmark_gradient(kernel_fn, inputs, num_warmup=5, num_runs=20):
    is_varifold = 'xx' in inputs

    if is_varifold:
        def loss_fn(xx, yy, uu, vv, gamma, gamma_1):
            return kernel_fn(xx, yy, uu, vv, gamma, gamma_1).sum()

        grad_fn = jit(grad(loss_fn, argnums=(0, 1, 2, 3)))
        args = (inputs['xx'], inputs['yy'], inputs['uu'], inputs['vv'], inputs['gamma'], inputs['gamma_1'])
    else:
        def loss_fn(x, y, gamma):
            return kernel_fn(x, y, gamma).sum()

        grad_fn = jit(grad(loss_fn, argnums=(0, 1)))
        args = (inputs['x'], inputs['y'], inputs['gamma'])

    for _ in range(num_warmup):
        grads = grad_fn(*args)
        tree_map(lambda x: x.block_until_ready(), grads)

    times = []
    for _ in range(num_runs):
        if 'grads' in locals():
            tree_map(lambda x: x.block_until_ready(), grads)

        start = time.perf_counter()
        grads = grad_fn(*args)
        tree_map(lambda x: x.block_until_ready(), grads)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    grad_norm = float(jnp.linalg.norm(grads[0]))
    return np.mean(times), np.std(times), grad_norm


def run_benchmark(name, kernel_fn, N, M, D, batch_size=1, num_warmup=5, num_runs=20, seed=42, pytorch_results=None,
                  pytorch_outputs=None):
    devices = get_devices()
    device_str = f"{devices[0].platform}:{devices[0].id}"
    print(f"\n{'-' * 60}\nBenchmark: {name} (N={N}, M={M}, B={batch_size})\n{'-' * 60}")

    data = create_test_data(N, M, D, batch_size, seed=seed)

    if 'varifold' in name.lower():
        inputs = {'xx': data['x'], 'yy': data['y'], 'uu': data['u'], 'vv': data['v'], 'gamma': data['gamma_spatial'],
                  'gamma_1': data['gamma_normal']}
    else:
        inputs = {'x': data['x'], 'y': data['y'], 'gamma': data['gamma_spatial']}

    result, fwd_mean, fwd_std = benchmark_forward(kernel_fn, inputs, num_warmup, num_runs)

    # --- DEBUG FINGERPRINT ---
    res_flat = result.flatten()
    print(f"  > FINGERPRINT: Mean={float(res_flat.mean()):.5f} | First={float(res_flat[0]):.5f}")
    # -------------------------

    pt_fwd_time = pytorch_results.get(name, {}).get('forward_time_ms') if pytorch_results else None
    speedup_str = ""
    if pt_fwd_time:
        speedup = pt_fwd_time / fwd_mean if fwd_mean > 0 else 0
        speedup_str = f" │ PyTorch: {pt_fwd_time:.3f} ms │ {Colors.GREEN if speedup >= 1 else Colors.RED}{speedup:.2f}x{Colors.END}"

    print(f"  Forward: {fwd_mean:.3f} ± {fwd_std:.3f} ms{speedup_str}")

    if 'varifold' in name.lower():
        grad_mean, grad_std, grad_norm = benchmark_gradient(kernel_fn, inputs, num_warmup, num_runs)
        print(f"  Gradient: {grad_mean:.3f} ± {grad_std:.3f} ms")
        print(f"  Grad norm: {grad_norm:.6f}")
    else:
        grad_mean, grad_std, grad_norm = 0.0, 0.0, 0.0

    result_np = np.array(result)
    return BenchmarkResult(
        name=name, forward_time_ms=fwd_mean, forward_time_std_ms=fwd_std,
        grad_time_ms=grad_mean, grad_time_std_ms=grad_std,
        output_shape=result_np.shape, output_mean=float(result_np.mean()), output_std=float(result_np.std()),
        grad_norm=grad_norm, num_runs=num_runs, device=device_str, dtype='float32'
    ), result_np


class Colors:
    HEADER, BLUE, CYAN, GREEN = '\033[95m', '\033[94m', '\033[96m', '\033[92m'
    YELLOW, RED, BOLD, UNDERLINE, END = '\033[93m', '\033[91m', '\033[1m', '\033[4m', '\033[0m'


def format_speedup(speedup: float) -> str:
    color = Colors.GREEN if speedup >= 1.0 else Colors.RED
    return f"{color}{speedup:.2f}x{Colors.END}"


def compare_with_pytorch(pytorch_file: str, jax_results: Dict, jax_outputs: Dict) -> List[ComparisonResult]:
    try:
        data = np.load(pytorch_file, allow_pickle=True)
        pt_results = json.loads(str(data['results']))
        pt_outputs = {k: data[k] for k in data.files if k != 'results'}
    except Exception as e:
        print(f"Could not load PyTorch results: {e}")
        return []

    comparisons = []
    print(f"\n{Colors.CYAN}{'=' * 80}\nCOMPARISON SUMMARY\n{'=' * 80}{Colors.END}")
    print(f"{'Benchmark':<35} {'JAX (ms)':<10} {'PT (ms)':<10} {'Speedup':<10} {'Max Diff':<10}")

    for name, jax_res in jax_results.items():
        if name in pt_results:
            pt_res = pt_results[name]
            jax_time = jax_res['forward_time_ms']
            pt_time = pt_res['forward_time_ms']
            speedup = pt_time / jax_time if jax_time > 0 else 0

            max_diff = 0.0
            passed = True
            if name in pt_outputs and name in jax_outputs:
                jax_out = jax_outputs[name]
                pt_out = pt_outputs[name]
                # Try simple reshape if sizes match (e.g. broadcasting diffs)
                if jax_out.shape != pt_out.shape:
                    if jax_out.size == pt_out.size:
                        jax_out = jax_out.reshape(pt_out.shape)
                try:
                    max_diff = float(np.abs(jax_out - pt_out).max())
                    passed = max_diff < 1e-4
                except ValueError:
                    max_diff = 999.0
                    passed = False

            diff_str = f"{max_diff:.1e}"
            diff_color = Colors.GREEN if passed else Colors.RED
            print(
                f"{name:<35} {jax_time:<10.2f} {pt_time:<10.2f} {format_speedup(speedup):<20} {diff_color}{diff_str}{Colors.END}")

    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pytorch-results', default='keops_pytorch_results.npz')
    parser.add_argument('--output', default='keops_jax_results.npz')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--multi-gpu', action='store_true')
    args = parser.parse_args()

    num_warmup = 3 if args.quick else 5
    num_runs = 10 if args.quick else 20

    standard_sizes = [(5000, 100, 3), (10000, 100, 3), (25000, 100, 3), (50000, 100, 3)]

    rectangular_sizes = [(2000, 2000, 3), (5000, 5000, 3), (10000, 10000, 3), (50000, 25000, 3), (100000, 50000, 3)]

    batch_scaling = [2, 10, 50]

    if not args.quick:
        standard_sizes.extend([(75000, 100, 3), (100000, 100, 3)])
        rectangular_sizes.append((20000, 20000, 3))

    pt_results, pt_outputs = None, None
    if Path(args.pytorch_results).exists():
        data = np.load(args.pytorch_results, allow_pickle=True)
        pt_results = json.loads(str(data['results']))
        pt_outputs = {k: data[k] for k in data.files if k != 'results'}

    all_results, all_outputs = {}, {}

    def run_suite(suite_name, sizes, is_batch=False):
        print(f"\n{Colors.CYAN}{'=' * 60}\nSECTION: {suite_name}\n{'=' * 60}{Colors.END}")
        for param in sizes:
            if is_batch:
                B = param
                N, M, D = 5000, 5000, 3
                suffix = f"batch_B{B}"
            else:
                N, M, D = param
                B = 4 if 'Rectangular' in suite_name else 1
                suffix = f"rect_N{N}_M{M}" if 'Rectangular' in suite_name else f"N{N}"

            res, out = run_benchmark(f"varifold_lazy_{suffix}", varifold_kernel_lazytensor, N, M, D, batch_size=B,
                                     num_warmup=num_warmup, num_runs=num_runs, pytorch_results=pt_results,
                                     pytorch_outputs=pt_outputs)
            all_results[res.name], all_outputs[res.name] = asdict(res), out

            res, out = run_benchmark(f"varifold_genred_{suffix}", varifold_kernel_genred, N, M, D, batch_size=B,
                                     num_warmup=num_warmup, num_runs=num_runs, pytorch_results=pt_results,
                                     pytorch_outputs=pt_outputs)
            all_results[res.name], all_outputs[res.name] = asdict(res), out

            if 'Rectangular' not in suite_name and not is_batch:
                res, out = run_benchmark(f"gaussian_lazy_{suffix}", gaussian_kernel_lazytensor, N, M, D, batch_size=B,
                                         num_warmup=num_warmup, num_runs=num_runs, pytorch_results=pt_results,
                                         pytorch_outputs=pt_outputs)
                all_results[res.name], all_outputs[res.name] = asdict(res), out
                res, out = run_benchmark(f"gaussian_genred_{suffix}", gaussian_kernel_genred, N, M, D, batch_size=B,
                                         num_warmup=num_warmup, num_runs=num_runs, pytorch_results=pt_results,
                                         pytorch_outputs=pt_outputs)
                all_results[res.name], all_outputs[res.name] = asdict(res), out

    run_suite("Standard N-Scaling", standard_sizes)
    run_suite("Batch Scaling", batch_scaling, is_batch=True)

    rect_B = batch_scaling[-1]
    print(f"\n{Colors.CYAN}{'=' * 60}\nSECTION: Rectangular (Batch={rect_B})\n{'=' * 60}{Colors.END}")
    for N, M, D in rectangular_sizes:
        suffix = f"rect_N{N}_M{M}"
        res, out = run_benchmark(f"varifold_lazy_{suffix}", varifold_kernel_lazytensor, N, M, D, batch_size=rect_B,
                                 num_warmup=num_warmup, num_runs=num_runs, pytorch_results=pt_results,
                                 pytorch_outputs=pt_outputs)
        all_results[res.name], all_outputs[res.name] = asdict(res), out
        res, out = run_benchmark(f"varifold_genred_{suffix}", varifold_kernel_genred, N, M, D, batch_size=rect_B,
                                 num_warmup=num_warmup, num_runs=num_runs, pytorch_results=pt_results,
                                 pytorch_outputs=pt_outputs)
        all_results[res.name], all_outputs[res.name] = asdict(res), out

    if pt_results:
        compare_with_pytorch(args.pytorch_results, all_results, all_outputs)

    np.savez(args.output, results=json.dumps(all_results), **all_outputs)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()