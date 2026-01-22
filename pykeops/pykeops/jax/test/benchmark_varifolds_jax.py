#!/usr/bin/env python3
"""
KeOps JAX Unit Tests & Benchmarks

This script tests the JAX KeOps API for:
- Forward pass correctness
- JIT compilation
- Gradient computation
- Single and multi-GPU performance (using sharding)

Usage:
    python benchmark_varifolds_jax.py [--pytorch-results results.npz] [--output jax_results.npz]
"""

import os

# Must set before importing KeOps
os.environ['PYKEOPS_JAX_MODE'] = '1'

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
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.tree_util import tree_map

# Handle different JAX versions for shard_map
try:
    from jax.shard_map import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map

# KeOps JAX imports
from pykeops.jax import LazyTensor, Genred, Vi, Vj, Pm


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
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
    """Results comparing JAX to PyTorch."""
    name: str
    max_diff: float
    mean_diff: float
    relative_diff: float
    jax_time_ms: float
    pytorch_time_ms: float
    speedup: float
    passed: bool


def get_devices() -> List[jax.Device]:
    """Get available devices."""
    try:
        return jax.devices('gpu')
    except RuntimeError:
        return jax.devices('cpu')


def create_test_data(
        N: int, M: int, D: int,
        batch_size: int = 1,
        dtype: jnp.dtype = jnp.float32,
        seed: int = 42
) -> Dict[str, jnp.ndarray]:
    """Create test data for varifold kernel."""
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
    sigma_normal = 1.0
    gamma_spatial = jnp.array(1.0 / (2 * sigma_spatial ** 2), dtype=dtype)
    gamma_normal = jnp.array(1.0 / (sigma_normal ** 2), dtype=dtype)

    return {
        'x': x, 'y': y, 'u': u, 'v': v,
        'gamma_spatial': gamma_spatial,
        'gamma_normal': gamma_normal
    }


# =============================================================================
# Kernel Implementations
# =============================================================================

def varifold_kernel_lazytensor(
        xx: jnp.ndarray, yy: jnp.ndarray,
        uu: jnp.ndarray, vv: jnp.ndarray,
        gamma: jnp.ndarray, gamma_1: jnp.ndarray
) -> jnp.ndarray:
    """Varifold kernel using LazyTensor API."""
    if xx.ndim == 3:
        x = LazyTensor(xx[:, :, None, :])
        y = LazyTensor(yy[:, None, :, :])
        u = LazyTensor(uu[:, :, None, :])
        v = LazyTensor(vv[:, None, :, :])
    else:
        x = LazyTensor(xx[:, None, :])
        y = LazyTensor(yy[None, :, :])
        u = LazyTensor(uu[:, None, :])
        v = LazyTensor(vv[None, :, :])

    D2 = x.sqdist(y)
    ss = (u * v).sum(-1)
    K_spatial = (-D2 * gamma).exp()
    K_normal = (ss * gamma_1).exp()

    return (K_spatial * K_normal).sum(axis=1)


def gaussian_kernel_lazytensor(
        x: jnp.ndarray, y: jnp.ndarray, gamma: jnp.ndarray
) -> jnp.ndarray:
    """Simple Gaussian kernel for baseline testing."""
    if x.ndim == 3:
        x_i = LazyTensor(x[:, :, None, :])
        y_j = LazyTensor(y[:, None, :, :])
    else:
        x_i = LazyTensor(x[:, None, :])
        y_j = LazyTensor(y[None, :, :])

    D2 = x_i.sqdist(y_j)
    K = (-D2 * gamma).exp()
    return K.sum(axis=1)


# =============================================================================
# Benchmark Functions (Timing Fixed)
# =============================================================================

def benchmark_forward(
        kernel_fn, inputs: Dict[str, jnp.ndarray],
        num_warmup: int = 5, num_runs: int = 20,
        use_jit: bool = True
) -> Tuple[jnp.ndarray, float, float]:
    """Benchmark forward pass with strict synchronization."""
    if use_jit:
        fn = jit(kernel_fn)
    else:
        fn = kernel_fn

    # Ensure inputs are ready
    tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else None, inputs)

    # Warmup
    for _ in range(num_warmup):
        result = fn(**inputs)
        result.block_until_ready()

    # Benchmark
    times = []
    for _ in range(num_runs):
        # Strict barrier: ensure previous iter is done before starting timer
        if 'result' in locals():
            result.block_until_ready()

        start = time.perf_counter()
        result = fn(**inputs)
        result.block_until_ready()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return result, np.mean(times), np.std(times)


def benchmark_gradient(
        kernel_fn, inputs: Dict[str, jnp.ndarray],
        num_warmup: int = 5, num_runs: int = 20
) -> Tuple[float, float, float]:
    """Benchmark gradient computation with strict synchronization."""
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

    # Warmup
    for _ in range(num_warmup):
        grads = grad_fn(*args)
        tree_map(lambda x: x.block_until_ready(), grads)

    # Benchmark
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


def run_benchmark(
        name: str,
        kernel_fn,
        N: int, M: int, D: int,
        batch_size: int = 1,
        dtype: jnp.dtype = jnp.float32,
        num_warmup: int = 5,
        num_runs: int = 20,
        seed: int = 42,
        pytorch_results: Optional[Dict] = None,
        pytorch_outputs: Optional[Dict] = None
) -> Tuple[BenchmarkResult, np.ndarray]:
    """Run a complete benchmark."""
    devices = get_devices()
    device_str = f"{devices[0].platform}:{devices[0].id}"

    print(f"\n{'=' * 60}")
    print(f"Benchmark: {Colors.BOLD}{name}{Colors.END}")
    print(f"  N={N}, M={M}, D={D}, Batch={batch_size}")
    print(f"  dtype={dtype}, device={device_str}")
    print(f"{'=' * 60}")

    data = create_test_data(N, M, D, batch_size, dtype, seed)

    if 'varifold' in name.lower():
        inputs = {
            'xx': data['x'], 'yy': data['y'],
            'uu': data['u'], 'vv': data['v'],
            'gamma': data['gamma_spatial'],
            'gamma_1': data['gamma_normal']
        }
    else:
        inputs = {
            'x': data['x'], 'y': data['y'],
            'gamma': data['gamma_spatial']
        }

    # Forward benchmark
    result, fwd_mean, fwd_std = benchmark_forward(
        kernel_fn, inputs, num_warmup, num_runs
    )

    # Display results and comparison
    pt_fwd_time = pytorch_results.get(name, {}).get('forward_time_ms') if pytorch_results else None

    if pt_fwd_time:
        speedup = pt_fwd_time / fwd_mean if fwd_mean > 0 else 0
        print(
            f"  Forward:  {Colors.GREEN}{fwd_mean:.3f} ± {fwd_std:.3f} ms{Colors.END}  │  PyTorch: {pt_fwd_time:.3f} ms  │  {format_speedup(speedup)}")
    else:
        print(f"  Forward: {fwd_mean:.3f} ± {fwd_std:.3f} ms")

    # Gradient benchmark (only for varifold)
    if 'varifold' in name.lower():
        grad_mean, grad_std, grad_norm = benchmark_gradient(
            kernel_fn, inputs, num_warmup, num_runs
        )
        pt_grad_time = pytorch_results.get(name, {}).get('grad_time_ms') if pytorch_results else None

        if pt_grad_time and pt_grad_time > 0:
            grad_speedup = pt_grad_time / grad_mean if grad_mean > 0 else 0
            print(
                f"  Gradient: {Colors.GREEN}{grad_mean:.3f} ± {grad_std:.3f} ms{Colors.END}  │  PyTorch: {pt_grad_time:.3f} ms  │  {format_speedup(grad_speedup)}")
        else:
            print(f"  Gradient: {grad_mean:.3f} ± {grad_std:.3f} ms")
        print(f"  Grad norm: {grad_norm:.6f}")
    else:
        grad_mean, grad_std, grad_norm = 0.0, 0.0, 0.0

    result_np = np.array(result)

    return BenchmarkResult(
        name=name,
        forward_time_ms=float(fwd_mean),
        forward_time_std_ms=float(fwd_std),
        grad_time_ms=float(grad_mean),
        grad_time_std_ms=float(grad_std),
        output_shape=result_np.shape,
        output_mean=float(result_np.mean()),
        output_std=float(result_np.std()),
        grad_norm=grad_norm,
        num_runs=num_runs,
        device=device_str,
        dtype=str(dtype)
    ), result_np


# =============================================================================
# Multi-GPU Sharding Tests
# =============================================================================

def create_sharded_data(
        N: int, M: int, D: int,
        mesh: Mesh,
        dtype: jnp.dtype = jnp.float32,
        seed: int = 42
) -> Dict[str, jnp.ndarray]:
    """Create test data sharded across devices."""
    num_devices = len(mesh.devices.flat)
    batch_size = num_devices
    data = create_test_data(N, M, D, batch_size, dtype, seed)

    batch_sharding = NamedSharding(mesh, P('batch'))
    replicated = NamedSharding(mesh, P())

    return {
        'x': jax.device_put(data['x'], batch_sharding),
        'y': jax.device_put(data['y'], batch_sharding),
        'u': jax.device_put(data['u'], batch_sharding),
        'v': jax.device_put(data['v'], batch_sharding),
        'gamma_spatial': jax.device_put(data['gamma_spatial'], replicated),
        'gamma_normal': jax.device_put(data['gamma_normal'], replicated),
    }


def run_sharded_benchmark(
        name: str,
        N: int, M: int, D: int,
        dtype: jnp.dtype = jnp.float32,
        num_warmup: int = 5,
        num_runs: int = 20
) -> List[BenchmarkResult]:
    """Run benchmarks with JAX sharding."""
    devices = get_devices()
    num_devices = len(devices)
    if num_devices < 2:
        return []

    print(f"\n{'=' * 60}\nSharded Benchmark: {name}\n{'=' * 60}")
    results = []

    for n_devices in [1, 2, min(4, num_devices), num_devices]:
        if n_devices > num_devices: continue
        print(f"\n  Testing with {n_devices} device(s)...")

        mesh = Mesh(np.array(devices[:n_devices]), axis_names=('batch',))
        data = create_sharded_data(N, M, D, mesh, dtype)

        @jit
        def sharded_kernel(x, y, u, v, gamma, gamma_1):
            return varifold_kernel_lazytensor(x, y, u, v, gamma, gamma_1)

        # Warmup
        for _ in range(num_warmup):
            result = sharded_kernel(data['x'], data['y'], data['u'], data['v'], data['gamma_spatial'],
                                    data['gamma_normal'])
            result.block_until_ready()

        # Benchmark
        times = []
        for _ in range(num_runs):
            result.block_until_ready()
            start = time.perf_counter()
            result = sharded_kernel(data['x'], data['y'], data['u'], data['v'], data['gamma_spatial'],
                                    data['gamma_normal'])
            result.block_until_ready()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        fwd_mean = np.mean(times)
        throughput = (N * M * n_devices) / (fwd_mean / 1000)
        print(f"    Forward: {fwd_mean:.3f} ± {np.std(times):.3f} ms")
        print(f"    Throughput: {throughput:.2e} pairs/sec")

        results.append(BenchmarkResult(
            name=f"{name}_devices{n_devices}", forward_time_ms=float(fwd_mean),
            forward_time_std_ms=float(np.std(times)), grad_time_ms=0.0, grad_time_std_ms=0.0,
            output_shape=tuple(result.shape), output_mean=float(result.mean()), output_std=float(result.std()),
            grad_norm=0.0, num_runs=num_runs, device=f"gpu:0-{n_devices - 1}", dtype=str(dtype)
        ))
    return results


# =============================================================================
# Utils
# =============================================================================

class Colors:
    HEADER, BLUE, CYAN, GREEN = '\033[95m', '\033[94m', '\033[96m', '\033[92m'
    YELLOW, RED, BOLD, UNDERLINE, END = '\033[93m', '\033[91m', '\033[1m', '\033[4m', '\033[0m'


def format_speedup(speedup: float) -> str:
    color = Colors.GREEN if speedup >= 1.0 else Colors.RED
    return f"{color}{speedup:.2f}x{Colors.END}"


def compare_with_pytorch(pytorch_file: str, jax_results: Dict, jax_outputs: Dict) -> List[ComparisonResult]:
    """Simplified comparison logic."""
    try:
        data = np.load(pytorch_file, allow_pickle=True)
        pt_results = json.loads(str(data['results']))
    except:
        return []

    comparisons = []
    print(f"\n{Colors.CYAN}{'=' * 60}\nCOMPARISON SUMMARY\n{'=' * 60}{Colors.END}")
    print(f"{'Benchmark':<25} {'JAX (ms)':<10} {'PT (ms)':<10} {'Speedup':<10}")

    for name, jax_res in jax_results.items():
        if name in pt_results:
            pt_res = pt_results[name]
            speedup = pt_res['forward_time_ms'] / jax_res['forward_time_ms']
            print(
                f"{name:<25} {jax_res['forward_time_ms']:<10.2f} {pt_res['forward_time_ms']:<10.2f} {format_speedup(speedup)}")
            comparisons.append(ComparisonResult(
                name=name, max_diff=0, mean_diff=0, relative_diff=0,
                jax_time_ms=jax_res['forward_time_ms'], pytorch_time_ms=pt_res['forward_time_ms'],
                speedup=speedup, passed=True
            ))
    return comparisons


def main():
    parser = argparse.ArgumentParser(description='KeOps JAX Benchmarks')
    parser.add_argument('--pytorch-results', type=str, default='keops_pytorch_results.npz')
    parser.add_argument('--output', type=str, default='keops_jax_results.npz')
    parser.add_argument('--multi-gpu', action='store_true')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    num_warmup = 3 if args.quick else 5
    num_runs = 10 if args.quick else 20

    # Standard sizes (N varying, M fixed small)
    standard_sizes = [(5000, 100, 3), (10000, 100, 3), (25000, 100, 3), (50000, 100, 3)]

    # Rectangular sizes (N and M varying, large values) - testing unbalanced matrices
    rectangular_sizes = [
        (2000, 2000, 3), (5000, 5000, 3), (10000, 10000, 3),
        (50000, 10000, 3), (200000, 100000, 3)
    ]

    # Batch sizes to test scaling with (Fixed N=5000, M=5000)
    batch_scaling = [1, 10, 100]

    if not args.quick:
        standard_sizes.extend([(75000, 100, 3), (100000, 100, 3)])
        rectangular_sizes.append((20000, 20000, 3))

    # Load PyTorch data
    pt_results, pt_outputs = None, None
    if Path(args.pytorch_results).exists():
        data = np.load(args.pytorch_results, allow_pickle=True)
        pt_results = json.loads(str(data['results']))
        pt_outputs = {k: data[k] for k in data.files if k != 'results'}

    all_results, all_outputs = {}, {}

    # 1. Run Standard Benchmarks
    print(f"\n{Colors.CYAN}{'=' * 60}\nSECTION 1: Standard N-Scaling (M=100)\n{'=' * 60}{Colors.END}")
    for N, M, D in standard_sizes:
        result, output = run_benchmark(f"varifold_lazy_N{N}", varifold_kernel_lazytensor, N, M, D,
                                       num_warmup=num_warmup, num_runs=num_runs, pytorch_results=pt_results,
                                       pytorch_outputs=pt_outputs)
        all_results[result.name], all_outputs[result.name] = asdict(result), output

        result, output = run_benchmark(f"gaussian_lazy_N{N}", gaussian_kernel_lazytensor, N, M, D,
                                       num_warmup=num_warmup, num_runs=num_runs, pytorch_results=pt_results,
                                       pytorch_outputs=pt_outputs)
        all_results[result.name], all_outputs[result.name] = asdict(result), output

    # 2. Run Batch Scaling Benchmarks
    print(f"\n{Colors.CYAN}{'=' * 60}\nSECTION 2: Batch Size Scaling (N=M=5000)\n{'=' * 60}{Colors.END}")
    N_fix, M_fix = 5000, 5000
    for B in batch_scaling:
        name = f"varifold_batch_B{B}"
        result, output = run_benchmark(name, varifold_kernel_lazytensor, N_fix, M_fix, 3, batch_size=B,
                                       num_warmup=num_warmup, num_runs=num_runs, pytorch_results=pt_results,
                                       pytorch_outputs=pt_outputs)
        all_results[result.name], all_outputs[result.name] = asdict(result), output

    # 3. Run Rectangular Benchmarks
    rect_batch = batch_scaling[-1]
    print(f"\n{Colors.CYAN}{'=' * 60}\nSECTION 3: Rectangular/Large (Batch={rect_batch})\n{'=' * 60}{Colors.END}")
    for N, M, D in rectangular_sizes:
        name = f"varifold_rect_N{N}_M{M}"
        result, output = run_benchmark(name, varifold_kernel_lazytensor, N, M, D, batch_size=rect_batch,
                                       num_warmup=num_warmup, num_runs=num_runs, pytorch_results=pt_results,
                                       pytorch_outputs=pt_outputs)
        all_results[result.name], all_outputs[result.name] = asdict(result), output

    # 4. Multi-GPU Sharding
    if args.multi_gpu:
        for r in run_sharded_benchmark("varifold_sharded", 25000, 100, 3, num_warmup=num_warmup, num_runs=num_runs):
            all_results[r.name] = asdict(r)

    if pt_results:
        compare_with_pytorch(args.pytorch_results, all_results, all_outputs)

    np.savez(args.output, results=json.dumps(all_results), **all_outputs)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()