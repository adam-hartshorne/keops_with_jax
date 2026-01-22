#!/usr/bin/env python3
"""
KeOps PyTorch Reference Tests & Benchmarks

Tests KeOps operations using PyTorch and saves results for comparison with JAX.
"""

import argparse
import time
import json
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import torch
from pykeops.torch import LazyTensor


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


def create_test_data(N: int, M: int, D: int, batch_size: int = 1, dtype=torch.float32, device='cuda:0', seed=42) -> \
Dict[str, torch.Tensor]:
    np.random.seed(seed)

    # Generate numpy data
    if batch_size > 1:
        x_np = np.random.randn(batch_size, N, D).astype(np.float32)
        y_np = np.random.randn(batch_size, M, D).astype(np.float32)
        u_np = np.random.randn(batch_size, N, D).astype(np.float32)
        v_np = np.random.randn(batch_size, M, D).astype(np.float32)
        u_np /= np.linalg.norm(u_np, axis=-1, keepdims=True)
        v_np /= np.linalg.norm(v_np, axis=-1, keepdims=True)
    else:
        x_np = np.random.randn(N, D).astype(np.float32)
        y_np = np.random.randn(M, D).astype(np.float32)
        u_np = np.random.randn(N, D).astype(np.float32)
        v_np = np.random.randn(M, D).astype(np.float32)
        u_np /= np.linalg.norm(u_np, axis=-1, keepdims=True)
        v_np /= np.linalg.norm(v_np, axis=-1, keepdims=True)

    # To Torch
    to_dev = lambda x: torch.from_numpy(x).to(device=device, dtype=dtype).requires_grad_(True)

    sigma_spatial, sigma_normal = 0.5, 1.0
    gamma_spatial = torch.tensor(1.0 / (2 * sigma_spatial ** 2), dtype=dtype, device=device)
    gamma_normal = torch.tensor(1.0 / (sigma_normal ** 2), dtype=dtype, device=device)

    return {
        'x': to_dev(x_np), 'y': to_dev(y_np), 'u': to_dev(u_np), 'v': to_dev(v_np),
        'gamma_spatial': gamma_spatial, 'gamma_normal': gamma_normal
    }


def varifold_kernel_lazytensor(xx, yy, uu, vv, gamma, gamma_1):
    if xx.dim() == 3:
        x, y = LazyTensor(xx[:, :, None, :]), LazyTensor(yy[:, None, :, :])
        u, v = LazyTensor(uu[:, :, None, :]), LazyTensor(vv[:, None, :, :])
    else:
        x, y = LazyTensor(xx[:, None, :]), LazyTensor(yy[None, :, :])
        u, v = LazyTensor(uu[:, None, :]), LazyTensor(vv[None, :, :])

    D2 = x.sqdist(y)
    ss = (u * v).sum(-1)
    return ((-D2 * gamma).exp() * (ss * gamma_1).exp()).sum(axis=1)


def gaussian_kernel_lazytensor(x, y, gamma):
    if x.dim() == 3:
        x_i, y_j = LazyTensor(x[:, :, None, :]), LazyTensor(y[:, None, :, :])
    else:
        x_i, y_j = LazyTensor(x[:, None, :]), LazyTensor(y[None, :, :])
    return ((-x_i.sqdist(y_j) * gamma).exp()).sum(axis=1)


def benchmark_forward(kernel_fn, inputs, num_warmup=5, num_runs=20):
    device = next(iter(inputs.values())).device

    # Warmup
    for _ in range(num_warmup):
        _ = kernel_fn(**inputs)
        if device.type == 'cuda': torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_runs):
        if device.type == 'cuda': torch.cuda.synchronize()  # Barrier before start
        start = time.perf_counter()
        result = kernel_fn(**inputs)
        if device.type == 'cuda': torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return result, np.mean(times), np.std(times)


def benchmark_gradient(kernel_fn, inputs, num_warmup=5, num_runs=20):
    device = next(iter(inputs.values())).device
    grad_keys = [k for k in inputs.keys() if k in ['x', 'y', 'u', 'v', 'xx', 'yy', 'uu', 'vv']]

    def zero_grads():
        for k in grad_keys:
            if inputs[k].grad is not None: inputs[k].grad.zero_()

    for _ in range(num_warmup):
        zero_grads()
        kernel_fn(**inputs).sum().backward()
        if device.type == 'cuda': torch.cuda.synchronize()

    times, grad_norm = [], 0.0
    for _ in range(num_runs):
        zero_grads()
        if device.type == 'cuda': torch.cuda.synchronize()  # Barrier before start
        start = time.perf_counter()
        kernel_fn(**inputs).sum().backward()
        if device.type == 'cuda': torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

        if len(grad_keys) > 0 and inputs[grad_keys[0]].grad is not None:
            grad_norm = float(inputs[grad_keys[0]].grad.norm())

    return np.mean(times), np.std(times), grad_norm


def run_benchmark(name, kernel_fn, N, M, D, batch_size=1, num_warmup=5, num_runs=20, device='cuda:0'):
    print(f"\n{'-' * 60}\nBenchmark: {name} (N={N}, M={M}, B={batch_size})\n{'-' * 60}")
    data = create_test_data(N, M, D, batch_size, device=device)

    if 'varifold' in name.lower():
        inputs = {'xx': data['x'], 'yy': data['y'], 'uu': data['u'], 'vv': data['v'], 'gamma': data['gamma_spatial'],
                  'gamma_1': data['gamma_normal']}
    else:
        inputs = {'x': data['x'], 'y': data['y'], 'gamma': data['gamma_spatial']}

    result, fwd_ms, fwd_std = benchmark_forward(kernel_fn, inputs, num_warmup, num_runs)
    print(f"  Forward: {fwd_ms:.3f} ± {fwd_std:.3f} ms")

    grad_ms, grad_std, grad_norm = benchmark_gradient(kernel_fn, inputs, num_warmup, num_runs)
    print(f"  Gradient: {grad_ms:.3f} ± {grad_std:.3f} ms")

    return BenchmarkResult(
        name=name, forward_time_ms=fwd_ms, forward_time_std_ms=fwd_std,
        grad_time_ms=grad_ms, grad_time_std_ms=grad_std,
        output_shape=result.shape, output_mean=float(result.mean()), output_std=float(result.std()),
        grad_norm=grad_norm, num_runs=num_runs, device=str(device), dtype='float32'
    ), result.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='keops_pytorch_results.npz')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_warmup = 3 if args.quick else 5
    num_runs = 10 if args.quick else 20

    # 1. Standard Sizes
    standard_sizes = [(5000, 100, 3), (10000, 100, 3), (25000, 100, 3), (50000, 100, 3)]

    # 2. Rectangular Sizes
    rectangular_sizes = [
        (2000, 2000, 3), (5000, 5000, 3), (10000, 10000, 3),
        (50000, 10000, 3), (200000, 100000, 3)
    ]

    # 3. Batch Scaling
    batch_scaling = [1, 10, 100]

    if not args.quick:
        standard_sizes.extend([(75000, 100, 3), (100000, 100, 3)])
        rectangular_sizes.append((20000, 20000, 3))

    all_results, all_outputs = {}, {}

    # Run Standard
    for N, M, D in standard_sizes:
        res, out = run_benchmark(f"varifold_lazy_N{N}", varifold_kernel_lazytensor, N, M, D, num_warmup=num_warmup,
                                 num_runs=num_runs, device=device)
        all_results[res.name], all_outputs[res.name] = asdict(res), out

        res, out = run_benchmark(f"gaussian_lazy_N{N}", gaussian_kernel_lazytensor, N, M, D, num_warmup=num_warmup,
                                 num_runs=num_runs, device=device)
        all_results[res.name], all_outputs[res.name] = asdict(res), out

    # Run Batch Scaling
    N_fix, M_fix = 5000, 5000
    for B in batch_scaling:
        name = f"varifold_batch_B{B}"
        res, out = run_benchmark(name, varifold_kernel_lazytensor, N_fix, M_fix, 3, batch_size=B, num_warmup=num_warmup,
                                 num_runs=num_runs, device=device)
        all_results[res.name], all_outputs[res.name] = asdict(res), out

    # Run Rectangular
    rect_batch = batch_scaling[-1]
    for N, M, D in rectangular_sizes:
        name = f"varifold_rect_N{N}_M{M}"
        res, out = run_benchmark(name, varifold_kernel_lazytensor, N, M, D, batch_size=rect_batch,
                                 num_warmup=num_warmup, num_runs=num_runs, device=device)
        all_results[res.name], all_outputs[res.name] = asdict(res), out

    np.savez(args.output, results=json.dumps(all_results), **all_outputs)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()