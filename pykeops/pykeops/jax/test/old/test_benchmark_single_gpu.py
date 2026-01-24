#!/usr/bin/env python3
"""
KeOps JAX Single GPU Benchmark
==============================
Benchmark JAX KeOps vs PyTorch KeOps on a single GPU.

Updated to match the settings from benchmark_jax_keops.py and benchmark_pytorch_keops.py:
- Explicit @jax.jit wrapping
- 100 iterations for stable statistics
- 10 warmup iterations for JAX, 5 for PyTorch
- Uses MEDIAN for JAX (accounts for async variance)
- Uses MEAN for PyTorch (more stable)
"""

import os

os.environ['JAX_KEOPS_DEBUG'] = '0'

import sys
import time
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional

import jax
import jax.numpy as jnp

# Import test utilities
from test_utils import (
    Colors, print_header, print_subheader,
    ASCIITable, TableColumn, format_speedup, color_speed
)

# Import KeOps JAX
try:
    from pykeops.jax import Genred as Genred_jax

    JAX_AVAILABLE = True
except ImportError as e:
    print(f"{Colors.RED}Error: pykeops.jax not found: {e}{Colors.RESET}")
    JAX_AVAILABLE = False

# Import PyTorch
try:
    import torch
    from pykeops.torch import Genred as Genred_torch

    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str
    nx: int
    ny: int
    dim: int
    formula: str
    reduction: str
    has_param: bool = False


# Benchmark configurations - matching the user's test scripts
BENCHMARK_CONFIGS = [
    # Small problems
    BenchmarkConfig("Small/Euclidean", 1000, 1000, 3, "SqNorm2(x-y)", "Sum"),
    BenchmarkConfig("Small/Gaussian", 1000, 1000, 3, "Exp(-SqNorm2(x-y)*s)", "Sum", True),

    # Medium problems
    BenchmarkConfig("Medium/Euclidean", 10000, 10000, 3, "SqNorm2(x-y)", "Sum"),
    BenchmarkConfig("Medium/Gaussian", 10000, 10000, 3, "Exp(-SqNorm2(x-y)*s)", "Sum", True),

    # Large problems
    BenchmarkConfig("Large/Euclidean", 50000, 20000, 3, "SqNorm2(x-y)", "Sum"),
    BenchmarkConfig("Large/Gaussian", 50000, 20000, 3, "Exp(-SqNorm2(x-y)*s)", "Sum", True),

    # Very large problems
    BenchmarkConfig("XLarge/Euclidean", 100000, 50000, 3, "SqNorm2(x-y)", "Sum"),

    # High-dimensional problems
    BenchmarkConfig("HighD/Euclidean", 10000, 10000, 10, "SqNorm2(x-y)", "Sum"),
    BenchmarkConfig("HighD/Gaussian", 10000, 10000, 10, "Exp(-SqNorm2(x-y)*s)", "Sum", True),
]


@dataclass
class BenchmarkResult:
    """Result from a single benchmark."""
    config_name: str
    nx: int
    ny: int
    dim: int
    jax_median_ms: float
    jax_mean_ms: float
    jax_std_ms: float
    jax_min_ms: float
    torch_mean_ms: Optional[float]
    torch_std_ms: Optional[float]
    torch_min_ms: Optional[float]
    speedup: Optional[float]
    pass_type: str


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_jax_forward(config: BenchmarkConfig, n_warmup: int, n_iter: int,
                          max_retries: int = 3) -> dict:
    """
    Benchmark JAX forward pass with explicit @jax.jit wrapping.
    """
    import time as time_module

    last_error = None
    for retry in range(max_retries):
        try:
            # Create data
            x = jnp.array(np.random.randn(config.nx, config.dim).astype(np.float32))
            y = jnp.array(np.random.randn(config.ny, config.dim).astype(np.float32))

            # Create operator with explicit @jax.jit
            if config.has_param:
                aliases = [f"x=Vi({config.dim})", f"y=Vj({config.dim})", "s=Pm(1)"]
                s = jnp.array([0.5], dtype=jnp.float32)
                op = Genred_jax(config.formula, aliases, reduction_op=config.reduction, axis=1)

                @jax.jit
                def compute(x, y, s):
                    return op(x, y, s)

                # Trigger compilation
                _ = compute(x, y, s)
                jax.block_until_ready(_)

                # Warmup
                for _ in range(n_warmup):
                    _ = compute(x, y, s)
                    jax.block_until_ready(_)

                # Benchmark
                times = []
                for _ in range(n_iter):
                    start = time.perf_counter()
                    result = compute(x, y, s)
                    jax.block_until_ready(result)
                    times.append((time.perf_counter() - start) * 1000)
            else:
                aliases = [f"x=Vi({config.dim})", f"y=Vj({config.dim})"]
                op = Genred_jax(config.formula, aliases, reduction_op=config.reduction, axis=1)

                @jax.jit
                def compute(x, y):
                    return op(x, y)

                # Trigger compilation
                _ = compute(x, y)
                jax.block_until_ready(_)

                # Warmup
                for _ in range(n_warmup):
                    _ = compute(x, y)
                    jax.block_until_ready(_)

                # Benchmark
                times = []
                for _ in range(n_iter):
                    start = time.perf_counter()
                    result = compute(x, y)
                    jax.block_until_ready(result)
                    times.append((time.perf_counter() - start) * 1000)

            times = np.array(times)
            return {
                'median': float(np.median(times)),
                'mean': float(np.mean(times)),
                'std': float(np.std(times)),
                'min': float(np.min(times)),
                'p10': float(np.percentile(times, 10)),
                'p90': float(np.percentile(times, 90)),
            }

        except FileNotFoundError as e:
            last_error = e
            if retry < max_retries - 1:
                time_module.sleep(0.5)
                continue
            raise
        except Exception as e:
            raise

    if last_error:
        raise last_error


def benchmark_torch_forward(config: BenchmarkConfig, n_warmup: int, n_iter: int) -> Optional[dict]:
    """Benchmark PyTorch forward pass."""
    if not TORCH_AVAILABLE:
        return None

    x = torch.randn(config.nx, config.dim, device='cuda', dtype=torch.float32)
    y = torch.randn(config.ny, config.dim, device='cuda', dtype=torch.float32)

    if config.has_param:
        aliases = [f"x=Vi({config.dim})", f"y=Vj({config.dim})", "s=Pm(1)"]
        s = torch.tensor([0.5], device='cuda', dtype=torch.float32)
        op = Genred_torch(config.formula, aliases, reduction_op=config.reduction, axis=1)

        # Warmup
        for _ in range(n_warmup):
            _ = op(x, y, s)
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(n_iter):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = op(x, y, s)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    else:
        aliases = [f"x=Vi({config.dim})", f"y=Vj({config.dim})"]
        op = Genred_torch(config.formula, aliases, reduction_op=config.reduction, axis=1)

        # Warmup
        for _ in range(n_warmup):
            _ = op(x, y)
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(n_iter):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = op(x, y)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    return {
        'mean': float(np.mean(times)),
        'std': float(np.std(times)),
        'min': float(np.min(times)),
        'median': float(np.median(times)),
    }


def benchmark_jax_backward(config: BenchmarkConfig, n_warmup: int, n_iter: int,
                           max_retries: int = 3) -> dict:
    """Benchmark JAX backward pass with explicit @jax.jit wrapping."""
    import time as time_module

    last_error = None
    for retry in range(max_retries):
        try:
            x = jnp.array(np.random.randn(config.nx, config.dim).astype(np.float32))
            y = jnp.array(np.random.randn(config.ny, config.dim).astype(np.float32))

            if config.has_param:
                aliases = [f"x=Vi({config.dim})", f"y=Vj({config.dim})", "s=Pm(1)"]
                s = jnp.array([0.5], dtype=jnp.float32)
                op = Genred_jax(config.formula, aliases, reduction_op=config.reduction, axis=1)

                @jax.jit
                def compute_grad(x, y, s):
                    def loss(x):
                        return jnp.sum(op(x, y, s))

                    return jax.grad(loss)(x)

                # Trigger compilation
                _ = compute_grad(x, y, s)
                jax.block_until_ready(_)

                # Warmup
                for _ in range(n_warmup):
                    _ = compute_grad(x, y, s)
                    jax.block_until_ready(_)

                # Benchmark
                times = []
                for _ in range(n_iter):
                    start = time.perf_counter()
                    result = compute_grad(x, y, s)
                    jax.block_until_ready(result)
                    times.append((time.perf_counter() - start) * 1000)
            else:
                aliases = [f"x=Vi({config.dim})", f"y=Vj({config.dim})"]
                op = Genred_jax(config.formula, aliases, reduction_op=config.reduction, axis=1)

                @jax.jit
                def compute_grad(x, y):
                    def loss(x):
                        return jnp.sum(op(x, y))

                    return jax.grad(loss)(x)

                # Trigger compilation
                _ = compute_grad(x, y)
                jax.block_until_ready(_)

                # Warmup
                for _ in range(n_warmup):
                    _ = compute_grad(x, y)
                    jax.block_until_ready(_)

                # Benchmark
                times = []
                for _ in range(n_iter):
                    start = time.perf_counter()
                    result = compute_grad(x, y)
                    jax.block_until_ready(result)
                    times.append((time.perf_counter() - start) * 1000)

            times = np.array(times)
            return {
                'median': float(np.median(times)),
                'mean': float(np.mean(times)),
                'std': float(np.std(times)),
                'min': float(np.min(times)),
            }

        except FileNotFoundError as e:
            last_error = e
            if retry < max_retries - 1:
                time_module.sleep(0.5)
                continue
            raise
        except Exception as e:
            raise

    if last_error:
        raise last_error


def benchmark_torch_backward(config: BenchmarkConfig, n_warmup: int, n_iter: int) -> Optional[dict]:
    """Benchmark PyTorch backward pass."""
    if not TORCH_AVAILABLE:
        return None

    x = torch.randn(config.nx, config.dim, device='cuda', dtype=torch.float32, requires_grad=True)
    y = torch.randn(config.ny, config.dim, device='cuda', dtype=torch.float32)

    if config.has_param:
        aliases = [f"x=Vi({config.dim})", f"y=Vj({config.dim})", "s=Pm(1)"]
        s = torch.tensor([0.5], device='cuda', dtype=torch.float32)
        op = Genred_torch(config.formula, aliases, reduction_op=config.reduction, axis=1)

        # Warmup
        for _ in range(n_warmup):
            if x.grad is not None:
                x.grad.zero_()
            out = op(x, y, s)
            loss = out.sum()
            loss.backward()
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(n_iter):
            if x.grad is not None:
                x.grad.zero_()
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = op(x, y, s)
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    else:
        aliases = [f"x=Vi({config.dim})", f"y=Vj({config.dim})"]
        op = Genred_torch(config.formula, aliases, reduction_op=config.reduction, axis=1)

        # Warmup
        for _ in range(n_warmup):
            if x.grad is not None:
                x.grad.zero_()
            out = op(x, y)
            loss = out.sum()
            loss.backward()
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(n_iter):
            if x.grad is not None:
                x.grad.zero_()
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = op(x, y)
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    return {
        'mean': float(np.mean(times)),
        'std': float(np.std(times)),
        'min': float(np.min(times)),
        'median': float(np.median(times)),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    if not JAX_AVAILABLE:
        print(f"{Colors.RED}JAX not available. Exiting.{Colors.RESET}")
        return

    # Configuration - matching user's benchmark scripts
    N_WARMUP_JAX = 50  # JAX needs more warmup
    N_WARMUP_TORCH = 50  # PyTorch needs less
    N_ITER = 100  # More iterations for stable statistics

    print_header("KeOps JAX vs PyTorch Single-GPU Benchmark")

    print(f"  JAX version: {jax.__version__}")
    if TORCH_AVAILABLE:
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  JAX warmup: {N_WARMUP_JAX}, PyTorch warmup: {N_WARMUP_TORCH}")
    print(f"  Benchmark iterations: {N_ITER}")
    print()
    print(f"  {Colors.CYAN}Note: Using MEDIAN for JAX (accounts for async variance)")
    print(f"        Using MEAN for PyTorch (more stable){Colors.RESET}")
    print()

    results: List[BenchmarkResult] = []

    # =========================
    # Forward Pass Benchmarks
    # =========================
    print_subheader("Forward Pass Benchmarks")

    table = ASCIITable([
        TableColumn("Problem", 20),
        TableColumn("Size", 15),
        TableColumn("JAX med (ms)", 12, 'right'),
        TableColumn("PyTorch (ms)", 12, 'right'),
        TableColumn("Speedup", 12, 'right'),
    ], title="Forward Pass: JAX (median) vs PyTorch (mean)")

    for config in BENCHMARK_CONFIGS:
        print(f"  Running {config.name}...", end='', flush=True)

        try:
            jax_result = benchmark_jax_forward(config, N_WARMUP_JAX, N_ITER)
            torch_result = benchmark_torch_forward(config, N_WARMUP_TORCH, N_ITER)

            if torch_result:
                # Compare JAX median to PyTorch mean (fairer comparison)
                speedup = torch_result['mean'] / jax_result['median']
                speedup_str = format_speedup(speedup)
                torch_str = f"{torch_result['mean']:.3f}"
            else:
                speedup = None
                speedup_str = "-"
                torch_str = "-"

            table.add_row([
                config.name,
                f"{config.nx}x{config.ny}x{config.dim}",
                f"{jax_result['median']:.3f}",
                torch_str,
                speedup_str,
            ])

            results.append(BenchmarkResult(
                config_name=config.name,
                nx=config.nx,
                ny=config.ny,
                dim=config.dim,
                jax_median_ms=jax_result['median'],
                jax_mean_ms=jax_result['mean'],
                jax_std_ms=jax_result['std'],
                jax_min_ms=jax_result['min'],
                torch_mean_ms=torch_result['mean'] if torch_result else None,
                torch_std_ms=torch_result['std'] if torch_result else None,
                torch_min_ms=torch_result['min'] if torch_result else None,
                speedup=speedup,
                pass_type='forward',
            ))

            print(f" done (JAX: {jax_result['median']:.3f}ms, PyTorch: {torch_result['mean']:.3f}ms)")

        except Exception as e:
            print(f" {Colors.RED}ERROR: {e}{Colors.RESET}")
            import traceback
            traceback.print_exc()
            table.add_row([config.name, f"{config.nx}x{config.ny}", "ERROR", "-", "-"])

    table.print()

    # =========================
    # Backward Pass Benchmarks
    # =========================
    print_subheader("Backward Pass Benchmarks")

    table = ASCIITable([
        TableColumn("Problem", 20),
        TableColumn("Size", 15),
        TableColumn("JAX med (ms)", 12, 'right'),
        TableColumn("PyTorch (ms)", 12, 'right'),
        TableColumn("Speedup", 12, 'right'),
    ], title="Backward Pass: JAX (median) vs PyTorch (mean)")

    # Use subset of configs for backward (gradient computation is slower)
    backward_configs = [c for c in BENCHMARK_CONFIGS if "XLarge" not in c.name]

    for config in backward_configs:
        print(f"  Running {config.name}...", end='', flush=True)

        try:
            jax_result = benchmark_jax_backward(config, N_WARMUP_JAX, N_ITER)
            torch_result = benchmark_torch_backward(config, N_WARMUP_TORCH, N_ITER)

            if torch_result:
                speedup = torch_result['mean'] / jax_result['median']
                speedup_str = format_speedup(speedup)
                torch_str = f"{torch_result['mean']:.3f}"
            else:
                speedup = None
                speedup_str = "-"
                torch_str = "-"

            table.add_row([
                config.name,
                f"{config.nx}x{config.ny}x{config.dim}",
                f"{jax_result['median']:.3f}",
                torch_str,
                speedup_str,
            ])

            results.append(BenchmarkResult(
                config_name=config.name,
                nx=config.nx,
                ny=config.ny,
                dim=config.dim,
                jax_median_ms=jax_result['median'],
                jax_mean_ms=jax_result['mean'],
                jax_std_ms=jax_result['std'],
                jax_min_ms=jax_result['min'],
                torch_mean_ms=torch_result['mean'] if torch_result else None,
                torch_std_ms=torch_result['std'] if torch_result else None,
                torch_min_ms=torch_result['min'] if torch_result else None,
                speedup=speedup,
                pass_type='backward',
            ))

            print(f" done (JAX: {jax_result['median']:.3f}ms, PyTorch: {torch_result['mean']:.3f}ms)")

        except Exception as e:
            print(f" {Colors.RED}ERROR: {e}{Colors.RESET}")
            table.add_row([config.name, f"{config.nx}x{config.ny}", "ERROR", "-", "-"])

    table.print()

    # =========================
    # Summary
    # =========================
    print_subheader("Summary")

    forward_results = [r for r in results if r.pass_type == 'forward' and r.speedup]
    backward_results = [r for r in results if r.pass_type == 'backward' and r.speedup]

    if forward_results:
        forward_speedups = [r.speedup for r in forward_results]
        print(f"  Forward pass:")
        print(f"    Average speedup: {format_speedup(np.mean(forward_speedups))}")
        print(f"    Range: {min(forward_speedups):.2f}x - {max(forward_speedups):.2f}x")

        # Count wins
        jax_wins = sum(1 for s in forward_speedups if s > 1.05)
        torch_wins = sum(1 for s in forward_speedups if s < 0.95)
        ties = len(forward_speedups) - jax_wins - torch_wins
        print(f"    JAX wins: {jax_wins}, PyTorch wins: {torch_wins}, Ties: {ties}")

    if backward_results:
        backward_speedups = [r.speedup for r in backward_results]
        print(f"  Backward pass:")
        print(f"    Average speedup: {format_speedup(np.mean(backward_speedups))}")
        print(f"    Range: {min(backward_speedups):.2f}x - {max(backward_speedups):.2f}x")

        jax_wins = sum(1 for s in backward_speedups if s > 1.05)
        torch_wins = sum(1 for s in backward_speedups if s < 0.95)
        ties = len(backward_speedups) - jax_wins - torch_wins
        print(f"    JAX wins: {jax_wins}, PyTorch wins: {torch_wins}, Ties: {ties}")

    print()
    print(f"  Legend: speedup > 1 means JAX is faster")
    print(
        f"  {Colors.GREEN}Green = JAX faster{Colors.RESET}, {Colors.RED}Red = PyTorch faster{Colors.RESET}, {Colors.YELLOW}Yellow = similar{Colors.RESET}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_single_gpu_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n  Results saved to: {filename}")


if __name__ == "__main__":
    main()