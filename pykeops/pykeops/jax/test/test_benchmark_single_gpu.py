#!/usr/bin/env python3
"""
KeOps JAX Single GPU Benchmark
==============================
Benchmark JAX KeOps vs PyTorch KeOps on a single GPU.

Measures:
- Forward pass performance
- Backward pass (gradient) performance
- Various problem sizes
- Multiple formula types
"""

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


# Benchmark configurations
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

    # High dimensional
    BenchmarkConfig("HighD/Euclidean", 10000, 10000, 16, "SqNorm2(x-y)", "Sum"),
    BenchmarkConfig("HighD/Gaussian", 10000, 10000, 16, "Exp(-SqNorm2(x-y)*s)", "Sum", True),
]

# Timing parameters
N_WARMUP = 10
N_ITER = 100


# =============================================================================
# Benchmark Results
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    config_name: str
    nx: int
    ny: int
    dim: int
    jax_median_ms: float
    jax_mean_ms: float
    jax_std_ms: float
    jax_min_ms: float
    torch_median_ms: Optional[float]
    torch_mean_ms: Optional[float]
    torch_std_ms: Optional[float]
    torch_min_ms: Optional[float]
    speedup: Optional[float]
    pass_type: str  # 'forward' or 'backward'


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_jax_forward(config: BenchmarkConfig, n_warmup: int, n_iter: int, max_retries: int = 3) -> dict:
    """Benchmark JAX forward pass with retry logic for cache race conditions."""
    import time as time_module

    last_error = None
    for retry in range(max_retries):
        try:
            # Create data
            key = jax.random.PRNGKey(42)
            k1, k2 = jax.random.split(key)

            x = jax.random.normal(k1, (config.nx, config.dim), dtype=jnp.float32)
            y = jax.random.normal(k2, (config.ny, config.dim), dtype=jnp.float32)

            # Create operator
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

                # Benchmark (matching old script timing methodology)
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

                # Benchmark (matching old script timing methodology)
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
            }

        except FileNotFoundError as e:
            # KeOps cache race condition - retry after a short delay
            last_error = e
            if retry < max_retries - 1:
                time_module.sleep(0.5)  # Wait before retrying
                continue
            raise
        except Exception as e:
            raise

    # Should not reach here, but just in case
    if last_error:
        raise last_error


def benchmark_jax_backward(config: BenchmarkConfig, n_warmup: int, n_iter: int, max_retries: int = 3) -> dict:
    """Benchmark JAX backward pass with retry logic for cache race conditions."""
    import time as time_module

    last_error = None
    for retry in range(max_retries):
        try:
            key = jax.random.PRNGKey(42)
            k1, k2 = jax.random.split(key)

            x = jax.random.normal(k1, (config.nx, config.dim), dtype=jnp.float32)
            y = jax.random.normal(k2, (config.ny, config.dim), dtype=jnp.float32)

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
                prev_result = _  # From warmup
                for _ in range(n_iter):
                    jax.block_until_ready(prev_result)  # Ensure previous iteration complete
                    start = time.perf_counter()
                    result = compute_grad(x, y, s)
                    jax.block_until_ready(result)
                    times.append((time.perf_counter() - start) * 1000)
                    prev_result = result
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
                prev_result = _  # From warmup
                for _ in range(n_iter):
                    jax.block_until_ready(prev_result)  # Ensure previous iteration complete
                    start = time.perf_counter()
                    result = compute_grad(x, y)
                    jax.block_until_ready(result)
                    times.append((time.perf_counter() - start) * 1000)
                    prev_result = result

            times = np.array(times)
            return {
                'median': float(np.median(times)),
                'mean': float(np.mean(times)),
                'std': float(np.std(times)),
                'min': float(np.min(times)),
            }

        except FileNotFoundError as e:
            # KeOps cache race condition - retry after a short delay
            last_error = e
            if retry < max_retries - 1:
                time_module.sleep(0.5)
                continue
            raise
        except Exception as e:
            raise

    if last_error:
        raise last_error


def benchmark_torch_forward(config: BenchmarkConfig, n_warmup: int, n_iter: int) -> dict:
    """Benchmark PyTorch forward pass."""
    if not TORCH_AVAILABLE:
        return None

    device = 'cuda'

    x = torch.randn(config.nx, config.dim, dtype=torch.float32, device=device)
    y = torch.randn(config.ny, config.dim, dtype=torch.float32, device=device)

    if config.has_param:
        aliases = [f"x=Vi({config.dim})", f"y=Vj({config.dim})", "s=Pm(1)"]
        s = torch.tensor([0.5], dtype=torch.float32, device=device)
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
        'median': float(np.median(times)),
        'mean': float(np.mean(times)),
        'std': float(np.std(times)),
        'min': float(np.min(times)),
    }


def benchmark_torch_backward(config: BenchmarkConfig, n_warmup: int, n_iter: int) -> dict:
    """Benchmark PyTorch backward pass."""
    if not TORCH_AVAILABLE:
        return None

    device = 'cuda'

    if config.has_param:
        aliases = [f"x=Vi({config.dim})", f"y=Vj({config.dim})", "s=Pm(1)"]
        s = torch.tensor([0.5], dtype=torch.float32, device=device)
        op = Genred_torch(config.formula, aliases, reduction_op=config.reduction, axis=1)

        def run_backward():
            x = torch.randn(config.nx, config.dim, dtype=torch.float32, device=device, requires_grad=True)
            y = torch.randn(config.ny, config.dim, dtype=torch.float32, device=device)
            result = op(x, y, s)
            loss = result.sum()
            loss.backward()
            return x.grad

        # Warmup
        for _ in range(n_warmup):
            _ = run_backward()
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(n_iter):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = run_backward()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    else:
        aliases = [f"x=Vi({config.dim})", f"y=Vj({config.dim})"]
        op = Genred_torch(config.formula, aliases, reduction_op=config.reduction, axis=1)

        def run_backward():
            x = torch.randn(config.nx, config.dim, dtype=torch.float32, device=device, requires_grad=True)
            y = torch.randn(config.ny, config.dim, dtype=torch.float32, device=device)
            result = op(x, y)
            loss = result.sum()
            loss.backward()
            return x.grad

        # Warmup
        for _ in range(n_warmup):
            _ = run_backward()
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(n_iter):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = run_backward()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    return {
        'median': float(np.median(times)),
        'mean': float(np.mean(times)),
        'std': float(np.std(times)),
        'min': float(np.min(times)),
    }


# =============================================================================
# Main Runner
# =============================================================================

def run_benchmarks(save_results: bool = True):
    """Run all benchmarks."""
    if not JAX_AVAILABLE:
        print(f"{Colors.RED}KeOps JAX not available. Cannot run benchmarks.{Colors.RESET}")
        return False

    print_header("KeOps JAX vs PyTorch Single-GPU Benchmark")

    # Print configuration
    print(f"  JAX version: {jax.__version__}")
    if TORCH_AVAILABLE:
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  {Colors.YELLOW}PyTorch not available - JAX only{Colors.RESET}")
    print(f"  Warmup iterations: {N_WARMUP}")
    print(f"  Benchmark iterations: {N_ITER}")
    print()

    results: List[BenchmarkResult] = []

    # =========================
    # Forward Pass Benchmarks
    # =========================
    print_subheader("Forward Pass Benchmarks")

    table = ASCIITable([
        TableColumn("Problem", 20),
        TableColumn("Size", 15),
        TableColumn("JAX (ms)", 12, 'right'),
        TableColumn("PyTorch (ms)", 12, 'right'),
        TableColumn("Speedup", 12, 'right'),
    ], title="Forward Pass: JAX vs PyTorch")

    for config in BENCHMARK_CONFIGS:
        print(f"  Running {config.name}...", end='', flush=True)

        try:
            jax_result = benchmark_jax_forward(config, N_WARMUP, N_ITER)
            torch_result = benchmark_torch_forward(config, N_WARMUP, N_ITER)

            if torch_result:
                speedup = torch_result['median'] / jax_result['median']
                speedup_str = format_speedup(speedup)
                torch_str = f"{torch_result['median']:.2f}"
            else:
                speedup = None
                speedup_str = "-"
                torch_str = "-"

            jax_str = color_speed(jax_result['median'], torch_result['median'] if torch_result else None)

            table.add_row([
                config.name,
                f"{config.nx}x{config.ny}x{config.dim}",
                f"{jax_result['median']:.2f}",
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
                torch_median_ms=torch_result['median'] if torch_result else None,
                torch_mean_ms=torch_result['mean'] if torch_result else None,
                torch_std_ms=torch_result['std'] if torch_result else None,
                torch_min_ms=torch_result['min'] if torch_result else None,
                speedup=speedup,
                pass_type='forward',
            ))

            print(f" done ({jax_result['median']:.2f}ms)")

        except Exception as e:
            print(f" {Colors.RED}ERROR: {e}{Colors.RESET}")
            table.add_row([config.name, f"{config.nx}x{config.ny}", "ERROR", "-", "-"])

    table.print()

    # =========================
    # Backward Pass Benchmarks
    # =========================
    print_subheader("Backward Pass Benchmarks")

    table = ASCIITable([
        TableColumn("Problem", 20),
        TableColumn("Size", 15),
        TableColumn("JAX (ms)", 12, 'right'),
        TableColumn("PyTorch (ms)", 12, 'right'),
        TableColumn("Speedup", 12, 'right'),
    ], title="Backward Pass: JAX vs PyTorch")

    # Use subset of configs for backward (gradient computation is slower)
    backward_configs = [c for c in BENCHMARK_CONFIGS if "XLarge" not in c.name]

    for config in backward_configs:
        print(f"  Running {config.name}...", end='', flush=True)

        try:
            jax_result = benchmark_jax_backward(config, N_WARMUP, N_ITER)
            torch_result = benchmark_torch_backward(config, N_WARMUP, N_ITER)

            if torch_result:
                speedup = torch_result['median'] / jax_result['median']
                speedup_str = format_speedup(speedup)
                torch_str = f"{torch_result['median']:.2f}"
            else:
                speedup = None
                speedup_str = "-"
                torch_str = "-"

            table.add_row([
                config.name,
                f"{config.nx}x{config.ny}x{config.dim}",
                f"{jax_result['median']:.2f}",
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
                torch_median_ms=torch_result['median'] if torch_result else None,
                torch_mean_ms=torch_result['mean'] if torch_result else None,
                torch_std_ms=torch_result['std'] if torch_result else None,
                torch_min_ms=torch_result['min'] if torch_result else None,
                speedup=speedup,
                pass_type='backward',
            ))

            print(f" done ({jax_result['median']:.2f}ms)")

        except Exception as e:
            print(f" {Colors.RED}ERROR: {e}{Colors.RESET}")
            table.add_row([config.name, f"{config.nx}x{config.ny}", "ERROR", "-", "-"])

    table.print()

    # =========================
    # Summary
    # =========================
    print_subheader("Summary")

    forward_results = [r for r in results if r.pass_type == 'forward' and r.speedup is not None]
    backward_results = [r for r in results if r.pass_type == 'backward' and r.speedup is not None]

    if forward_results:
        avg_fwd_speedup = np.mean([r.speedup for r in forward_results])
        min_fwd_speedup = min(r.speedup for r in forward_results)
        max_fwd_speedup = max(r.speedup for r in forward_results)
        print(f"  Forward pass:")
        print(f"    Average speedup: {format_speedup(avg_fwd_speedup)}")
        print(f"    Range: {min_fwd_speedup:.2f}x - {max_fwd_speedup:.2f}x")

    if backward_results:
        avg_bwd_speedup = np.mean([r.speedup for r in backward_results])
        min_bwd_speedup = min(r.speedup for r in backward_results)
        max_bwd_speedup = max(r.speedup for r in backward_results)
        print(f"  Backward pass:")
        print(f"    Average speedup: {format_speedup(avg_bwd_speedup)}")
        print(f"    Range: {min_bwd_speedup:.2f}x - {max_bwd_speedup:.2f}x")

    print()

    # Color coding legend
    print(f"  {Colors.DIM}Legend: speedup > 1 means JAX is faster{Colors.RESET}")
    print(
        f"  {Colors.GREEN}Green{Colors.RESET} = JAX faster, {Colors.RED}Red{Colors.RESET} = PyTorch faster, {Colors.YELLOW}Yellow{Colors.RESET} = similar")

    # Save results
    if save_results:
        output_file = f"benchmark_single_gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\n  Results saved to: {output_file}")

    print()
    return True


if __name__ == '__main__':
    success = run_benchmarks()
    sys.exit(0 if success else 1)