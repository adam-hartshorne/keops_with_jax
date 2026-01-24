#!/usr/bin/env python3
"""
KeOps JAX Multi-GPU Benchmark
=============================
Benchmark single GPU vs multi-GPU performance.

Updated with improved accuracy:
- More iterations (50) for stable statistics
- More warmup iterations (10)
- Uses MEDIAN for fair comparison (accounts for async variance)
- Explicit @jax.jit wrapping
- Percentile statistics (P10, P90)
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

# Import test utilities
from test_utils import (
    Colors, print_header, print_subheader,
    ASCIITable, TableColumn, format_speedup, format_efficiency
)

# =============================================================================
# JAX Setup
# =============================================================================

# Try to import JAX with multi-GPU support
try:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
    from jax.experimental import shard_map

    JAX_AVAILABLE = True
    N_DEVICES = len(jax.devices())
except ImportError as e:
    print(f"{Colors.RED}Error: JAX not found: {e}{Colors.RESET}")
    JAX_AVAILABLE = False
    N_DEVICES = 0

# Import KeOps JAX
try:
    from pykeops.jax import Genred

    KEOPS_AVAILABLE = True
except ImportError as e:
    print(f"{Colors.RED}Error: pykeops.jax not found: {e}{Colors.RESET}")
    KEOPS_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU benchmark."""
    name: str
    n: int
    dim: int
    formula: str
    has_param: bool = False


# Problem sizes that benefit from multi-GPU
MULTIGPU_CONFIGS = [
    MultiGPUConfig("Small", 10_000, 3, "SqDist(x, y)"),
    MultiGPUConfig("Medium", 50_000, 3, "SqDist(x, y)"),
    MultiGPUConfig("Large", 100_000, 3, "SqDist(x, y)"),
    MultiGPUConfig("Huge", 500_000, 3, "SqDist(x, y)"),
    MultiGPUConfig("HighD/Medium", 50_000, 16, "SqDist(x, y)"),
    MultiGPUConfig("Gaussian/Medium", 50_000, 3, "Exp(-SqNorm2(x-y)*s)", True),
]

# Updated benchmark parameters for better accuracy
N_WARMUP = 10  # More warmup for JAX async stabilization
N_ITER = 50  # More iterations for stable statistics


# =============================================================================
# Benchmark Results
# =============================================================================

@dataclass
class MultiGPUResult:
    """Result of a multi-GPU benchmark."""
    config_name: str
    n: int
    dim: int
    n_devices: int
    single_gpu_median_ms: float
    single_gpu_mean_ms: float
    single_gpu_min_ms: float
    multi_gpu_median_ms: float
    multi_gpu_mean_ms: float
    multi_gpu_min_ms: float
    speedup: float
    efficiency_pct: float
    pass_type: str


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_single_gpu(config: MultiGPUConfig, n_warmup: int, n_iter: int) -> dict:
    """Benchmark single GPU performance with explicit @jax.jit."""
    devices = jax.devices()
    device = devices[0]

    # Create data on single device
    x = jax.device_put(
        jnp.array(np.random.randn(config.n, config.dim).astype(np.float32)),
        device
    )
    y = jax.device_put(
        jnp.array(np.random.randn(config.n, config.dim).astype(np.float32)),
        device
    )

    # Create operator with explicit @jax.jit
    if config.has_param:
        aliases = [f"x=Vi({config.dim})", f"y=Vj({config.dim})", "s=Pm(1)"]
        s = jax.device_put(jnp.array([0.5], dtype=jnp.float32), device)
        op = Genred(config.formula, aliases, reduction_op='Sum', axis=1)

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
        op = Genred(config.formula, aliases, reduction_op='Sum', axis=1)

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


def benchmark_multi_gpu(config: MultiGPUConfig, n_warmup: int, n_iter: int) -> dict:
    """Benchmark multi-GPU performance with sharding."""
    devices = jax.devices()
    n_devices = len(devices)

    if n_devices < 2:
        return None

    # Create mesh
    mesh = Mesh(np.array(devices), axis_names=('batch',))

    # Create data
    x_host = np.random.randn(config.n, config.dim).astype(np.float32)
    y_host = np.random.randn(config.n, config.dim).astype(np.float32)

    # Shard x across devices, replicate y
    shard_x = NamedSharding(mesh, P('batch', None))
    repl_y = NamedSharding(mesh, P(None, None))

    x = jax.device_put(x_host, shard_x)
    y = jax.device_put(y_host, repl_y)

    # Create operator with explicit @jax.jit
    if config.has_param:
        aliases = [f"x=Vi({config.dim})", f"y=Vj({config.dim})", "s=Pm(1)"]
        s = jnp.array([0.5], dtype=jnp.float32)
        op = Genred(config.formula, aliases, reduction_op='Sum', axis=1)

        def local_compute(x_loc, y_repl):
            return op(x_loc, y_repl, s)

        @jax.jit
        @shard_map.shard_map(mesh=mesh, in_specs=(P('batch', None), P(None, None)), out_specs=P('batch', None))
        def compute_sharded(x_loc, y_repl):
            return local_compute(x_loc, y_repl)
    else:
        aliases = [f"x=Vi({config.dim})", f"y=Vj({config.dim})"]
        op = Genred(config.formula, aliases, reduction_op='Sum', axis=1)

        def local_compute(x_loc, y_repl):
            return op(x_loc, y_repl)

        @jax.jit
        @shard_map.shard_map(mesh=mesh, in_specs=(P('batch', None), P(None, None)), out_specs=P('batch', None))
        def compute_sharded(x_loc, y_repl):
            return local_compute(x_loc, y_repl)

    # Warmup with error handling
    try:
        _ = compute_sharded(x, y)
        jax.block_until_ready(_)
    except Exception as e:
        print(f"    {Colors.RED}Multi-GPU setup failed: {e}{Colors.RESET}")
        return None

    for _ in range(n_warmup):
        _ = compute_sharded(x, y)
        jax.block_until_ready(_)

    # Benchmark
    times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        result = compute_sharded(x, y)
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


def benchmark_single_gpu_backward(config: MultiGPUConfig, n_warmup: int, n_iter: int) -> dict:
    """Benchmark single GPU gradient computation with explicit @jax.jit."""
    devices = jax.devices()
    device = devices[0]

    x = jax.device_put(
        jnp.array(np.random.randn(config.n, config.dim).astype(np.float32)),
        device
    )
    y = jax.device_put(
        jnp.array(np.random.randn(config.n, config.dim).astype(np.float32)),
        device
    )

    if config.has_param:
        aliases = [f"x=Vi({config.dim})", f"y=Vj({config.dim})", "s=Pm(1)"]
        s = jax.device_put(jnp.array([0.5], dtype=jnp.float32), device)
        op = Genred(config.formula, aliases, reduction_op='Sum', axis=1)

        @jax.jit
        def compute_grad(x, y):
            def loss(x):
                return jnp.sum(op(x, y, s))

            return jax.grad(loss)(x)
    else:
        aliases = [f"x=Vi({config.dim})", f"y=Vj({config.dim})"]
        op = Genred(config.formula, aliases, reduction_op='Sum', axis=1)

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
        'p10': float(np.percentile(times, 10)),
        'p90': float(np.percentile(times, 90)),
    }


def benchmark_multi_gpu_backward(config: MultiGPUConfig, n_warmup: int, n_iter: int) -> dict:
    """Benchmark multi-GPU gradient computation."""
    devices = jax.devices()
    n_devices = len(devices)

    if n_devices < 2:
        return None

    mesh = Mesh(np.array(devices), axis_names=('batch',))

    x_host = np.random.randn(config.n, config.dim).astype(np.float32)
    y_host = np.random.randn(config.n, config.dim).astype(np.float32)

    shard_x = NamedSharding(mesh, P('batch', None))
    repl_y = NamedSharding(mesh, P(None, None))

    x = jax.device_put(x_host, shard_x)
    y = jax.device_put(y_host, repl_y)

    if config.has_param:
        aliases = [f"x=Vi({config.dim})", f"y=Vj({config.dim})", "s=Pm(1)"]
        s = jnp.array([0.5], dtype=jnp.float32)
        op = Genred(config.formula, aliases, reduction_op='Sum', axis=1)

        @shard_map.shard_map(mesh=mesh, in_specs=(P('batch', None), P(None, None)), out_specs=P())
        def loss_fn(x_loc, y_repl):
            local_sum = jnp.sum(op(x_loc, y_repl, s))
            global_loss = jax.lax.psum(local_sum, 'batch')
            return global_loss / jax.lax.psum(1, 'batch')
    else:
        aliases = [f"x=Vi({config.dim})", f"y=Vj({config.dim})"]
        op = Genred(config.formula, aliases, reduction_op='Sum', axis=1)

        @shard_map.shard_map(mesh=mesh, in_specs=(P('batch', None), P(None, None)), out_specs=P())
        def loss_fn(x_loc, y_repl):
            local_sum = jnp.sum(op(x_loc, y_repl))
            global_loss = jax.lax.psum(local_sum, 'batch')
            return global_loss / jax.lax.psum(1, 'batch')

    @jax.jit
    def compute_grad(x, y):
        return jax.grad(loss_fn)(x, y)

    # Warmup with error handling
    try:
        _ = compute_grad(x, y)
        jax.block_until_ready(_)
    except Exception as e:
        print(f"    {Colors.RED}Multi-GPU backward setup failed: {e}{Colors.RESET}")
        return None

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
        'p10': float(np.percentile(times, 10)),
        'p90': float(np.percentile(times, 90)),
    }


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_multi_gpu_benchmarks(save_results: bool = True):
    """Run multi-GPU benchmarks."""
    if not JAX_AVAILABLE or not KEOPS_AVAILABLE:
        print(f"{Colors.RED}JAX or KeOps not available. Cannot run benchmarks.{Colors.RESET}")
        return False

    print_header("KeOps JAX Multi-GPU Scaling Benchmark")

    # Print configuration
    print(f"  JAX version: {jax.__version__}")
    print(f"  Number of devices: {N_DEVICES}")
    if N_DEVICES > 0:
        for i, dev in enumerate(jax.devices()):
            print(f"    Device {i}: {dev}")
    print(f"  Warmup iterations: {N_WARMUP}")
    print(f"  Benchmark iterations: {N_ITER}")
    print()
    print(f"  {Colors.CYAN}Note: Using MEDIAN for all comparisons (accounts for async variance){Colors.RESET}")
    print()

    if N_DEVICES < 2:
        print(
            f"{Colors.YELLOW}Warning: Only {N_DEVICES} device(s) available. Multi-GPU tests will be skipped.{Colors.RESET}")
        print()

    results: List[MultiGPUResult] = []

    # =========================
    # Forward Pass Benchmarks
    # =========================
    print_subheader("Forward Pass: Single vs Multi-GPU")

    table = ASCIITable([
        TableColumn("Problem", 18),
        TableColumn("N", 10, 'right'),
        TableColumn("Single (ms)", 12, 'right'),
        TableColumn(f"Multi-{N_DEVICES}GPU", 12, 'right'),
        TableColumn("Speedup", 10, 'right'),
        TableColumn("Efficiency", 10, 'right'),
    ], title=f"Forward Pass Scaling ({N_DEVICES} GPUs) - Median times")

    for config in MULTIGPU_CONFIGS:
        print(f"  Running {config.name}...", end='', flush=True)

        try:
            single_result = benchmark_single_gpu(config, N_WARMUP, N_ITER)
            multi_result = benchmark_multi_gpu(config, N_WARMUP, N_ITER)

            if multi_result:
                speedup = single_result['median'] / multi_result['median']
                efficiency = (speedup / N_DEVICES) * 100

                table.add_row([
                    config.name,
                    f"{config.n:,}",
                    f"{single_result['median']:.3f}",
                    f"{multi_result['median']:.3f}",
                    format_speedup(speedup),
                    format_efficiency(efficiency),
                ])

                results.append(MultiGPUResult(
                    config_name=config.name,
                    n=config.n,
                    dim=config.dim,
                    n_devices=N_DEVICES,
                    single_gpu_median_ms=single_result['median'],
                    single_gpu_mean_ms=single_result['mean'],
                    single_gpu_min_ms=single_result['min'],
                    multi_gpu_median_ms=multi_result['median'],
                    multi_gpu_mean_ms=multi_result['mean'],
                    multi_gpu_min_ms=multi_result['min'],
                    speedup=speedup,
                    efficiency_pct=efficiency,
                    pass_type='forward',
                ))

                print(f" done (1GPU: {single_result['median']:.3f}ms, {N_DEVICES}GPU: {multi_result['median']:.3f}ms)")
            else:
                table.add_row([
                    config.name,
                    f"{config.n:,}",
                    f"{single_result['median']:.3f}",
                    "-",
                    "-",
                    "-",
                ])
                print(f" done (single GPU only)")

        except Exception as e:
            print(f" {Colors.RED}ERROR: {e}{Colors.RESET}")
            import traceback
            traceback.print_exc()
            table.add_row([config.name, f"{config.n:,}", "ERROR", "-", "-", "-"])

    table.print()

    # =========================
    # Backward Pass Benchmarks
    # =========================
    if N_DEVICES >= 2:
        print_subheader("Backward Pass: Single vs Multi-GPU")

        table = ASCIITable([
            TableColumn("Problem", 18),
            TableColumn("N", 10, 'right'),
            TableColumn("Single (ms)", 12, 'right'),
            TableColumn(f"Multi-{N_DEVICES}GPU", 12, 'right'),
            TableColumn("Speedup", 10, 'right'),
            TableColumn("Efficiency", 10, 'right'),
        ], title=f"Backward Pass Scaling ({N_DEVICES} GPUs) - Median times")

        # Use smaller configs for backward
        backward_configs = [c for c in MULTIGPU_CONFIGS if c.n <= 100_000]

        for config in backward_configs:
            print(f"  Running {config.name}...", end='', flush=True)

            try:
                single_result = benchmark_single_gpu_backward(config, N_WARMUP, N_ITER)
                multi_result = benchmark_multi_gpu_backward(config, N_WARMUP, N_ITER)

                if multi_result:
                    speedup = single_result['median'] / multi_result['median']
                    efficiency = (speedup / N_DEVICES) * 100

                    table.add_row([
                        config.name,
                        f"{config.n:,}",
                        f"{single_result['median']:.3f}",
                        f"{multi_result['median']:.3f}",
                        format_speedup(speedup),
                        format_efficiency(efficiency),
                    ])

                    results.append(MultiGPUResult(
                        config_name=config.name,
                        n=config.n,
                        dim=config.dim,
                        n_devices=N_DEVICES,
                        single_gpu_median_ms=single_result['median'],
                        single_gpu_mean_ms=single_result['mean'],
                        single_gpu_min_ms=single_result['min'],
                        multi_gpu_median_ms=multi_result['median'],
                        multi_gpu_mean_ms=multi_result['mean'],
                        multi_gpu_min_ms=multi_result['min'],
                        speedup=speedup,
                        efficiency_pct=efficiency,
                        pass_type='backward',
                    ))

                    print(
                        f" done (1GPU: {single_result['median']:.3f}ms, {N_DEVICES}GPU: {multi_result['median']:.3f}ms)")
                else:
                    table.add_row([
                        config.name,
                        f"{config.n:,}",
                        f"{single_result['median']:.3f}",
                        "-",
                        "-",
                        "-",
                    ])
                    print(f" done (single GPU only)")

            except Exception as e:
                print(f" {Colors.RED}ERROR: {e}{Colors.RESET}")
                table.add_row([config.name, f"{config.n:,}", "ERROR", "-", "-", "-"])

        table.print()

    # =========================
    # Summary
    # =========================
    print_subheader("Summary")

    forward_results = [r for r in results if r.pass_type == 'forward']
    backward_results = [r for r in results if r.pass_type == 'backward']

    if forward_results:
        speedups = [r.speedup for r in forward_results]
        efficiencies = [r.efficiency_pct for r in forward_results]
        print(f"  Forward pass ({len(forward_results)} tests):")
        print(f"    Average speedup: {format_speedup(np.mean(speedups))}")
        print(f"    Speedup range: {min(speedups):.2f}x - {max(speedups):.2f}x")
        print(f"    Average efficiency: {np.mean(efficiencies):.1f}%")
        print(f"    Efficiency range: {min(efficiencies):.1f}% - {max(efficiencies):.1f}%")

    if backward_results:
        speedups = [r.speedup for r in backward_results]
        efficiencies = [r.efficiency_pct for r in backward_results]
        print(f"  Backward pass ({len(backward_results)} tests):")
        print(f"    Average speedup: {format_speedup(np.mean(speedups))}")
        print(f"    Speedup range: {min(speedups):.2f}x - {max(speedups):.2f}x")
        print(f"    Average efficiency: {np.mean(efficiencies):.1f}%")
        print(f"    Efficiency range: {min(efficiencies):.1f}% - {max(efficiencies):.1f}%")

    print()
    print(f"  {Colors.CYAN}Efficiency = (speedup / num_gpus) * 100%")
    print(f"  100% = perfect linear scaling, >50% = good scaling{Colors.RESET}")

    # Save results
    if save_results and results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_multi_gpu_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\n  Results saved to: {filename}")

    return True


# =============================================================================
# Entry Points
# =============================================================================

def test_multi_gpu_scaling():
    """Test function for pytest."""
    if N_DEVICES < 2:
        print(f"Skipping multi-GPU test: only {N_DEVICES} device(s) available")
        return True

    return run_multi_gpu_benchmarks(save_results=False)


def main():
    """Main entry point."""
    success = run_multi_gpu_benchmarks(save_results=True)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())