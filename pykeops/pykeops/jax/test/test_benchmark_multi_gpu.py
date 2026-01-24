#!/usr/bin/env python3
"""
KeOps JAX Multi-GPU Benchmark
=============================
Tests multi-GPU scaling efficiency using JAX's sharding capabilities.

Measures:
- Single GPU baseline
- Multi-GPU parallel execution
- Scaling efficiency (speedup vs linear ideal)
- Memory distribution

Requirements:
- Multiple NVIDIA GPUs
- JAX with CUDA support
"""

import os
os.environ['JAX_KEOPS_DEBUG'] = '0'

import sys
import time
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

# =============================================================================
# JAX Setup - Must be before imports that use JAX
# =============================================================================

import jax
import jax.numpy as jnp

from test_utils import (
    print_header, print_subheader, print_info, print_success, print_warning, print_error,
    print_benchmark_table, print_environment_info, RICH_AVAILABLE
)

if RICH_AVAILABLE:
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from test_utils import console

# Check GPU availability
devices = jax.devices('gpu')
N_GPUS = len(devices)

if N_GPUS < 2:
    print_warning(f"Only {N_GPUS} GPU(s) available. Multi-GPU tests require 2+ GPUs.")
    print_info("Running single-GPU benchmark instead.")

# =============================================================================
# Import KeOps
# =============================================================================

try:
    from pykeops.jax import Genred, LazyTensor
    KEOPS_AVAILABLE = True
except ImportError as e:
    print_error(f"pykeops.jax not found: {e}")
    sys.exit(1)

# JAX sharding imports
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ScalingConfig:
    """Configuration for a scaling test."""
    name: str
    nx: int
    ny: int
    dim: int
    formula: str
    has_param: bool = False


@dataclass 
class ScalingResult:
    """Result from a scaling test."""
    config_name: str
    n_gpus: int
    time_ms: float
    time_std: float
    speedup: float
    efficiency_pct: float


# Test configurations
SCALING_CONFIGS = [
    ScalingConfig("Medium/SqDist", 20000, 20000, 3, "SqNorm2(x-y)"),
    ScalingConfig("Medium/Gaussian", 20000, 20000, 3, "Exp(-SqNorm2(x-y)*s)", True),
    ScalingConfig("Large/SqDist", 50000, 50000, 3, "SqNorm2(x-y)"),
    ScalingConfig("Large/Gaussian", 50000, 50000, 3, "Exp(-SqNorm2(x-y)*s)", True),
    ScalingConfig("XLarge/SqDist", 100000, 100000, 3, "SqNorm2(x-y)"),
]

N_WARMUP = 5
N_ITERS = 100
N_BATCH = 20  # Number of iterations per batch (sync only at end of batch)


# =============================================================================
# Sharding Utilities
# =============================================================================

def create_sharded_data(nx, ny, dim, n_gpus, seed=42):
    """Create data sharded across GPUs."""
    np.random.seed(seed)
    
    x_np = np.random.randn(nx, dim).astype(np.float32)
    y_np = np.random.randn(ny, dim).astype(np.float32)
    s_np = np.array([0.5], dtype=np.float32)
    
    # Create mesh
    devices = jax.devices('gpu')[:n_gpus]
    mesh = Mesh(np.array(devices), ('batch',))
    
    # Create sharding specs
    # Shard x across batch dimension
    x_sharding = NamedSharding(mesh, P('batch', None))
    y_sharding = NamedSharding(mesh, P(None, None))  # Replicate y
    s_sharding = NamedSharding(mesh, P(None))  # Replicate s
    
    # Create sharded arrays
    x = jax.device_put(jnp.array(x_np), x_sharding)
    y = jax.device_put(jnp.array(y_np), y_sharding)
    s = jax.device_put(jnp.array(s_np), s_sharding)
    
    return x, y, s, mesh


def time_sharded_forward(op, args, mesh, n_warmup=N_WARMUP, n_iters=N_ITERS, batch_size=N_BATCH):
    """
    Time sharded forward pass using batched timing.
    
    To avoid block_until_ready() overhead dominating the measurement,
    we run `batch_size` iterations, then sync once and divide.
    """
    # Create sharded compute function
    @jax.jit
    def compute(*a):
        return op(*a)
    
    # Warmup
    for _ in range(n_warmup):
        result = compute(*args)
    jax.block_until_ready(result)
    
    # Time in batches
    times = []
    n_batches = n_iters // batch_size
    
    for _ in range(n_batches):
        start = time.perf_counter()
        for _ in range(batch_size):
            result = compute(*args)
        jax.block_until_ready(result)  # Sync once per batch
        batch_time = (time.perf_counter() - start) * 1000
        times.append(batch_time / batch_size)  # Per-iteration time
    
    return np.median(times), np.std(times)


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_scaling_test(config: ScalingConfig, gpu_counts: List[int]) -> List[ScalingResult]:
    """Run scaling test for different GPU counts."""
    results = []
    baseline_time = None
    
    for n_gpus in gpu_counts:
        if n_gpus > N_GPUS:
            continue
        
        # Create data
        x, y, s, mesh = create_sharded_data(config.nx, config.ny, config.dim, n_gpus)
        
        # Create operator
        aliases = [f"x=Vi({config.dim})", f"y=Vj({config.dim})"]
        if config.has_param:
            aliases.append("s=Pm(1)")
        
        op = Genred(config.formula, aliases, reduction_op='Sum', axis=1)
        args = (x, y, s) if config.has_param else (x, y)
        
        # Time
        time_ms, time_std = time_sharded_forward(op, args, mesh)
        
        # Compute speedup
        if baseline_time is None:
            baseline_time = time_ms
            speedup = 1.0
        else:
            speedup = baseline_time / time_ms
        
        # Efficiency (compared to linear scaling)
        efficiency = (speedup / n_gpus) * 100 if n_gpus > 1 else 100.0
        
        results.append(ScalingResult(
            config_name=config.name,
            n_gpus=n_gpus,
            time_ms=time_ms,
            time_std=time_std,
            speedup=speedup,
            efficiency_pct=efficiency,
        ))
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print_header("KeOps JAX Multi-GPU Benchmark", 
                f"Scaling test across {N_GPUS} GPUs")
    
    print_environment_info()
    
    if N_GPUS < 2:
        print_error("Multi-GPU benchmark requires at least 2 GPUs")
        return 1
    
    # GPU counts to test
    gpu_counts = [1, 2, 4, 8]
    gpu_counts = [g for g in gpu_counts if g <= N_GPUS]
    
    print_info(f"Testing with GPU counts: {gpu_counts}")
    
    all_results: Dict[str, List[ScalingResult]] = {}
    
    print_subheader("Running Scaling Tests")
    
    for config in SCALING_CONFIGS:
        print_info(f"Testing {config.name}...")
        
        try:
            results = run_scaling_test(config, gpu_counts)
            all_results[config.name] = results
            
            # Print immediate results
            for r in results:
                speedup_str = f"↑{r.speedup:.2f}x" if r.speedup > 1 else f"→{r.speedup:.2f}x"
                print(f"    {r.n_gpus} GPU(s): {r.time_ms:.2f}ms ({speedup_str}, {r.efficiency_pct:.0f}% eff)")
        
        except Exception as e:
            print_warning(f"Failed: {e}")
            continue
    
    # ==========================================================================
    # Summary Tables
    # ==========================================================================
    print_subheader("Scaling Results Summary")
    
    for config_name, results in all_results.items():
        rows = []
        for r in results:
            rows.append({
                'n_gpus': f"{r.n_gpus} GPU(s)",
                'time_ms': r.time_ms,
                'time_std': r.time_std,
                'speedup': r.speedup,
                'efficiency': f"{r.efficiency_pct:.0f}%",
            })
        
        columns = [
            ('n_gpus', 'Configuration', 12),
            ('time_ms', 'Time (ms)', 12),
            ('time_std', '± std', 8),
            ('speedup', 'Speedup', 10),
            ('efficiency', 'Efficiency', 10),
        ]
        
        print_benchmark_table(f"Scaling: {config_name}", rows, columns)
    
    # ==========================================================================
    # Overall Summary
    # ==========================================================================
    print_subheader("Overall Scaling Efficiency")
    
    # Calculate average efficiency for each GPU count
    for n_gpus in gpu_counts[1:]:  # Skip 1 GPU
        efficiencies = []
        for results in all_results.values():
            for r in results:
                if r.n_gpus == n_gpus:
                    efficiencies.append(r.efficiency_pct)
        
        if efficiencies:
            avg_eff = np.mean(efficiencies)
            if avg_eff >= 80:
                print_success(f"{n_gpus} GPUs: {avg_eff:.0f}% average efficiency (excellent)")
            elif avg_eff >= 50:
                print_info(f"{n_gpus} GPUs: {avg_eff:.0f}% average efficiency (good)")
            else:
                print_warning(f"{n_gpus} GPUs: {avg_eff:.0f}% average efficiency (poor)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_multi_gpu_{timestamp}.json"
    
    serializable = {k: [asdict(r) for r in v] for k, v in all_results.items()}
    with open(output_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print_info(f"Results saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
