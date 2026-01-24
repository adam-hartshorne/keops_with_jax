#!/usr/bin/env python3
"""
KeOps JAX Single GPU Benchmark
==============================
Performance comparison: JAX KeOps vs PyTorch KeOps on a single GPU.

Measures:
- Forward pass timing
- Gradient computation timing
- Various problem sizes and formulas
- Memory efficiency
"""

import os
os.environ['JAX_KEOPS_DEBUG'] = '0'

import sys
import time
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import jax
import jax.numpy as jnp

from test_utils import (
    print_header, print_subheader, print_info, print_success, print_warning,
    print_benchmark_table, print_environment_info, RICH_AVAILABLE
)

if RICH_AVAILABLE:
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from test_utils import console

# =============================================================================
# Import KeOps
# =============================================================================

try:
    from pykeops.jax import Genred as Genred_jax, LazyTensor as LazyTensor_jax
    JAX_AVAILABLE = True
except ImportError as e:
    print(f"Error: pykeops.jax not found: {e}")
    JAX_AVAILABLE = False
    sys.exit(1)

try:
    import torch
    from pykeops.torch import Genred as Genred_torch, LazyTensor as LazyTensor_torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


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
    has_param: bool = False


@dataclass
class BenchmarkResult:
    """Result from a single benchmark."""
    config_name: str
    jax_forward_ms: float
    jax_forward_std: float
    jax_grad_ms: float
    jax_grad_std: float
    torch_forward_ms: Optional[float] = None
    torch_forward_std: Optional[float] = None
    torch_grad_ms: Optional[float] = None
    torch_grad_std: Optional[float] = None
    speedup_forward: Optional[float] = None
    speedup_grad: Optional[float] = None


# Benchmark configurations
BENCHMARK_CONFIGS = [
    # Small problems - should show overhead
    BenchmarkConfig("Small/SqDist", 1000, 1000, 3, "SqNorm2(x-y)"),
    BenchmarkConfig("Small/Gaussian", 1000, 1000, 3, "Exp(-SqNorm2(x-y)*s)", True),
    
    # Medium problems - typical use case
    BenchmarkConfig("Medium/SqDist", 10000, 10000, 3, "SqNorm2(x-y)"),
    BenchmarkConfig("Medium/Gaussian", 10000, 10000, 3, "Exp(-SqNorm2(x-y)*s)", True),
    
    # Large problems - GPU-bound
    BenchmarkConfig("Large/SqDist", 50000, 20000, 3, "SqNorm2(x-y)"),
    BenchmarkConfig("Large/Gaussian", 50000, 20000, 3, "Exp(-SqNorm2(x-y)*s)", True),
    
    # Very large - stress test
    BenchmarkConfig("XLarge/SqDist", 100000, 50000, 3, "SqNorm2(x-y)"),
    
    # High-dimensional
    BenchmarkConfig("HighDim/SqDist", 10000, 10000, 32, "SqNorm2(x-y)"),
]

# Number of iterations for timing
N_WARMUP = 10
N_ITERS = 100
N_BATCH = 20  # Number of iterations per batch (sync only at end of batch)


# =============================================================================
# Timing Utilities
# =============================================================================

def time_jax_forward(op, args, n_warmup=N_WARMUP, n_iters=N_ITERS, batch_size=N_BATCH):
    """
    Time JAX forward pass using batched timing.
    
    To avoid block_until_ready() overhead dominating the measurement,
    we run `batch_size` iterations, then sync once and divide.
    """
    jit_op = jax.jit(lambda *a: op(*a))
    
    # Warmup
    for _ in range(n_warmup):
        result = jit_op(*args)
    jax.block_until_ready(result)
    
    # Time in batches
    times = []
    n_batches = n_iters // batch_size
    
    for _ in range(n_batches):
        start = time.perf_counter()
        for _ in range(batch_size):
            result = jit_op(*args)
        jax.block_until_ready(result)  # Sync once per batch
        batch_time = (time.perf_counter() - start) * 1000
        times.append(batch_time / batch_size)  # Per-iteration time
    
    return np.median(times), np.std(times)


def time_jax_gradient(op, args, grad_idx=0, n_warmup=N_WARMUP, n_iters=N_ITERS, batch_size=N_BATCH):
    """
    Time JAX gradient computation using batched timing.
    """
    def forward(*a):
        return op(*a).sum()
    
    grad_fn = jax.jit(jax.grad(forward, argnums=grad_idx))
    
    # Warmup
    for _ in range(n_warmup):
        result = grad_fn(*args)
    jax.block_until_ready(result)
    
    # Time in batches
    times = []
    n_batches = n_iters // batch_size
    
    for _ in range(n_batches):
        start = time.perf_counter()
        for _ in range(batch_size):
            result = grad_fn(*args)
        jax.block_until_ready(result)
        batch_time = (time.perf_counter() - start) * 1000
        times.append(batch_time / batch_size)
    
    return np.median(times), np.std(times)


def time_torch_forward(op, args, n_warmup=5, n_iters=N_ITERS, batch_size=N_BATCH):
    """
    Time PyTorch forward pass using batched timing.
    """
    if not TORCH_AVAILABLE:
        return None, None
    
    # Warmup
    for _ in range(n_warmup):
        _ = op(*args)
    torch.cuda.synchronize()
    
    # Time in batches
    times = []
    n_batches = n_iters // batch_size
    
    for _ in range(n_batches):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(batch_size):
            _ = op(*args)
        torch.cuda.synchronize()
        batch_time = (time.perf_counter() - start) * 1000
        times.append(batch_time / batch_size)
    
    return np.mean(times), np.std(times)


def time_torch_gradient(op, args, grad_idx=0, n_warmup=5, n_iters=N_ITERS, batch_size=N_BATCH):
    """
    Time PyTorch gradient computation using batched timing.
    """
    if not TORCH_AVAILABLE:
        return None, None
    
    # Warmup
    for _ in range(n_warmup):
        args[grad_idx].grad = None
        result = op(*args).sum()
        result.backward()
    torch.cuda.synchronize()
    
    # Time in batches
    times = []
    n_batches = n_iters // batch_size
    
    for _ in range(n_batches):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(batch_size):
            args[grad_idx].grad = None
            result = op(*args).sum()
            result.backward()
        torch.cuda.synchronize()
        batch_time = (time.perf_counter() - start) * 1000
        times.append(batch_time / batch_size)
    
    return np.mean(times), np.std(times)


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    np.random.seed(42)
    
    # Generate data
    x_np = np.random.randn(config.nx, config.dim).astype(np.float32)
    y_np = np.random.randn(config.ny, config.dim).astype(np.float32)
    s_np = np.array([0.5], dtype=np.float32)
    
    # JAX arrays
    x_jax = jnp.array(x_np)
    y_jax = jnp.array(y_np)
    s_jax = jnp.array(s_np)
    
    # Create JAX operator
    aliases = [f"x=Vi({config.dim})", f"y=Vj({config.dim})"]
    if config.has_param:
        aliases.append("s=Pm(1)")
    
    op_jax = Genred_jax(config.formula, aliases, reduction_op='Sum', axis=1)
    jax_args = (x_jax, y_jax, s_jax) if config.has_param else (x_jax, y_jax)
    
    # Time JAX
    jax_fwd_ms, jax_fwd_std = time_jax_forward(op_jax, jax_args)
    jax_grad_ms, jax_grad_std = time_jax_gradient(op_jax, jax_args)
    
    # PyTorch
    torch_fwd_ms, torch_fwd_std = None, None
    torch_grad_ms, torch_grad_std = None, None
    
    if TORCH_AVAILABLE:
        x_torch = torch.tensor(x_np, device='cuda', requires_grad=True)
        y_torch = torch.tensor(y_np, device='cuda')
        s_torch = torch.tensor(s_np, device='cuda')
        
        op_torch = Genred_torch(config.formula, aliases, reduction_op='Sum', axis=1)
        torch_args = [x_torch, y_torch, s_torch] if config.has_param else [x_torch, y_torch]
        
        torch_fwd_ms, torch_fwd_std = time_torch_forward(op_torch, torch_args)
        torch_grad_ms, torch_grad_std = time_torch_gradient(op_torch, torch_args)
    
    # Compute speedups
    speedup_fwd = torch_fwd_ms / jax_fwd_ms if torch_fwd_ms else None
    speedup_grad = torch_grad_ms / jax_grad_ms if torch_grad_ms else None
    
    return BenchmarkResult(
        config_name=config.name,
        jax_forward_ms=jax_fwd_ms,
        jax_forward_std=jax_fwd_std,
        jax_grad_ms=jax_grad_ms,
        jax_grad_std=jax_grad_std,
        torch_forward_ms=torch_fwd_ms,
        torch_forward_std=torch_fwd_std,
        torch_grad_ms=torch_grad_ms,
        torch_grad_std=torch_grad_std,
        speedup_forward=speedup_fwd,
        speedup_grad=speedup_grad,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    print_header("KeOps JAX Single GPU Benchmark", 
                "JAX KeOps vs PyTorch KeOps Performance Comparison")
    
    print_environment_info()
    
    if not TORCH_AVAILABLE:
        print_warning("PyTorch not available - only JAX times will be shown")
    
    results: List[BenchmarkResult] = []
    
    print_subheader("Running Benchmarks")
    print_info(f"Warmup: {N_WARMUP} iterations, Timing: {N_ITERS} iterations")
    print()
    
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Benchmarking...", total=len(BENCHMARK_CONFIGS))
            
            for config in BENCHMARK_CONFIGS:
                progress.update(task, description=f"Running {config.name}...")
                result = run_benchmark(config)
                results.append(result)
                progress.advance(task)
    else:
        for i, config in enumerate(BENCHMARK_CONFIGS):
            print(f"  [{i+1}/{len(BENCHMARK_CONFIGS)}] {config.name}...", end='', flush=True)
            result = run_benchmark(config)
            results.append(result)
            print(f" JAX: {result.jax_forward_ms:.2f}ms", end='')
            if result.torch_forward_ms:
                print(f", PyTorch: {result.torch_forward_ms:.2f}ms", end='')
            print()
    
    # ==========================================================================
    # Print Results
    # ==========================================================================
    print_subheader("Forward Pass Results")
    
    rows = []
    for r in results:
        row = {
            'name': r.config_name,
            'jax_ms': r.jax_forward_ms,
            'jax_std': r.jax_forward_std,
        }
        if r.torch_forward_ms:
            row['torch_ms'] = r.torch_forward_ms
            row['torch_std'] = r.torch_forward_std
            row['speedup'] = r.speedup_forward
        rows.append(row)
    
    columns = [
        ('name', 'Configuration', 20),
        ('jax_ms', 'JAX (ms)', 12),
        ('jax_std', '± std', 8),
    ]
    if TORCH_AVAILABLE:
        columns.extend([
            ('torch_ms', 'PyTorch (ms)', 12),
            ('torch_std', '± std', 8),
            ('speedup', 'Speedup', 10),
        ])
    
    print_benchmark_table("Forward Pass Timing", rows, columns)
    
    print_subheader("Gradient Results")
    
    rows = []
    for r in results:
        row = {
            'name': r.config_name,
            'jax_ms': r.jax_grad_ms,
            'jax_std': r.jax_grad_std,
        }
        if r.torch_grad_ms:
            row['torch_ms'] = r.torch_grad_ms
            row['torch_std'] = r.torch_grad_std
            row['speedup'] = r.speedup_grad
        rows.append(row)
    
    print_benchmark_table("Gradient Timing", rows, columns)
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print_subheader("Summary")
    
    if TORCH_AVAILABLE:
        fwd_speedups = [r.speedup_forward for r in results if r.speedup_forward]
        grad_speedups = [r.speedup_grad for r in results if r.speedup_grad]
        
        avg_fwd = np.mean(fwd_speedups) if fwd_speedups else 0
        avg_grad = np.mean(grad_speedups) if grad_speedups else 0
        
        print_info(f"Average forward speedup: {avg_fwd:.2f}x (PyTorch/JAX)")
        print_info(f"Average gradient speedup: {avg_grad:.2f}x (PyTorch/JAX)")
        
        if avg_fwd > 1.0:
            print_success("JAX KeOps is faster than PyTorch KeOps on average!")
        elif avg_fwd > 0.9:
            print_info("JAX KeOps has similar performance to PyTorch KeOps")
        else:
            print_warning(f"JAX KeOps is {1/avg_fwd:.1f}x slower than PyTorch KeOps")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_single_gpu_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print_info(f"Results saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
