#!/usr/bin/env python3
"""
KeOps JAX Single GPU Benchmark (Visualized + Varifold + Summary)
================================================================
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ['JAX_KEOPS_DEBUG'] = '0'

import sys
import pytest

# Every test here needs a GPU and compares against PyTorch KeOps. conftest.py registers these
# markers and skips on missing hardware, so declaring them at module level is what makes
# `pytest -m pytorch` and `pytest -m gpu` select anything.
pytestmark = [pytest.mark.gpu, pytest.mark.pytorch]

import time
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional

import jax
import jax.numpy as jnp

# --- Visual Dependencies ---
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.panel import Panel
    from rich import box

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    print("For best visuals, please install 'rich': pip install rich")

try:
    from pykeops.jax import Genred as Genred_jax

    JAX_AVAILABLE = True
except ImportError as e:
    print(f"Error: pykeops.jax not found: {e}")
    sys.exit(1)

try:
    import torch
    from pykeops.torch import Genred as Genred_torch

    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    name: str
    nx: int
    ny: int
    dim: int
    formula: str
    mode: str = "standard"  # 'standard' or 'varifold'


@dataclass
class BenchmarkResult:
    config_name: str
    size_str: str  # Added size string
    jax_forward_ms: float
    jax_forward_std: float
    jax_backward_ms: float = 0.0
    jax_backward_std: float = 0.0
    torch_forward_ms: Optional[float] = None
    torch_backward_ms: Optional[float] = None
    speedup_forward: Optional[float] = None
    speedup_backward: Optional[float] = None


BENCHMARK_CONFIGS = [
    # --- Standard ---
    BenchmarkConfig("Small (SqDist)", 5000, 5000, 3, "SqNorm2(x-y)"),
    BenchmarkConfig("Medium (SqDist)", 20000, 20000, 3, "SqNorm2(x-y)"),
    BenchmarkConfig("Large (SqDist)", 50000, 50000, 3, "SqNorm2(x-y)"),
    BenchmarkConfig("Huge (SqDist)", 500000, 500000, 3, "SqNorm2(x-y)"),

    # --- Varifolds ---
    # Formula: Gaussian(x,y) * (u|v)^2
    BenchmarkConfig("Large (Varifold)", 50000, 50000, 3,
                    "Exp(-SqNorm2(x-y)*s) * Square((u|v))", mode="varifold"),
    BenchmarkConfig("Huge (Varifold)", 500000, 500000, 3,
                    "Exp(-SqNorm2(x-y)*s) * Square((u|v))", mode="varifold"),
]

N_WARMUP = 10
N_ITERS = 50
TIMING_BATCH = 10


# =============================================================================
# Timing & Execution
# =============================================================================

def time_func(name, func, args, n_warmup, n_iters, batch_size):
    """Generic timer with Rich progress bars."""
    times = []

    if RICH_AVAILABLE:
        progress_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True
        )
    else:
        from contextlib import nullcontext
        progress_ctx = nullcontext()

    with progress_ctx as progress:
        # 1. Warmup
        if RICH_AVAILABLE:
            task = progress.add_task(f"[yellow]{name} Warmup...", total=n_warmup)

        for _ in range(n_warmup):
            func(*args)
            if RICH_AVAILABLE: progress.advance(task)

        # 2. Timing
        if RICH_AVAILABLE:
            task = progress.add_task(f"[green]{name} Benchmarking...", total=n_iters)

        n_batches = max(1, n_iters // batch_size)

        for _ in range(n_batches):
            # Sync before
            func(*args, sync=True)

            start = time.perf_counter()
            for _ in range(batch_size):
                func(*args, sync=False)  # Async calls

            # Sync after
            func(*args, sync=True)
            end = time.perf_counter()

            times.append((end - start) * 1000 / batch_size)
            if RICH_AVAILABLE: progress.advance(task, advance=batch_size)

    return np.median(times), np.std(times)


# Wrappers to handle sync
def jax_run(op, args, sync=False):
    res = op(*args)
    if sync: res.block_until_ready()
    return res


def torch_run(op, args, sync=False):
    res = op(*args)
    if sync: torch.cuda.synchronize()
    return res


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    size_str = f"N={config.nx:,}"

    if RICH_AVAILABLE:
        # Add size to header
        console.rule(f"[bold blue]{config.name} ({size_str})[/bold blue]")
    else:
        print(f"--- {config.name} ({size_str}) ---")

    # 1. Generate Data
    np.random.seed(42)
    M, N, D = config.nx, config.ny, config.dim

    # Base Data
    x_np = np.random.randn(M, D).astype(np.float32)
    y_np = np.random.randn(N, D).astype(np.float32)
    s_np = np.array([0.5], dtype=np.float32)

    arrays_np = [x_np, y_np]

    if config.mode == "varifold":
        u_np = np.random.randn(M, D).astype(np.float32)
        v_np = np.random.randn(N, D).astype(np.float32)
        arrays_np.extend([u_np, v_np])

    arrays_np.append(s_np)  # Param always last

    # 2. Setup Aliases
    if config.mode == "varifold":
        aliases = [f"x=Vi({D})", f"y=Vj({D})", f"u=Vi({D})", f"v=Vj({D})", "s=Pm(1)"]
    else:
        aliases = [f"x=Vi({D})", f"y=Vj({D})", "s=Pm(1)"]

    # --- JAX Forward ---
    jax_args = [jnp.array(a) for a in arrays_np]
    op_jax = Genred_jax(config.formula, aliases, reduction_op='Sum', axis=1)
    jit_op = jax.jit(lambda *a: op_jax(*a))  # JIT compile the op

    jax_ms, jax_std = time_func("JAX Forward", lambda *a, sync=False: jax_run(jit_op, a, sync),
                                jax_args, N_WARMUP, N_ITERS, TIMING_BATCH)

    # --- JAX Backward ---
    def jax_grad_fn(*args):
        def loss(*a):
            return op_jax(*a).sum()

        return jax.grad(loss, argnums=0)(*args)

    jit_grad_fn = jax.jit(jax_grad_fn)

    jax_bwd_ms, jax_bwd_std = time_func("JAX Backward", lambda *a, sync=False: jax_run(jit_grad_fn, a, sync),
                                        jax_args, N_WARMUP, N_ITERS, TIMING_BATCH)

    # --- Torch Forward ---
    torch_ms = None
    torch_bwd_ms = None
    if TORCH_AVAILABLE:
        torch_args = [torch.tensor(a, device='cuda') for a in arrays_np]
        torch_args[0].requires_grad_(True)  # Enable grad for x
        op_torch = Genred_torch(config.formula, aliases, reduction_op='Sum', axis=1)

        torch_ms, torch_std = time_func("PyTorch Forward", lambda *a, sync=False: torch_run(op_torch, a, sync),
                                        torch_args, N_WARMUP, N_ITERS, TIMING_BATCH)

        # --- Torch Backward ---
        def torch_backward_fn(*args, sync=False):
            args[0].grad = None
            result = op_torch(*args).sum()
            result.backward()
            if sync:
                torch.cuda.synchronize()
            return args[0].grad

        torch_bwd_ms, torch_bwd_std = time_func("PyTorch Backward", torch_backward_fn,
                                                torch_args, N_WARMUP, N_ITERS, TIMING_BATCH)

    # Report
    speedup_fwd = torch_ms / jax_ms if torch_ms else None
    speedup_bwd = torch_bwd_ms / jax_bwd_ms if torch_bwd_ms else None

    def format_speedup_inline(speedup):
        """Format speedup with arrow and color for inline display."""
        if speedup is None:
            return "-"
        if speedup > 1.05:
            return f"[bold green]↑ {speedup:.2f}x[/bold green]"
        elif speedup < 0.95:
            return f"[bold red]↓ {speedup:.2f}x[/bold red]"
        else:
            return f"[bold yellow]→ {speedup:.2f}x[/bold yellow]"

    if RICH_AVAILABLE:
        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("JAX Forward", f"[green]{jax_ms:.3f} ms[/green] ± {jax_std:.3f}")
        table.add_row("JAX Backward", f"[green]{jax_bwd_ms:.3f} ms[/green] ± {jax_bwd_std:.3f}")
        if torch_ms:
            table.add_row("PyTorch Forward", f"[yellow]{torch_ms:.3f} ms[/yellow]")
            table.add_row("PyTorch Backward", f"[yellow]{torch_bwd_ms:.3f} ms[/yellow]")
            table.add_row("Speedup Forward", format_speedup_inline(speedup_fwd))
            table.add_row("Speedup Backward", format_speedup_inline(speedup_bwd))
        console.print(table)
        console.print()

    return BenchmarkResult(
        config_name=config.name,
        size_str=size_str,
        jax_forward_ms=jax_ms,
        jax_forward_std=jax_std,
        jax_backward_ms=jax_bwd_ms,
        jax_backward_std=jax_bwd_std,
        torch_forward_ms=torch_ms,
        torch_backward_ms=torch_bwd_ms,
        speedup_forward=speedup_fwd,
        speedup_backward=speedup_bwd,
    )


# =============================================================================
# Final Summary
# =============================================================================

def print_summary(results: List[BenchmarkResult]):
    """Prints a consolidated summary table of all results."""
    if not RICH_AVAILABLE:
        return

    def format_speedup(speedup):
        """Format speedup with arrow and color."""
        if speedup is None:
            return "-"
        if speedup > 1.05:
            return f"[bold green]↑ {speedup:.2f}x[/bold green]"  # JAX faster
        elif speedup < 0.95:
            return f"[bold red]↓ {speedup:.2f}x[/bold red]"  # JAX slower
        else:
            return f"[bold yellow]→ {speedup:.2f}x[/bold yellow]"  # About the same

    # Forward Pass Table
    table = Table(title="[bold cyan]Forward Pass Performance[/bold cyan]", box=box.ROUNDED)
    table.add_column("Configuration", style="cyan", justify="left")
    table.add_column("Size", style="dim", justify="right")
    table.add_column("JAX (ms)", style="green", justify="right")
    table.add_column("PyTorch (ms)", style="yellow", justify="right")
    table.add_column("Speedup", justify="right")

    for r in results:
        jax_str = f"{r.jax_forward_ms:.2f}"
        torch_str = f"{r.torch_forward_ms:.2f}" if r.torch_forward_ms else "-"
        speedup_str = format_speedup(r.speedup_forward)
        table.add_row(r.config_name, r.size_str, jax_str, torch_str, speedup_str)

    console.print()
    console.print(table)

    # Backward Pass Table
    table2 = Table(title="[bold cyan]Backward Pass Performance[/bold cyan]", box=box.ROUNDED)
    table2.add_column("Configuration", style="cyan", justify="left")
    table2.add_column("Size", style="dim", justify="right")
    table2.add_column("JAX (ms)", style="green", justify="right")
    table2.add_column("PyTorch (ms)", style="yellow", justify="right")
    table2.add_column("Speedup", justify="right")

    for r in results:
        jax_str = f"{r.jax_backward_ms:.2f}"
        torch_str = f"{r.torch_backward_ms:.2f}" if r.torch_backward_ms else "-"
        speedup_str = format_speedup(r.speedup_backward)
        table2.add_row(r.config_name, r.size_str, jax_str, torch_str, speedup_str)

    console.print()
    console.print(table2)

    # Summary Statistics
    if any(r.speedup_forward for r in results):
        fwd_speedups = [r.speedup_forward for r in results if r.speedup_forward]
        bwd_speedups = [r.speedup_backward for r in results if r.speedup_backward]

        avg_fwd = np.mean(fwd_speedups) if fwd_speedups else 0
        avg_bwd = np.mean(bwd_speedups) if bwd_speedups else 0

        console.print()
        console.print(Panel(
            f"[bold]Average Forward Speedup:[/bold] {format_speedup(avg_fwd)}\n"
            f"[bold]Average Backward Speedup:[/bold] {format_speedup(avg_bwd)}\n\n"
            f"[dim]↑ = JAX faster | → = Similar | ↓ = PyTorch faster[/dim]",
            title="[bold]Summary[/bold]",
            border_style="green"
        ))


def main():
    if RICH_AVAILABLE:
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        grid.add_row(f"[bold cyan]KeOps Single GPU Benchmark[/bold cyan]")
        console.print(Panel(grid, style="bold white", border_style="blue"))

    results = []
    for config in BENCHMARK_CONFIGS:
        try:
            results.append(run_benchmark(config))
        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"[red]Failed {config.name}: {e}[/red]")
            else:
                print(f"Failed {config.name}: {e}")

    # --- RESTORED SUMMARY TABLE ---
    print_summary(results)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_single_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    if RICH_AVAILABLE:
        console.print(f"[dim]Results saved to {output_file}[/dim]")
    else:
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()