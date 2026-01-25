#!/usr/bin/env python3
"""
KeOps JAX Multi-GPU Benchmark (Dynamic Batch Sizes)
===================================================
- Tests "Split-Batch" parallelism.
- Calculates Total Batch Size (B) based on a fixed "Batch per GPU" (bg).
  B = bg * MAX_GPUS.
- Ensures divisibility and enables testing different batch loads.
"""

import os

# Ensure we see all GPUs.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ['JAX_KEOPS_DEBUG'] = '0'

import sys
import time
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

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

# Check GPU availability
devices = jax.devices('gpu')
N_GPUS = len(devices)

try:
    from pykeops.jax import Genred

    KEOPS_AVAILABLE = True
except ImportError as e:
    print(f"pykeops.jax not found: {e}")
    sys.exit(1)

# =============================================================================
# Configuration
# =============================================================================

# We test these different "Batch per GPU" settings
# If MAX_GPUS=8:
#   bg=2  -> Total Batch=16
#   bg=5  -> Total Batch=40
#   bg=10 -> Total Batch=80
BATCHES_PER_GPU_LIST = [2, 5, 10]


@dataclass
class ScalingConfig:
    name: str
    nx: int
    ny: int
    dim: int
    formula: str
    mode: str = "standard"  # 'standard' (x,y,s) or 'varifold' (x,y,u,v,s)


@dataclass
class ScalingResult:
    config_name: str
    size_str: str
    total_batch_size: int
    bg: int
    n_gpus: int
    forward_ms: float
    backward_ms: float
    speedup_forward: float
    speedup_backward: float
    efficiency_forward_pct: float
    efficiency_backward_pct: float


# Benchmark Configurations
SCALING_CONFIGS = [
    # --- Gaussian / SqDist ---
    ScalingConfig("Medium (SqDist)", 20_000, 20_000, 3, "SqNorm2(x-y)"),
    ScalingConfig("Large (SqDist)", 50_000, 50_000, 3, "SqNorm2(x-y)"),
    ScalingConfig("XLarge (SqDist)", 100_000, 100_000, 3, "SqNorm2(x-y)"),

    # --- Varifold Kernels ---
    # Formula: Gaussian(x,y) * (u|v)^2
    ScalingConfig("Large (Varifold)", 50_000, 50_000, 3,
                  "Exp(-SqNorm2(x-y)*s) * Square((u|v))", mode="varifold"),
    ScalingConfig("XLarge (Varifold)", 100_000, 100_000, 3,
                  "Exp(-SqNorm2(x-y)*s) * Square((u|v))", mode="varifold"),
]

N_WARMUP = 5
N_ITERS = 20
TIMING_BATCH = 5  # Inner loop size


# =============================================================================
# Data Generation & Sharding
# =============================================================================

def create_sharded_data(config: ScalingConfig, n_gpus, total_batch_size, seed=42):
    """
    Creates and shards data based on the config mode.
    Splits along axis 0 (Batch) for all inputs.
    """
    if total_batch_size % n_gpus != 0:
        raise ValueError(f"Batch size {total_batch_size} cannot be split evenly by {n_gpus} GPUs.")

    np.random.seed(seed)

    B, M, N, D = total_batch_size, config.nx, config.ny, config.dim

    arrays_np = []

    # X, Y (Positions)
    arrays_np.append(np.random.randn(B, M, D).astype(np.float32))  # x
    arrays_np.append(np.random.randn(B, N, D).astype(np.float32))  # y

    if config.mode == "varifold":
        # U, V (Vectors/Normals)
        arrays_np.append(np.random.randn(B, M, D).astype(np.float32))  # u
        arrays_np.append(np.random.randn(B, N, D).astype(np.float32))  # v

    # S (Parameter - Replicated)
    s_np = np.array([0.5], dtype=np.float32)

    # 2. Define Sharding
    active_devices = jax.devices('gpu')[:n_gpus]
    mesh = Mesh(np.array(active_devices), ('batch',))

    sharded_arrays = []

    # Helper to push array to device
    def push(arr, is_param=False):
        if is_param:
            spec = P(None)  # Replicate
        else:
            spec = P('batch', None, None)  # Split dim 0

        sharding = NamedSharding(mesh, spec)
        return jax.device_put(jnp.array(arr), sharding)

    for arr in arrays_np:
        sharded_arrays.append(push(arr, is_param=False))

    sharded_arrays.append(push(s_np, is_param=True))

    return tuple(sharded_arrays), mesh


# =============================================================================
# Benchmarking Core
# =============================================================================

def benchmark_op(op, args, mesh, mode="forward"):
    """Run benchmark with visual progress bar. Mode can be 'forward' or 'backward'."""

    in_specs = []
    for i in range(len(args) - 1):
        in_specs.append(P('batch', None, None))
    in_specs.append(P(None))  # last arg is 's'

    out_specs = P('batch', None, None)

    if mode == "forward":
        # Forward pass
        @jax.jit
        @shard_map(mesh=mesh, in_specs=tuple(in_specs), out_specs=out_specs)
        def compute(x_loc, y_loc, *others):
            return op(x_loc, y_loc, *others)
    else:
        # Backward pass (gradient w.r.t. first argument)
        @jax.jit
        @shard_map(mesh=mesh, in_specs=tuple(in_specs), out_specs=in_specs[0])
        def compute(x_loc, y_loc, *others):
            def loss_fn(x):
                return op(x, y_loc, *others).sum()

            return jax.grad(loss_fn)(x_loc)

    # Execution
    times = []
    mode_str = "Forward" if mode == "forward" else "Backward"

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console if RICH_AVAILABLE else None,
            transient=True
    ) as progress:

        # Warmup
        task = progress.add_task(f"[yellow]{mode_str} Warmup...", total=N_WARMUP)
        for _ in range(N_WARMUP):
            res = compute(*args)
            res.block_until_ready()
            progress.advance(task)

        # Timing
        task = progress.add_task(f"[green]{mode_str} Benchmarking...", total=N_ITERS)
        n_loops = max(1, N_ITERS // TIMING_BATCH)

        for _ in range(n_loops):
            start = time.perf_counter()
            for _ in range(TIMING_BATCH):
                res = compute(*args)
            res.block_until_ready()
            end = time.perf_counter()

            times.append((end - start) * 1000 / TIMING_BATCH)
            progress.advance(task, advance=TIMING_BATCH)

    return np.median(times)


# =============================================================================
# Runner & Visuals
# =============================================================================

def print_header():
    if RICH_AVAILABLE:
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        grid.add_row(f"[bold cyan]KeOps JAX Multi-GPU Benchmark[/bold cyan]")
        grid.add_row(f"[dim]Dynamic Batch Calculation: B = bg * {N_GPUS}[/dim]")
        console.print(Panel(grid, style="bold white", border_style="blue"))


def format_speedup(speedup):
    """Format speedup with arrow and color."""
    if speedup is None:
        return "-"
    if speedup > 1.05:
        return f"[bold green]↑ {speedup:.2f}x[/bold green]"  # Faster
    elif speedup < 0.95:
        return f"[bold red]↓ {speedup:.2f}x[/bold red]"  # Slower
    else:
        return f"[bold yellow]→ {speedup:.2f}x[/bold yellow]"  # About the same


def format_efficiency(eff_pct):
    """Format efficiency with color."""
    if eff_pct is None:
        return "-"
    if eff_pct >= 80:
        return f"[bold green]{eff_pct:.0f}%[/bold green]"
    elif eff_pct >= 50:
        return f"[bold yellow]{eff_pct:.0f}%[/bold yellow]"
    else:
        return f"[bold red]{eff_pct:.0f}%[/bold red]"


def run_scaling_test(config: ScalingConfig, gpu_counts: List[int], bg: int):
    results = []
    baseline_forward = None
    baseline_backward = None
    size_str = f"N={config.nx:,}"

    # Calculate Total Batch Size
    total_batch_size = bg * N_GPUS  # Use Max GPUs to determine B

    # KeOps Operator Setup
    D = config.dim
    if config.mode == "varifold":
        aliases = [f"x=Vi({D})", f"y=Vj({D})", f"u=Vi({D})", f"v=Vj({D})", "s=Pm(1)"]
    else:
        aliases = [f"x=Vi({D})", f"y=Vj({D})", "s=Pm(1)"]

    op = Genred(config.formula, aliases, reduction_op='Sum', axis=1)

    if RICH_AVAILABLE:
        console.print(f"[bold]  Batch Size B={total_batch_size} (bg={bg})[/bold]")

    for n_gpus in gpu_counts:
        try:
            # 1. Prepare Data
            args, mesh = create_sharded_data(config, n_gpus, total_batch_size)

            # 2. Benchmark Forward
            if RICH_AVAILABLE:
                console.print(f"    Running on [bold magenta]{n_gpus} GPUs[/bold magenta]...", end="\r")

            forward_ms = benchmark_op(op, args, mesh, mode="forward")

            # 3. Benchmark Backward
            backward_ms = benchmark_op(op, args, mesh, mode="backward")

            # Format arrows for inline display (with colors)
            def inline_speedup(speedup):
                if speedup > 1.05:
                    return f"[bold green]↑{speedup:.2f}x[/bold green]"
                elif speedup < 0.95:
                    return f"[bold red]↓{speedup:.2f}x[/bold red]"
                else:
                    return f"[bold yellow]→{speedup:.2f}x[/bold yellow]"

            if baseline_forward is None:
                fwd_str = "[dim]baseline[/dim]"
                bwd_str = "[dim]baseline[/dim]"
            else:
                speedup_fwd_tmp = baseline_forward / forward_ms
                speedup_bwd_tmp = baseline_backward / backward_ms
                fwd_str = inline_speedup(speedup_fwd_tmp)
                bwd_str = inline_speedup(speedup_bwd_tmp)

            if RICH_AVAILABLE:
                console.print(
                    f"    [green]✓[/green] {n_gpus} GPUs: Fwd {forward_ms:.2f}ms ({fwd_str}), Bwd {backward_ms:.2f}ms ({bwd_str})")

            # 4. Stats
            if baseline_forward is None:
                baseline_forward = forward_ms
                baseline_backward = backward_ms
                speedup_forward = 1.0
                speedup_backward = 1.0
            else:
                speedup_forward = baseline_forward / forward_ms
                speedup_backward = baseline_backward / backward_ms

            efficiency_forward = (speedup_forward / n_gpus) * 100 if n_gpus > 1 else 100.0
            efficiency_backward = (speedup_backward / n_gpus) * 100 if n_gpus > 1 else 100.0

            results.append(
                ScalingResult(
                    config_name=config.name,
                    size_str=size_str,
                    total_batch_size=total_batch_size,
                    bg=bg,
                    n_gpus=n_gpus,
                    forward_ms=forward_ms,
                    backward_ms=backward_ms,
                    speedup_forward=speedup_forward,
                    speedup_backward=speedup_backward,
                    efficiency_forward_pct=efficiency_forward,
                    efficiency_backward_pct=efficiency_backward,
                ))

        except Exception as e:
            console.print(f"[red]Error on {n_gpus} GPUs: {e}[/red]")
            import traceback
            traceback.print_exc()

    return results


def print_summary_table(all_results: Dict[str, List[ScalingResult]], max_gpus):
    """Generates summary tables for forward and backward passes."""
    if not RICH_AVAILABLE: return

    # Forward Pass Table
    table_fwd = Table(title="[bold cyan]Forward Pass - Multi-GPU Performance[/bold cyan]", box=box.ROUNDED)
    table_fwd.add_column("Problem", style="cyan", justify="left")
    table_fwd.add_column("Size", style="dim", justify="right")
    table_fwd.add_column("Batch (B)", justify="right")
    table_fwd.add_column("1 GPU (ms)", style="yellow", justify="right")
    table_fwd.add_column(f"{max_gpus} GPU (ms)", style="green", justify="right")
    table_fwd.add_column("Speedup", justify="right")
    table_fwd.add_column("Efficiency", justify="right")

    for config_name_bg, res_list in all_results.items():
        r1 = next((r for r in res_list if r.n_gpus == 1), None)
        r_max = next((r for r in res_list if r.n_gpus == max_gpus), None)

        if r1 and r_max:
            table_fwd.add_row(
                r1.config_name,
                r1.size_str,
                str(r1.total_batch_size),
                f"{r1.forward_ms:.2f}",
                f"{r_max.forward_ms:.2f}",
                format_speedup(r_max.speedup_forward),
                format_efficiency(r_max.efficiency_forward_pct),
            )

    console.print()
    console.print(table_fwd)

    # Backward Pass Table
    table_bwd = Table(title="[bold cyan]Backward Pass - Multi-GPU Performance[/bold cyan]", box=box.ROUNDED)
    table_bwd.add_column("Problem", style="cyan", justify="left")
    table_bwd.add_column("Size", style="dim", justify="right")
    table_bwd.add_column("Batch (B)", justify="right")
    table_bwd.add_column("1 GPU (ms)", style="yellow", justify="right")
    table_bwd.add_column(f"{max_gpus} GPU (ms)", style="green", justify="right")
    table_bwd.add_column("Speedup", justify="right")
    table_bwd.add_column("Efficiency", justify="right")

    for config_name_bg, res_list in all_results.items():
        r1 = next((r for r in res_list if r.n_gpus == 1), None)
        r_max = next((r for r in res_list if r.n_gpus == max_gpus), None)

        if r1 and r_max:
            table_bwd.add_row(
                r1.config_name,
                r1.size_str,
                str(r1.total_batch_size),
                f"{r1.backward_ms:.2f}",
                f"{r_max.backward_ms:.2f}",
                format_speedup(r_max.speedup_backward),
                format_efficiency(r_max.efficiency_backward_pct),
            )

    console.print()
    console.print(table_bwd)

    # Summary Statistics
    fwd_efficiencies = []
    bwd_efficiencies = []
    for res_list in all_results.values():
        r_max = next((r for r in res_list if r.n_gpus == max_gpus), None)
        if r_max:
            fwd_efficiencies.append(r_max.efficiency_forward_pct)
            bwd_efficiencies.append(r_max.efficiency_backward_pct)

    if fwd_efficiencies:
        avg_fwd_eff = np.mean(fwd_efficiencies)
        avg_bwd_eff = np.mean(bwd_efficiencies)

        console.print()
        console.print(Panel(
            f"[bold]Average Forward Scaling Efficiency ({max_gpus} GPUs):[/bold] {format_efficiency(avg_fwd_eff)}\n"
            f"[bold]Average Backward Scaling Efficiency ({max_gpus} GPUs):[/bold] {format_efficiency(avg_bwd_eff)}\n\n"
            f"[dim]↑ = Scaling well | → = Linear | ↓ = Sub-linear scaling[/dim]\n"
            f"[dim]Efficiency: Green ≥80% | Yellow ≥50% | Red <50%[/dim]",
            title="[bold]Summary[/bold]",
            border_style="green"
        ))


def main():
    print_header()

    if N_GPUS < 1:
        console.print("[bold red]No GPUs found![/bold red]")
        return

    # We test 1 GPU (Baseline) and then powers of 2 up to Max
    gpu_counts = [1, 2, 4, 8]
    gpu_counts = [g for g in gpu_counts if g <= N_GPUS]

    all_results = {}

    for config in SCALING_CONFIGS:
        console.rule(f"[bold blue]{config.name} (N={config.nx:,})[/bold blue]")

        # Iterate over different Batch-per-GPU settings
        for bg in BATCHES_PER_GPU_LIST:
            results = run_scaling_test(config, gpu_counts, bg)
            # Store with unique key to separate bg in dictionary
            key = f"{config.name}_bg{bg}"
            all_results[key] = results

        console.print()

    # Final Summary Table
    print_summary_table(all_results, max(gpu_counts))

    # Save Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_multigpu_{timestamp}.json"
    with open(output_file, 'w') as f:
        # Convert keys to string for JSON compatibility
        data = {k: [asdict(r) for r in v] for k, v in all_results.items()}
        json.dump(data, f, indent=2)
    console.print(f"[dim]Results saved to {output_file}[/dim]")


if __name__ == "__main__":
    main()