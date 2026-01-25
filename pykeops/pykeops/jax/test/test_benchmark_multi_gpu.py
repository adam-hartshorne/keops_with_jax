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
    time_ms: float
    speedup: float
    efficiency_pct: float


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

def benchmark_op(op, args, mesh):
    """Run benchmark with visual progress bar."""

    in_specs = []
    for i in range(len(args) - 1):
        in_specs.append(P('batch', None, None))
    in_specs.append(P(None))  # last arg is 's'

    out_specs = P('batch', None, None)

    # Compile
    @jax.jit
    @shard_map(mesh=mesh, in_specs=tuple(in_specs), out_specs=out_specs)
    def compute(x_loc, y_loc, *others):
        return op(x_loc, y_loc, *others)

    # Execution
    times = []
    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console if RICH_AVAILABLE else None,
            transient=True
    ) as progress:

        # Warmup
        task = progress.add_task("[yellow]Warming up...", total=N_WARMUP)
        for _ in range(N_WARMUP):
            res = compute(*args)
            res.block_until_ready()
            progress.advance(task)

        # Timing
        task = progress.add_task("[green]Benchmarking...", total=N_ITERS)
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


def run_scaling_test(config: ScalingConfig, gpu_counts: List[int], bg: int):
    results = []
    baseline_time = None
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

            # 2. Benchmark
            if RICH_AVAILABLE:
                console.print(f"    Running on [bold magenta]{n_gpus} GPUs[/bold magenta]...", end="\r")

            time_ms = benchmark_op(op, args, mesh)

            if RICH_AVAILABLE:
                console.print(f"    [green]✓[/green] {n_gpus} GPUs: {time_ms:.2f} ms")

            # 3. Stats
            if baseline_time is None:
                baseline_time = time_ms
                speedup = 1.0
            else:
                speedup = baseline_time / time_ms

            efficiency = (speedup / n_gpus) * 100 if n_gpus > 1 else 100.0

            results.append(
                ScalingResult(config.name, size_str, total_batch_size, bg, n_gpus, time_ms, speedup, efficiency))

        except Exception as e:
            console.print(f"[red]Error on {n_gpus} GPUs: {e}[/red]")

    return results


def print_summary_table(all_results: Dict[str, List[ScalingResult]], max_gpus):
    """Generates the single summary table requested."""
    if not RICH_AVAILABLE: return

    table = Table(title="Multi-GPU Performance Summary", box=box.ROUNDED)
    table.add_column("Problem", style="cyan", justify="left")
    table.add_column("Size", style="dim", justify="right")
    table.add_column("Batch (B)", justify="right")
    table.add_column("Single GPU (ms)", style="red", justify="right")
    table.add_column(f"Multi GPU ({max_gpus}) (ms)", style="green", justify="right")
    table.add_column("Speedup", style="bold white", justify="right")
    table.add_column("Scaling Eff.", justify="right")

    for config_name_bg, res_list in all_results.items():
        # Find 1 GPU result
        r1 = next((r for r in res_list if r.n_gpus == 1), None)
        # Find Max GPU result
        r_max = next((r for r in res_list if r.n_gpus == max_gpus), None)

        if r1 and r_max:
            speedup_str = f"{r_max.speedup:.2f}x"
            eff = r_max.efficiency_pct
            eff_color = "green" if eff > 80 else ("yellow" if eff > 50 else "red")
            eff_str = f"[{eff_color}]{eff:.0f}%[/{eff_color}]"

            # Clean Config Name (remove unique ID suffix if needed, or keep clean)
            display_name = r1.config_name

            table.add_row(
                display_name,
                r1.size_str,
                str(r1.total_batch_size),
                f"{r1.time_ms:.2f}",
                f"{r_max.time_ms:.2f}",
                speedup_str,
                eff_str
            )

    console.print()
    console.print(table)


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