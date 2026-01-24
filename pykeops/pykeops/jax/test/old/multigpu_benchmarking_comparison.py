import os

# --- Configuration ---
# Ensure we see all GPUs. Adjust indices if you have fewer than 8.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' # Optional: Use if hitting OOM

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax import shard_map
from pykeops.jax import Genred

# --- Rich UI Setup ---
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich import box

    console = Console()
except ImportError:
    print("Please install 'rich' for better graphics: pip install rich")
    exit()


# --- Helper: Timing Function ---
def benchmark_with_spinner(label, fn, args, n_iters=20):
    """Runs benchmark with a UI spinner and timing."""
    # Warmup
    fn(*args).block_until_ready()

    start = time.perf_counter()
    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
    ) as progress:
        progress.add_task(description=f"Running {label} ({n_iters} iters)...", total=None)
        for _ in range(n_iters):
            fn(*args).block_until_ready()

    end = time.perf_counter()
    avg_time = ((end - start) / n_iters) * 1000
    return avg_time


def check_correctness(single_res, multi_res):
    """Checks if single and multi-gpu results are close."""
    # Move to host for comparison
    s = np.array(single_res)
    m = np.array(multi_res)

    # Check relative error
    # We add a small epsilon to denominator to avoid division by zero
    diff = np.linalg.norm(s - m) / (np.linalg.norm(s) + 1e-6)
    passed = diff < 1e-4

    msg = f"[green]PASS[/green] (RelErr: {diff:.2e})" if passed else f"[bold red]FAIL[/bold red] (RelErr: {diff:.2e})"
    return msg


def run_full_comparison():
    console.print(Panel.fit("[bold magenta]KeOps JAX: Multi-GPU Benchmark[/bold magenta]", border_style="magenta"))

    devices = jax.devices()
    n_devices = len(devices)
    console.print(f"⚡ [bold]Total Devices Found:[/bold] {n_devices}")

    if n_devices < 2:
        console.print("[bold red]⚠️  Need at least 2 GPUs for this test.[/bold red]")
        return

    # --- 1. Define Operations ---
    D = 3
    formula = "SqDist(x, y)"
    aliases = [f"x = Vi({D})", f"y = Vj({D})"]
    op = Genred(formula, aliases, reduction_op='Sum', axis=1)

    # --- 2. Single GPU Implementation ---
    @jax.jit
    def single_fwd(x, y):
        return op(x, y)

    @jax.jit
    def single_bwd(x, y):
        # We define loss as the sum of the reduction
        loss = lambda x, y: jnp.sum(op(x, y))
        return jax.grad(loss)(x, y)

    # --- 3. Multi GPU Implementation ---
    mesh = Mesh(devices, axis_names=('batch',))

    # Local computation for shard_map
    def local_fwd_fn(x_loc, y_repl):
        return op(x_loc, y_repl)

    # Distributed Forward
    @jax.jit
    @shard_map(mesh=mesh, in_specs=(P('batch', None), P(None, None)), out_specs=P('batch', None))
    def multi_fwd(x_loc, y_repl):
        return local_fwd_fn(x_loc, y_repl)

    # Distributed Backward Logic
    @shard_map(mesh=mesh, in_specs=(P('batch', None), P(None, None)), out_specs=P())
    def multi_loss_fn(x_loc, y_repl):
        # 1. Compute local sum
        local_sum = jnp.sum(op(x_loc, y_repl))

        # 2. Sum across all GPUs (global reduce)
        global_loss = jax.lax.psum(local_sum, 'batch')

        # 3. [CRITICAL FIX] Normalize by mesh size.
        # jax.grad sums the output across devices.
        # If we return L on every device, grad sees (L * n_devices).
        # We must return (L / n_devices) so sum equals L.
        return global_loss / jax.lax.psum(1, 'batch')

    @jax.jit
    def multi_bwd(x, y):
        return jax.grad(multi_loss_fn)(x, y)

    # --- 4. Test Loop ---
    problem_sizes = [
        ("Small", 10_000),
        ("Medium", 50_000),
        ("Huge", 500_000),
        # ("Massive", 1_000_000)
    ]

    results_summary = []

    for label, N in problem_sizes:
        console.rule(f"[bold cyan]Testing: {label} (N={N:,})[/bold cyan]")

        # Data Creation
        with console.status("[dim]Generating & Sharding Data...[/dim]"):
            x_host = np.random.randn(N, D).astype(np.float32)
            y_host = np.random.randn(N, D).astype(np.float32)

            # Single GPU Data (Device 0)
            x_s = jax.device_put(x_host, devices[0])
            y_s = jax.device_put(y_host, devices[0])

            # Multi GPU Data (Sharded X, Replicated Y)
            shard_x = NamedSharding(mesh, P('batch', None))
            repl_y = NamedSharding(mesh, P(None, None))
            x_m = jax.device_put(x_host, shard_x)
            y_m = jax.device_put(y_host, repl_y)

        # Correctness Checks
        console.print("   [dim]Verifying numerical correctness...[/dim]", end=" ")

        # Check Output
        fwd_check = check_correctness(single_fwd(x_s, y_s), multi_fwd(x_m, y_m))

        # Check Gradients
        # Note: single_bwd returns a tuple (grad_x, grad_y). We are only checking grad_x here.
        grads_s = single_bwd(x_s, y_s)
        grads_m = multi_bwd(x_m, y_m)
        bwd_check = check_correctness(grads_s[0], grads_m[0])

        console.print(f"Fwd: {fwd_check} | Bwd: {bwd_check}")

        # Benchmark Forward
        t_fwd_s = benchmark_with_spinner("Forward (Single)", single_fwd, (x_s, y_s))
        t_fwd_m = benchmark_with_spinner("Forward (Multi)", multi_fwd, (x_m, y_m))
        speedup_fwd = t_fwd_s / t_fwd_m

        # Benchmark Backward
        t_bwd_s = benchmark_with_spinner("Backward (Single)", single_bwd, (x_s, y_s))
        t_bwd_m = benchmark_with_spinner("Backward (Multi)", multi_bwd, (x_m, y_m))
        speedup_bwd = t_bwd_s / t_bwd_m

        # Store for final summary
        results_summary.append({
            "Size": f"{label} (N={N:,})",
            "Fwd Single": t_fwd_s, "Fwd Multi": t_fwd_m, "Fwd Speed": speedup_fwd,
            "Bwd Single": t_bwd_s, "Bwd Multi": t_bwd_m, "Bwd Speed": speedup_bwd,
        })

        # Mini Output
        console.print(f"   [bold]Results for {label}:[/bold]")
        console.print(
            f"   Forward:  [blue]{t_fwd_s:.1f}ms[/blue] -> [green]{t_fwd_m:.1f}ms[/green] (x{speedup_fwd:.2f})")
        console.print(
            f"   Backward: [blue]{t_bwd_s:.1f}ms[/blue] -> [green]{t_bwd_m:.1f}ms[/green] (x{speedup_bwd:.2f})")
        print()

    # --- 5. Final Consolidated Summary ---
    console.print("\n")
    console.rule("[bold green]FINAL BENCHMARK SUMMARY[/bold green]")

    table = Table(box=box.ROUNDED)
    table.add_column("Problem Size", justify="left", style="cyan", no_wrap=True)
    table.add_column("Op Type", style="magenta")
    table.add_column("Single GPU (ms)", justify="right", style="red")
    table.add_column("Multi GPU (ms)", justify="right", style="green")
    table.add_column("Speedup", justify="right", style="bold yellow")
    table.add_column("Scaling Eff.", justify="right")

    for row in results_summary:
        # Forward Row
        eff_fwd = (row['Fwd Speed'] / n_devices) * 100
        table.add_row(
            row['Size'], "Forward",
            f"{row['Fwd Single']:.1f}", f"{row['Fwd Multi']:.1f}",
            f"{row['Fwd Speed']:.2f}x", f"{eff_fwd:.0f}%"
        )
        # Backward Row
        eff_bwd = (row['Bwd Speed'] / n_devices) * 100
        table.add_row(
            "", "Backward",
            f"{row['Bwd Single']:.1f}", f"{row['Bwd Multi']:.1f}",
            f"{row['Bwd Speed']:.2f}x", f"{eff_bwd:.0f}%"
        )
        table.add_section()

    console.print(table)
    console.print(f"[dim]Scaling Efficiency = (Speedup / {n_devices} GPUs) * 100[/dim]")


if __name__ == "__main__":
    run_full_comparison()
