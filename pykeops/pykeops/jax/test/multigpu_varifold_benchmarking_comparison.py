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
from pykeops.jax import LazyTensor

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


# --- Kernel Definition (LazyTensor) ---
def varifold_kernel_lazytensor(xx, yy, uu, vv, gamma, gamma_1):
    if xx.ndim == 3:
        x = LazyTensor(xx[:, :, None, :])
        y = LazyTensor(yy[:, None, :, :])
        u = LazyTensor(uu[:, :, None, :])
        v = LazyTensor(vv[:, None, :, :])
        reduction_axis = 2
    else:
        x = LazyTensor(xx[:, None, :])
        y = LazyTensor(yy[None, :, :])
        u = LazyTensor(uu[:, None, :])
        v = LazyTensor(vv[None, :, :])
        reduction_axis = 1

    D2 = x.sqdist(y)
    ss = (u * v).sum(-1)
    K_spatial = (-D2 * gamma).exp()
    K_normal = (ss * gamma_1).exp()
    return (K_spatial * K_normal).sum(axis=reduction_axis)


def run_full_comparison():
    console.print(Panel.fit("[bold magenta]KeOps JAX: Multi-GPU Batched Benchmark (LazyTensor)[/bold magenta]",
                            border_style="magenta"))

    devices = jax.devices()
    n_devices = len(devices)
    console.print(f"⚡ [bold]Total Devices Found:[/bold] {n_devices}")

    if n_devices < 2:
        console.print("[bold red]⚠️  Need at least 2 GPUs for this test.[/bold red]")
        return

    # --- Batch Config ---
    # B = 10 * Number of GPUs
    B = 10 * n_devices
    D = 3
    console.print(f"📦 [bold]Batch Size (B):[/bold] {B} (Distributed as {B // n_devices} per GPU)")
    console.print(f"🔧 [bold]Kernel:[/bold] Varifold (LazyTensor)")

    # --- 1. Single GPU Implementation ---
    # We use a single GPU to process the WHOLE batch to check correctness
    @jax.jit
    def single_fwd(x, y, u, v, g, g1):
        return varifold_kernel_lazytensor(x, y, u, v, g, g1)

    @jax.jit
    def single_bwd(x, y, u, v, g, g1):
        loss = lambda x, y, u, v, g, g1: jnp.sum(varifold_kernel_lazytensor(x, y, u, v, g, g1))
        return jax.grad(loss, argnums=0)(x, y, u, v, g, g1)

    # --- 2. Multi GPU Implementation ---
    mesh = Mesh(devices, axis_names=('batch',))

    # Local computation for shard_map (processes local batch shard)
    def local_fwd_fn(x_l, y_l, u_l, v_l, g_r, g1_r):
        return varifold_kernel_lazytensor(x_l, y_l, u_l, v_l, g_r, g1_r)

    # Distributed Forward
    # Inputs: Arrays sharded on Batch axis. Scalars replicated.
    # Output: Array sharded on Batch axis.
    @jax.jit
    @shard_map(
        mesh=mesh,
        in_specs=(P('batch', None, None), P('batch', None, None), P('batch', None, None), P('batch', None, None),
                  P(None), P(None)),
        out_specs=P('batch', None, None)
    )
    def multi_fwd(x_l, y_l, u_l, v_l, g_r, g1_r):
        return local_fwd_fn(x_l, y_l, u_l, v_l, g_r, g1_r)

    # Distributed Backward
    # Loss: Sum over everything.
    # shard_map output: Scalar (P())
    @shard_map(
        mesh=mesh,
        in_specs=(P('batch', None, None), P('batch', None, None), P('batch', None, None), P('batch', None, None),
                  P(None), P(None)),
        out_specs=P()
    )
    def multi_loss_fn(x_l, y_l, u_l, v_l, g_r, g1_r):
        # Local sum over the local batch chunk
        local_sum = jnp.sum(varifold_kernel_lazytensor(x_l, y_l, u_l, v_l, g_r, g1_r))

        # Global sum
        global_loss = jax.lax.psum(local_sum, 'batch')

        # Normalize for grad
        return global_loss / jax.lax.psum(1, 'batch')

    @jax.jit
    def multi_bwd(x, y, u, v, g, g1):
        return jax.grad(multi_loss_fn, argnums=0)(x, y, u, v, g, g1)

    # --- 3. Test Loop ---
    # M sizes (number of points)
    problem_sizes = [
        ("Small", 10_000),
        ("Medium", 50_000),
        ("Huge", 100_000),
    ]

    results_summary = []

    for label, M in problem_sizes:
        console.rule(f"[bold cyan]Testing: {label} (Batch={B}, M={M:,}, D={D})[/bold cyan]")

        # Data Creation
        with console.status("[dim]Generating & Sharding Data...[/dim]"):
            # Generate Full Batch on Host
            # Shape: (B, M, D)
            shape = (B, M, D)
            x_host = np.random.randn(*shape).astype(np.float32)
            y_host = np.random.randn(*shape).astype(np.float32)

            u_host = np.random.randn(*shape).astype(np.float32)
            u_host /= np.linalg.norm(u_host, axis=2, keepdims=True)
            v_host = np.random.randn(*shape).astype(np.float32)
            v_host /= np.linalg.norm(v_host, axis=2, keepdims=True)

            sigma_s, sigma_n = 0.5, 0.75
            gamma_val = np.array([1.0 / (2 * sigma_s ** 2)], dtype=np.float32)
            gamma1_val = np.array([1.0 / (sigma_n ** 2)], dtype=np.float32)

            # Single GPU Args (Process whole batch on device 0 for verification)
            # Warning: For "Huge", single GPU memory might be tight, but KeOps is memory efficient.
            args_s = [
                jax.device_put(d, devices[0])
                for d in [x_host, y_host, u_host, v_host, gamma_val, gamma1_val]
            ]

            # Multi GPU Args
            # Arrays: Sharded on axis 0 (Batch)
            # Scalars: Replicated
            shard_spec = NamedSharding(mesh, P('batch', None, None))
            repl_spec = NamedSharding(mesh, P(None))

            x_m = jax.device_put(x_host, shard_spec)
            y_m = jax.device_put(y_host, shard_spec)
            u_m = jax.device_put(u_host, shard_spec)
            v_m = jax.device_put(v_host, shard_spec)
            g_m = jax.device_put(gamma_val, repl_spec)
            g1_m = jax.device_put(gamma1_val, repl_spec)

            args_m = (x_m, y_m, u_m, v_m, g_m, g1_m)

        # Correctness Checks
        console.print("   [dim]Verifying numerical correctness...[/dim]", end=" ")

        # Check Output
        res_s = single_fwd(*args_s)
        res_m = multi_fwd(*args_m)
        fwd_check = check_correctness(res_s, res_m)

        # Check Gradients
        grad_s = single_bwd(*args_s)
        grad_m = multi_bwd(*args_m)
        bwd_check = check_correctness(grad_s, grad_m)

        console.print(f"Fwd: {fwd_check} | Bwd: {bwd_check}")

        # Benchmark Forward
        t_fwd_s = benchmark_with_spinner("Forward (Single)", single_fwd, args_s)
        t_fwd_m = benchmark_with_spinner("Forward (Multi)", multi_fwd, args_m)
        speedup_fwd = t_fwd_s / t_fwd_m

        # Benchmark Backward
        t_bwd_s = benchmark_with_spinner("Backward (Single)", single_bwd, args_s)
        t_bwd_m = benchmark_with_spinner("Backward (Multi)", multi_bwd, args_m)
        speedup_bwd = t_bwd_s / t_bwd_m

        # Store for final summary
        results_summary.append({
            "Size": f"{label} (M={M:,})",
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

    # --- 4. Final Consolidated Summary ---
    console.print("\n")
    console.rule("[bold green]FINAL BENCHMARK SUMMARY[/bold green]")

    table = Table(box=box.ROUNDED)
    table.add_column("Problem Size (M)", justify="left", style="cyan", no_wrap=True)
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