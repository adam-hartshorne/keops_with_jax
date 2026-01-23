import os

# --- Configuration ---
# Ensure we see all 4 GPUs (or however many you have)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7" 


import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax import shard_map
from pykeops.jax import Genred
import numpy as np
import time

# --- Helper: Timing Function ---
def benchmark(label, fn, args, n_iters=20):
    # Warmup
    print(f"  > Warmup {label}...", end="", flush=True)
    res = fn(*args)
    jax.block_until_ready(res)
    print(" Done.")

    # Measure
    print(f"  > Running {label} ({n_iters} iters)...", end="", flush=True)
    start = time.perf_counter()
    for _ in range(n_iters):
        res = fn(*args)
        jax.block_until_ready(res)
    end = time.perf_counter()
    print(" Done.")
    
    return ((end - start) / n_iters) * 1000

def run_full_comparison():
    print("=" * 70)
    print("KeOps JAX: Full Single vs Multi-GPU Comparison (Forward & Backward)")
    print("=" * 70)

    devices = jax.devices()
    n_devices = len(devices)
    print(f"Total Devices: {n_devices}")
    
    if n_devices < 2:
        print("⚠️  Need at least 2 GPUs for this test.")
        return

    # --- 1. Define Operations ---
    D = 3
    formula = "SqDist(x, y)"
    aliases = [f"x = Vi({D})", f"y = Vj({D})"]
    op = Genred(formula, aliases, reduction_op='Sum', axis=1)

    # --- 2. Single GPU Implementation ---
    @jax.jit
    def single_fwd(x, y):
        # Returns (N, 1)
        return op(x, y)

    @jax.jit
    def single_bwd(x, y):
        # Sum output to scalar to allow simple grad w.r.t x
        loss = lambda x, y: jnp.sum(op(x, y))
        return jax.grad(loss)(x, y)

    # --- 3. Multi GPU Implementation (shard_map) ---
    mesh = Mesh(devices, axis_names=('batch',))
    
    # Local computation for shard_map
    def local_fwd_fn(x_loc, y_repl):
        return op(x_loc, y_repl)

    # Distributed Forward
    @jax.jit
    @shard_map(mesh=mesh, in_specs=(P('batch', None), P(None, None)), out_specs=P('batch', None))
    def multi_fwd(x_loc, y_repl):
        return local_fwd_fn(x_loc, y_repl)

    # Distributed Backward logic
    # We need a forward pass that outputs a SCALAR for jax.grad to work
    @shard_map(mesh=mesh, in_specs=(P('batch', None), P(None, None)), out_specs=P())
    def multi_loss_fn(x_loc, y_repl):
        # Compute local sum
        local_sum = jnp.sum(op(x_loc, y_repl))
        # Sum across all GPUs (global reduce)
        return jax.lax.psum(local_sum, 'batch')

    @jax.jit
    def multi_bwd(x, y):
        return jax.grad(multi_loss_fn)(x, y)


    # --- 4. Test Loop ---
    problem_sizes = [
        ("Medium (N=50k)", 50000),
        ("Huge   (N=500k)", 500000)
    ]

    for label, N in problem_sizes:
        print(f"\n\n--- TEST CASE: {label} ---")
        
        # Data Creation
        print("Generating Data...", end=" ")
        x_host = np.random.randn(N, D).astype(np.float32)
        y_host = np.random.randn(N, D).astype(np.float32)
        print("Done.")

        # --- SINGLE GPU DATA (Force to Device 0) ---
        x_s = jax.device_put(x_host, devices[0])
        y_s = jax.device_put(y_host, devices[0])

        # --- MULTI GPU DATA (Shard X, Replicate Y) ---
        shard_x = NamedSharding(mesh, P('batch', None))
        repl_y  = NamedSharding(mesh, P(None, None))
        x_m = jax.device_put(x_host, shard_x)
        y_m = jax.device_put(y_host, repl_y)

        # ----------------------------------------
        # Benchmarking Forward
        # ----------------------------------------
        print(f"\n[Forward Pass JIT]")
        t_fwd_s = benchmark("Single GPU", single_fwd, (x_s, y_s))
        t_fwd_m = benchmark("Multi  GPU", multi_fwd,  (x_m, y_m))
        
        speedup_fwd = t_fwd_s / t_fwd_m
        print(f"   >>> Forward Speedup: {speedup_fwd:.2f}x")

        # ----------------------------------------
        # Benchmarking Backward
        # ----------------------------------------
        print(f"\n[Backward Pass Grad+JIT]")
        t_bwd_s = benchmark("Single GPU", single_bwd, (x_s, y_s))
        t_bwd_m = benchmark("Multi  GPU", multi_bwd,  (x_m, y_m))
        
        speedup_bwd = t_bwd_s / t_bwd_m
        print(f"   >>> Backward Speedup: {speedup_bwd:.2f}x")

        # ----------------------------------------
        # Summary Table
        # ----------------------------------------
        print(f"\n{'-'*40}")
        print(f"SUMMARY ({label})")
        print(f"{'-'*40}")
        print(f"{'Type':<10} | {'Single (ms)':<12} | {'Multi (ms)':<12} | {'Speedup':<8}")
        print(f"{'-'*40}")
        print(f"{'Forward':<10} | {t_fwd_s:<12.3f} | {t_fwd_m:<12.3f} | {speedup_fwd:.2f}x")
        print(f"{'Backward':<10} | {t_bwd_s:<12.3f} | {t_bwd_m:<12.3f} | {speedup_bwd:.2f}x")
        print(f"{'-'*40}")

if __name__ == "__main__":
    run_full_comparison()
