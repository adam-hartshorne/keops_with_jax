import os

# --- Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax import shard_map  # <--- FIXED IMPORT for JAX 0.8.x
from pykeops.jax import Genred
import numpy as np
import time

def benchmark_run(label, fn_jit, x, y, n_iters=100):
    print(f"  > Warmup {label}...", end="", flush=True)
    _ = fn_jit(x, y)
    jax.block_until_ready(_)
    print(" Done.")

    print(f"  > Running {label} ({n_iters} iters)...", end="", flush=True)
    start = time.perf_counter()
    for _ in range(n_iters):
        res = fn_jit(x, y)
        jax.block_until_ready(res)
    end = time.perf_counter()
    print(" Done.")
    
    avg_ms = ((end - start) / n_iters) * 1000
    return avg_ms

def run_scaling_test():
    print("=" * 60)
    print("KeOps JAX: Scaling with shard_map (Explicit SPMD)")
    print("=" * 60)

    devices = jax.devices()
    n_devices = len(devices)
    print(f"Total JAX Devices: {n_devices}")
    
    if n_devices < 2:
        print("⚠️  Need at least 2 GPUs. Exiting.")
        return

    # --- 1. Define Mesh ---
    mesh = Mesh(devices, axis_names=('batch',))

    # --- 2. Define Operator ---
    D = 3
    formula = "SqDist(x, y)"
    aliases = [f"x = Vi({D})", f"y = Vj({D})"]
    op = Genred(formula, aliases, reduction_op='Sum', axis=1)

    # --- 3. Define the Distributed Function ---
    
    # Define the local computation (what happens on ONE GPU)
    def local_op(x_local, y_replicated):
        # x_local will be slice (N/4, D)
        # y_replicated will be full (N, D)
        return op(x_local, y_replicated)

    # Transform it with shard_map (Explicit SPMD)
    # This forces JAX to split input 'x' and stitch output 'res'
    distributed_op = shard_map(
        local_op,
        mesh=mesh,
        in_specs=(P('batch', None), P(None, None)), # Split x, Keep y full
        out_specs=P('batch', None)                  # Stitch result back
    )

    # JIT the distributed function
    run_multi = jax.jit(distributed_op)

    # Single GPU version (standard JIT)
    @jax.jit
    def run_single(x, y):
        return op(x, y)

    # --- Test Cases ---
    problem_sizes = [
        ("Medium N=50k", 50000),
        ("Large N=100k", 100000),
        ("Huge N=500k", 500000)
    ]

    for label, N in problem_sizes:
        print(f"\n--- Benchmarking: {label} ---")
        
        # Create Data
        x_host = np.random.randn(N, D).astype(np.float32)
        y_host = np.random.randn(N, D).astype(np.float32)

        # 1. Single GPU
        # Put data on device 0
        x_1 = jax.device_put(x_host, devices[0])
        y_1 = jax.device_put(y_host, devices[0])
        
        t_single = benchmark_run("Single GPU", run_single, x_1, y_1)
        print(f"  ⏱️  Single GPU Time: {t_single:.3f} ms")

        # 2. Multi GPU (Sharded)
        # We must create the sharded arrays so shard_map receives the correct layout
        # (jax.jit(shard_map) handles the distribution, but inputs must be compatible)
        sharding_x = jax.sharding.NamedSharding(mesh, P('batch', None))
        sharding_y = jax.sharding.NamedSharding(mesh, P(None, None))
        
        x_m = jax.device_put(x_host, sharding_x)
        y_m = jax.device_put(y_host, sharding_y)

        t_multi = benchmark_run(f"Multi-GPU ({n_devices})", run_multi, x_m, y_m)
        print(f"  ⏱️  Multi-GPU Time:  {t_multi:.3f} ms")

        # Stats
        if t_multi > 0:
            speedup = t_single / t_multi
            efficiency = (speedup / n_devices) * 100
        else:
            speedup = 0
            efficiency = 0
        
        print(f"\n  📊 Results for N={N}:")
        print(f"     Speedup:    {speedup:.2f}x")
        print(f"     Efficiency: {efficiency:.1f}%")

if __name__ == "__main__":
    run_scaling_test()
