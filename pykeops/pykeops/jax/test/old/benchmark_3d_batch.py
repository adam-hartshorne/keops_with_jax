import os

# --- Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7" 
os.environ['PYKEOPS_JAX_MODE'] = '1'

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
    
    avg = ((end - start) / n_iters) * 1000
    print(f" {avg:.3f} ms")
    return avg

def run_3d_benchmark_sizes():
    print("=" * 70)
    print("KeOps JAX: 3D Batch (B, N, D) Scaling Analysis")
    print("=" * 70)

    devices = jax.devices()
    n_devices = len(devices)
    print(f"Total Devices: {n_devices}")
    
    if n_devices < 2:
        print("⚠️  Need at least 2 GPUs.")
        return

    # --- 1. Operations ---
    D = 3
    formula = "SqDist(x, y)"
    aliases = [f"x = Vi({D})", f"y = Vj({D})"]
    op = Genred(formula, aliases, reduction_op='Sum', axis=1)

    # --- 2. Single GPU Functions ---
    @jax.jit
    def single_fwd(x, y):
        return op(x, y)

    @jax.jit
    def single_bwd(x, y):
        loss = lambda x, y: jnp.sum(op(x, y))
        return jax.grad(loss)(x, y)

    # --- 3. Multi GPU Functions (Sharded Batch) ---
    mesh = Mesh(devices, axis_names=('data',))
    
    @jax.jit
    @shard_map(mesh=mesh, 
               in_specs=(P('data', None, None), P('data', None, None)), 
               out_specs=P('data', None, None))
    def multi_fwd(x_loc, y_loc):
        return op(x_loc, y_loc)

    @shard_map(mesh=mesh, 
               in_specs=(P('data', None, None), P('data', None, None)), 
               out_specs=P())
    def multi_loss_fn(x_loc, y_loc):
        return jax.lax.psum(jnp.sum(op(x_loc, y_loc)), 'data')

    @jax.jit
    def multi_bwd(x, y):
        return jax.grad(multi_loss_fn)(x, y)

    # --- 4. Test Loop ---
    # We keep Batch Size constant and vary N (Trajectory Length)
    B = 128
    trajectory_sizes = [2000, 20000, 50000]

    for N in trajectory_sizes:
        print(f"\n\n{'-'*60}")
        print(f"SCENARIO: Batch={B}, Length={N}, Dim={D}")
        print(f"{'-'*60}")
        
        # Data Gen
        x_host = np.random.randn(B, N, D).astype(np.float32)
        y_host = np.random.randn(B, N, D).astype(np.float32)

        # Distribution
        x_s = jax.device_put(x_host, devices[0])
        y_s = jax.device_put(y_host, devices[0])
        
        shard_spec = NamedSharding(mesh, P('data', None, None))
        x_m = jax.device_put(x_host, shard_spec)
        y_m = jax.device_put(y_host, shard_spec)

        # Benchmarks
        print(f"\n[Forward Pass]")
        t_fwd_s = benchmark("Single", single_fwd, (x_s, y_s))
        t_fwd_m = benchmark("Multi ", multi_fwd,  (x_m, y_m))
        
        print(f"\n[Backward Pass]")
        t_bwd_s = benchmark("Single", single_bwd, (x_s, y_s))
        t_bwd_m = benchmark("Multi ", multi_bwd,  (x_m, y_m))

        # Results
        sf = t_fwd_s / t_fwd_m
        sb = t_bwd_s / t_bwd_m
        
        print(f"\n>>> Results (N={N}):")
        print(f"    Forward:  {t_fwd_s:.2f}ms -> {t_fwd_m:.2f}ms (Speedup: {sf:.2f}x)")
        print(f"    Backward: {t_bwd_s:.2f}ms -> {t_bwd_m:.2f}ms (Speedup: {sb:.2f}x)")
        
        eff = (sb / n_devices) * 100
        if eff > 85: print("    ✅ Scaling: Excellent")
        elif eff > 50: print("    ⚠️ Scaling: Moderate (Overhead limited)")
        else: print("    ❌ Scaling: Poor (Latency bound)")

if __name__ == "__main__":
    run_3d_benchmark_sizes()
