import os

# --- Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['PYKEOPS_JAX_MODE'] = '1'

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax import shard_map
from pykeops.jax import Genred
import numpy as np
import time

def benchmark_run(label, fn, args, n_iters=50):
    print(f"  > Warmup {label}...", end="", flush=True)
    res = fn(*args)
    jax.block_until_ready(res)
    print(" Done.")

    print(f"  > Running {label} ({n_iters} iters)...", end="", flush=True)
    start = time.perf_counter()
    for _ in range(n_iters):
        res = fn(*args)
        jax.block_until_ready(res)
    end = time.perf_counter()
    print(" Done.")
    
    avg_ms = ((end - start) / n_iters) * 1000
    return avg_ms

def run_grad_benchmark():
    print("=" * 60)
    print("KeOps JAX: Gradient (VJP) Benchmark with shard_map")
    print("=" * 60)

    devices = jax.devices()
    n_devices = len(devices)
    print(f"Total JAX Devices: {n_devices}")
    
    if n_devices < 2:
        print("⚠️  Need at least 2 GPUs. Exiting.")
        return

    # --- 1. Define Mesh & Sharding ---
    mesh = Mesh(devices, axis_names=('batch',))
    shard_x = jax.sharding.NamedSharding(mesh, P('batch', None))
    repl_y  = jax.sharding.NamedSharding(mesh, P(None, None))

    # --- 2. Define Operator ---
    D = 3
    formula = "SqDist(x, y)"
    aliases = [f"x = Vi({D})", f"y = Vj({D})"]
    op = Genred(formula, aliases, reduction_op='Sum', axis=1)

    # --- 3. Define Distributed Logic ---
    
    # Local Forward Function (per GPU)
    def local_op(x_local, y_repl):
        # We sum the output to get a scalar loss for easy gradient check
        return jnp.sum(op(x_local, y_repl))

    # Sharded Forward Function
    @jax.jit
    @shard_map(mesh=mesh, in_specs=(P('batch', None), P(None, None)), out_specs=P())
    def distributed_forward(x_local, y_repl):
        # The output of local_op is a scalar (per shard).
        # shard_map will effectively return a P('batch') of scalars (one per GPU).
        # To get a global scalar loss, we must sum these up.
        local_sum = local_op(x_local, y_repl)
        return jax.lax.psum(local_sum, 'batch')

    # Sharded Gradient Function
    # We differentiate w.r.t 'x'
    distributed_grad = jax.jit(jax.grad(distributed_forward, argnums=0))

    # --- Test Cases ---
    # We use larger sizes because gradients are compute-heavy
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

        # Distribute Data
        x_m = jax.device_put(x_host, shard_x)
        y_m = jax.device_put(y_host, repl_y)

        # 1. Forward Pass
        t_fwd = benchmark_run("Forward", distributed_forward, (x_m, y_m))
        print(f"  ⏱️  Forward Time:  {t_fwd:.3f} ms")

        # 2. Backward Pass
        t_bwd = benchmark_run("Backward (Grad)", distributed_grad, (x_m, y_m))
        print(f"  ⏱️  Backward Time: {t_bwd:.3f} ms")

        # Stats
        ratio = t_bwd / t_fwd
        print(f"\n  📊 Ratio (Bwd / Fwd): {ratio:.2f}x")
        print("     (Expected: ~2.0x - 3.0x for kernel ops)")

if __name__ == "__main__":
    run_grad_benchmark()
