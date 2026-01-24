#!/usr/bin/env python3
"""
Test to isolate whether the overhead is in synchronization or kernel execution.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYKEOPS_JAX_MODE"] = "1"

import time
import numpy as np
import jax
import jax.numpy as jnp
from pykeops.jax import Genred

import torch
from pykeops.torch import Genred as TorchGenred

n, m, d = 50000, 20000, 3

formula = "SqDist(x,y)"
aliases = ["x=Vi(3)", "y=Vj(3)"]

print("=" * 60)
print("Test: Batch multiple calls to amortize sync overhead")
print("=" * 60)

# JAX setup
x_jax = jnp.array(np.random.randn(n, d).astype(np.float32))
y_jax = jnp.array(np.random.randn(m, d).astype(np.float32))
op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1)

# PyTorch setup
x_torch = torch.randn(n, d, device='cuda', dtype=torch.float32)
y_torch = torch.randn(m, d, device='cuda', dtype=torch.float32)
op_torch = TorchGenred(formula, aliases, reduction_op='Sum', axis=1)

# Warmup
for _ in range(5):
    r = op_jax(x_jax, y_jax)
    r.block_until_ready()
    r = op_torch(x_torch, y_torch)
    torch.cuda.synchronize()

print("\nSingle call timing:")
print("-" * 40)

# Single JAX call
times = []
for _ in range(50):
    jax.block_until_ready(x_jax)
    start = time.perf_counter()
    r = op_jax(x_jax, y_jax)
    r.block_until_ready()
    times.append((time.perf_counter() - start) * 1000)
jax_single = np.mean(times)
print(f"JAX single call:     {jax_single:.3f} ms")

# Single PyTorch call
times = []
for _ in range(50):
    torch.cuda.synchronize()
    start = time.perf_counter()
    r = op_torch(x_torch, y_torch)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1000)
torch_single = np.mean(times)
print(f"PyTorch single call: {torch_single:.3f} ms")

print(f"\nSingle call overhead: {jax_single - torch_single:.3f} ms ({jax_single/torch_single:.2f}x)")

print("\n\nBatch of 10 calls (amortize sync):")
print("-" * 40)

# Batch of 10 JAX calls, single sync at end
times = []
for _ in range(20):
    jax.block_until_ready(x_jax)
    start = time.perf_counter()
    results = []
    for _ in range(10):
        results.append(op_jax(x_jax, y_jax))
    # Single sync at end
    for r in results:
        r.block_until_ready()
    times.append((time.perf_counter() - start) * 1000 / 10)
jax_batch = np.mean(times)
print(f"JAX per-call (batched):     {jax_batch:.3f} ms")

# Batch of 10 PyTorch calls, single sync at end
times = []
for _ in range(20):
    torch.cuda.synchronize()
    start = time.perf_counter()
    results = []
    for _ in range(10):
        results.append(op_torch(x_torch, y_torch))
    torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1000 / 10)
torch_batch = np.mean(times)
print(f"PyTorch per-call (batched): {torch_batch:.3f} ms")

print(f"\nBatched call overhead: {jax_batch - torch_batch:.3f} ms ({jax_batch/torch_batch:.2f}x)")

print("\n" + "=" * 60)
print("Analysis")
print("=" * 60)
print(f"""
If batched overhead is similar to single overhead:
  -> Overhead is per-kernel (kernel execution slower)
  
If batched overhead is much smaller than single overhead:
  -> Overhead is per-sync (synchronization/FFI overhead)

Single call gap:  {jax_single - torch_single:.3f} ms
Batched call gap: {jax_batch - torch_batch:.3f} ms
""")

# Additional test: Check JAX lowering overhead
print("\n" + "=" * 60)
print("Test: JAX compilation/lowering overhead")
print("=" * 60)

# Time just the FFI call without JIT tracing
@jax.jit
def jitted_op(x, y):
    return op_jax(x, y)

# First call - includes JIT compilation
start = time.perf_counter()
r = jitted_op(x_jax, y_jax)
r.block_until_ready()
first_call = (time.perf_counter() - start) * 1000
print(f"First JIT call: {first_call:.3f} ms")

# Subsequent calls
times = []
for _ in range(50):
    start = time.perf_counter()
    r = jitted_op(x_jax, y_jax)
    r.block_until_ready()
    times.append((time.perf_counter() - start) * 1000)
print(f"Cached JIT call: {np.mean(times):.3f} ± {np.std(times):.3f} ms")
