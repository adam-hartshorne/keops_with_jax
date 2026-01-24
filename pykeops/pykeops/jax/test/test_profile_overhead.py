#!/usr/bin/env python3
"""
Profile JAX KeOps overhead to identify bottlenecks.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import numpy as np

# Test 1: Pure CUDA kernel launch overhead (no KeOps)
print("=" * 60)
print("Test 1: Baseline JAX operation overhead")
print("=" * 60)

import jax
import jax.numpy as jnp

n, m, d = 50000, 20000, 3
x_jax = jnp.array(np.random.randn(n, d).astype(np.float32))
y_jax = jnp.array(np.random.randn(m, d).astype(np.float32))


# Simple JAX matmul for comparison
@jax.jit
def simple_matmul(x, y):
    return x @ y.T


# Warmup
for _ in range(5):
    result = simple_matmul(x_jax, y_jax[:, :d])
    result.block_until_ready()

# Benchmark
times = []
for _ in range(20):
    start = time.perf_counter()
    result = simple_matmul(x_jax, y_jax[:, :d])
    result.block_until_ready()
    times.append((time.perf_counter() - start) * 1000)

print(f"JAX matmul ({n}x{d}) @ ({d}x{m}): {np.mean(times):.3f} ± {np.std(times):.3f} ms")

# Test 2: KeOps with detailed timing
print("\n" + "=" * 60)
print("Test 2: KeOps operation breakdown")
print("=" * 60)

os.environ["PYKEOPS_JAX_MODE"] = "1"
from pykeops.jax import Genred

formula = "SqDist(x,y)"
aliases = ["x=Vi(3)", "y=Vj(3)"]

op = Genred(formula, aliases, reduction_op='Sum', axis=1)

# Warmup
for _ in range(5):
    result = op(x_jax, y_jax)
    result.block_until_ready()

# Benchmark with detailed timing
times_total = []
times_call = []
times_wait = []

for _ in range(50):
    start = time.perf_counter()
    result = op(x_jax, y_jax)
    after_call = time.perf_counter()
    result.block_until_ready()
    after_wait = time.perf_counter()

    times_total.append((after_wait - start) * 1000)
    times_call.append((after_call - start) * 1000)
    times_wait.append((after_wait - after_call) * 1000)

print(f"Total time:     {np.mean(times_total):.3f} ± {np.std(times_total):.3f} ms")
print(f"  - Call time:  {np.mean(times_call):.3f} ± {np.std(times_call):.3f} ms (Python → FFI → launch)")
print(f"  - Wait time:  {np.mean(times_wait):.3f} ± {np.std(times_wait):.3f} ms (kernel execution)")

# Test 3: Compare with PyTorch
print("\n" + "=" * 60)
print("Test 3: PyTorch KeOps breakdown")
print("=" * 60)

import torch
from pykeops.torch import Genred as TorchGenred

x_torch = torch.from_numpy(np.random.randn(n, d).astype(np.float32)).cuda()
y_torch = torch.from_numpy(np.random.randn(m, d).astype(np.float32)).cuda()

op_torch = TorchGenred(formula, aliases, reduction_op='Sum', axis=1)

# Warmup
for _ in range(5):
    result = op_torch(x_torch, y_torch)
    torch.cuda.synchronize()

times_total_torch = []
times_call_torch = []
times_wait_torch = []

for _ in range(50):
    start = time.perf_counter()
    result = op_torch(x_torch, y_torch)
    after_call = time.perf_counter()
    torch.cuda.synchronize()
    after_wait = time.perf_counter()

    times_total_torch.append((after_wait - start) * 1000)
    times_call_torch.append((after_call - start) * 1000)
    times_wait_torch.append((after_wait - after_call) * 1000)

print(f"Total time:     {np.mean(times_total_torch):.3f} ± {np.std(times_total_torch):.3f} ms")
print(f"  - Call time:  {np.mean(times_call_torch):.3f} ± {np.std(times_call_torch):.3f} ms (Python → launch)")
print(f"  - Wait time:  {np.mean(times_wait_torch):.3f} ± {np.std(times_wait_torch):.3f} ms (kernel execution)")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"JAX total:     {np.mean(times_total):.3f} ms")
print(f"PyTorch total: {np.mean(times_total_torch):.3f} ms")
print(f"Difference:    {np.mean(times_total) - np.mean(times_total_torch):.3f} ms")
print(f"\nJAX call overhead:     {np.mean(times_call):.3f} ms")
print(f"PyTorch call overhead: {np.mean(times_call_torch):.3f} ms")
print(f"Call overhead diff:    {np.mean(times_call) - np.mean(times_call_torch):.3f} ms")
print(f"\nJAX kernel time:     {np.mean(times_wait):.3f} ms")
print(f"PyTorch kernel time: {np.mean(times_wait_torch):.3f} ms")
print(f"Kernel time diff:    {np.mean(times_wait) - np.mean(times_wait_torch):.3f} ms")