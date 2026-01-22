"""
KeOps PyTorch Backend Performance Benchmark
Runs various problem sizes and saves results to file
"""
import os
os.environ['PYKEOPS_JAX_MODE'] = '0'  # Ensure PyTorch mode

import time
import torch
import numpy as np
from pykeops.torch import Genred
import json

print("="*70)
print("KeOps PyTorch Backend Performance Benchmark")
print("="*70)

# Test configurations
test_configs = [
    # (name, nx, ny, dim, formula, reduction)
    ("Small: Euclidean distance", 1000, 1000, 3, "SqNorm2(x-y)", "Sum"),
    ("Medium: Euclidean distance", 10000, 10000, 3, "SqNorm2(x-y)", "Sum"),
    ("Large: Euclidean distance", 50000, 20000, 3, "SqNorm2(x-y)", "Sum"),
    ("Very Large: Euclidean distance", 100000, 50000, 3, "SqNorm2(x-y)", "Sum"),

    ("Small: Gaussian kernel", 1000, 1000, 3, "Exp(-SqNorm2(x-y)*s)", "Sum"),
    ("Medium: Gaussian kernel", 10000, 10000, 3, "Exp(-SqNorm2(x-y)*s)", "Sum"),
    ("Large: Gaussian kernel", 50000, 20000, 3, "Exp(-SqNorm2(x-y)*s)", "Sum"),

    ("High dim: Euclidean", 10000, 10000, 10, "SqNorm2(x-y)", "Sum"),
    ("High dim: Gaussian", 10000, 10000, 10, "Exp(-SqNorm2(x-y)*s)", "Sum"),
]

n_warmup = 5
n_iter = 20

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print(f"Warmup iterations: {n_warmup}")
print(f"Benchmark iterations: {n_iter}")
print()

results = []

for test_name, nx, ny, dim, formula, reduction in test_configs:
    print(f"\n{'='*70}")
    print(f"{test_name}")
    print(f"  Problem: nx={nx}, ny={ny}, dim={dim}")
    print(f"  Formula: {formula}")
    print(f"  Reduction: {reduction}")
    print(f"{'='*70}")

    # Create data
    x = torch.randn(nx, dim, dtype=torch.float32, device=device)
    y = torch.randn(ny, dim, dtype=torch.float32, device=device)

    # Create operator
    if 's)' in formula:  # Has parameter
        aliases = [f"x=Vi({dim})", f"y=Vj({dim})", "s=Pm(1)"]
        sigma = 1.0
        inv_two_sigma_sq = torch.tensor([1.0 / (2 * sigma**2)], dtype=torch.float32, device=device)
        op = Genred(formula, aliases, reduction_op=reduction, axis=1)

        # Warmup
        print("  Warming up...", end=" ", flush=True)
        for _ in range(n_warmup):
            _ = op(x, y, inv_two_sigma_sq)
        if device == 'cuda':
            torch.cuda.synchronize()
        print("done")

        # Benchmark
        print("  Benchmarking...", end=" ", flush=True)
        times = []
        for _ in range(n_iter):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = op(x, y, inv_two_sigma_sq)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        print("done")
    else:  # No parameter
        aliases = [f"x=Vi({dim})", f"y=Vj({dim})"]
        op = Genred(formula, aliases, reduction_op=reduction, axis=1)

        # Warmup
        print("  Warming up...", end=" ", flush=True)
        for _ in range(n_warmup):
            _ = op(x, y)
        if device == 'cuda':
            torch.cuda.synchronize()
        print("done")

        # Benchmark
        print("  Benchmarking...", end=" ", flush=True)
        times = []
        for _ in range(n_iter):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = op(x, y)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        print("done")

    mean_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000

    print(f"\n  Results:")
    print(f"    Mean: {mean_time:.3f} ± {std_time:.3f} ms")
    print(f"    Min:  {min_time:.3f} ms")
    print(f"    Max:  {max_time:.3f} ms")

    results.append({
        'name': test_name,
        'nx': nx,
        'ny': ny,
        'dim': dim,
        'formula': formula,
        'reduction': reduction,
        'mean_ms': mean_time,
        'std_ms': std_time,
        'min_ms': min_time,
        'max_ms': max_time,
        'backend': 'pytorch'
    })

    # Clean up
    del x, y, op
    if device == 'cuda':
        torch.cuda.empty_cache()

# Save results
output_file = 'benchmark_pytorch.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"{'Test':<35} {'Mean (ms)':<15} {'Std (ms)':<15}")
print("-"*70)

for r in results:
    print(f"{r['name']:<35} {r['mean_ms']:<15.3f} {r['std_ms']:<15.3f}")

print(f"\n✅ Results saved to: {output_file}")
print(f"{'='*70}")