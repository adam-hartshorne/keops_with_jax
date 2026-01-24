#!/usr/bin/env python3
"""
Test script to verify chunking support works in JAX backend.

Chunking is used for high-dimensional variables and should be transparent
to the user - the same code works with or without chunking.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import time

# Import KeOps
from pykeops.jax import Genred

def test_high_dim_gaussian():
    """Test high-dimensional Gaussian kernel where chunking helps."""
    print("\n" + "="*60)
    print("Testing high-dimensional Gaussian kernel (D=128)")
    print("="*60)
    
    n, m, d = 5000, 5000, 128  # High dimension
    
    np.random.seed(42)
    x_np = np.random.randn(n, d).astype(np.float32)
    y_np = np.random.randn(m, d).astype(np.float32)
    
    x = jnp.array(x_np)
    y = jnp.array(y_np)
    
    # Gaussian kernel
    formula = "Exp(-SqNorm2(x-y))"
    aliases = [f"x=Vi({d})", f"y=Vj({d})"]
    
    print(f"Formula: {formula}")
    print(f"Input shapes: x={x.shape}, y={y.shape}")
    
    op = Genred(formula, aliases, reduction_op='Sum', axis=1)
    
    # Warmup
    print("\nWarming up...")
    result = op(x, y)
    result.block_until_ready()
    
    # Benchmark
    print("Benchmarking...")
    n_runs = 10
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = op(x, y)
        result.block_until_ready()
        times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    
    print(f"Result shape: {result.shape}")
    print(f"Average time: {avg_time:.2f} ± {std_time:.2f} ms")
    
    return True

def test_high_dim_with_params():
    """Test high-dimensional kernel with parameters."""
    print("\n" + "="*60)
    print("Testing high-dimensional WeightedSum (D=64)")
    print("="*60)
    
    n, m, d = 3000, 3000, 64
    
    np.random.seed(42)
    x_np = np.random.randn(n, d).astype(np.float32)
    y_np = np.random.randn(m, d).astype(np.float32)
    b_np = np.random.randn(m, d).astype(np.float32)
    s_np = np.array([0.5], dtype=np.float32)
    
    x = jnp.array(x_np)
    y = jnp.array(y_np)
    b = jnp.array(b_np)
    s = jnp.array(s_np)
    
    formula = "Exp(-SqNorm2(x-y)*s)*b"
    aliases = [f"x=Vi({d})", f"y=Vj({d})", f"b=Vj({d})", "s=Pm(1)"]
    
    print(f"Formula: {formula}")
    print(f"Input shapes: x={x.shape}, y={y.shape}, b={b.shape}, s={s.shape}")
    
    op = Genred(formula, aliases, reduction_op='Sum', axis=1)
    
    # Forward pass
    print("\nTesting forward pass...")
    result = op(x, y, b, s)
    print(f"Forward pass OK! Result shape: {result.shape}")
    
    # Gradient
    print("\nTesting gradient...")
    def loss(x_in):
        return jnp.sum(op(x_in, y, b, s))
    
    grad = jax.grad(loss)(x)
    print(f"Gradient OK! Shape: {grad.shape}")
    
    return True

def test_batched_high_dim():
    """Test batched high-dimensional kernel."""
    print("\n" + "="*60)
    print("Testing batched high-dimensional kernel (B=4, D=96)")
    print("="*60)
    
    b, n, m, d = 4, 2000, 2000, 96
    
    np.random.seed(42)
    x_np = np.random.randn(b, n, d).astype(np.float32)
    y_np = np.random.randn(b, m, d).astype(np.float32)
    
    x = jnp.array(x_np)
    y = jnp.array(y_np)
    
    formula = "Exp(-SqNorm2(x-y))"
    aliases = [f"x=Vi({d})", f"y=Vj({d})"]
    
    print(f"Formula: {formula}")
    print(f"Input shapes: x={x.shape}, y={y.shape}")
    
    op = Genred(formula, aliases, reduction_op='Sum', axis=1)
    
    result = op(x, y)
    print(f"Result shape: {result.shape}")
    
    # Verify batch dimension preserved
    assert result.shape == (b, n, 1), f"Expected {(b, n, 1)}, got {result.shape}"
    print("Batch dimension preserved correctly!")
    
    return True

def test_gradient_high_dim():
    """Test gradient computation with high-dimensional variables."""
    print("\n" + "="*60)
    print("Testing gradient for high-dimensional Cauchy kernel (D=80)")
    print("="*60)
    
    n, m, d = 1000, 1000, 80
    
    np.random.seed(42)
    x_np = np.random.randn(n, d).astype(np.float32)
    y_np = np.random.randn(m, d).astype(np.float32)
    
    x = jnp.array(x_np)
    y = jnp.array(y_np)
    
    formula = "Inv(IntCst(1)+SqNorm2(x-y))"
    aliases = [f"x=Vi({d})", f"y=Vj({d})"]
    
    print(f"Formula: {formula}")
    print(f"Input shapes: x={x.shape}, y={y.shape}")
    
    op = Genred(formula, aliases, reduction_op='Sum', axis=1)
    
    # Forward pass
    result = op(x, y)
    print(f"Forward pass OK! Result shape: {result.shape}")
    
    # Gradient w.r.t. x
    def loss_x(x_in):
        return jnp.sum(op(x_in, y))
    
    grad_x = jax.grad(loss_x)(x)
    print(f"Gradient w.r.t. x OK! Shape: {grad_x.shape}")
    
    # Gradient w.r.t. y
    def loss_y(y_in):
        return jnp.sum(op(x, y_in))
    
    grad_y = jax.grad(loss_y)(y)
    print(f"Gradient w.r.t. y OK! Shape: {grad_y.shape}")
    
    return True

if __name__ == '__main__':
    print("="*60)
    print("Chunking Support Test Suite")
    print("="*60)
    
    all_passed = True
    
    try:
        test_high_dim_gaussian()
    except Exception as e:
        print(f"\n✗ test_high_dim_gaussian failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        test_high_dim_with_params()
    except Exception as e:
        print(f"\n✗ test_high_dim_with_params failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        test_batched_high_dim()
    except Exception as e:
        print(f"\n✗ test_batched_high_dim failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        test_gradient_high_dim()
    except Exception as e:
        print(f"\n✗ test_gradient_high_dim failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("="*60)
