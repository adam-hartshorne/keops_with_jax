#!/usr/bin/env python3
"""
Verify data consistency between PyTorch and JAX test runs.
This script checks if the saved PyTorch results match what JAX expects.
"""
import numpy as np
import argparse


def check_saved_results(pytorch_file):
    """Check what's in the PyTorch results file."""
    try:
        data = np.load(pytorch_file)

        print("=" * 70)
        print(f"Contents of {pytorch_file}")
        print("=" * 70)

        for key in sorted(data.files):
            arr = data[key]
            print(f"\n{key}:")
            print(f"  Shape: {arr.shape}")
            print(f"  Dtype: {arr.dtype}")
            print(f"  Mean:  {arr.mean():.6f}")
            print(f"  Std:   {arr.std():.6f}")

            if arr.ndim == 2:
                print(f"  Sample values:")
                print(f"    [0, 0]:    {arr[0, 0]:.6f}")
                print(f"    [0, -1]:   {arr[0, -1]:.6f}")
                print(f"    [-1, 0]:   {arr[-1, 0]:.6f}")
                print(f"    [-1, -1]:  {arr[-1, -1]:.6f}")
            elif arr.ndim == 1:
                print(f"  Sample values:")
                print(f"    [0]:   {arr[0]:.6f}")
                print(f"    [-1]:  {arr[-1]:.6f}")

        return data

    except FileNotFoundError:
        print(f"ERROR: File not found: {pytorch_file}")
        print("Run debug_varifold_torch.py first to generate this file.")
        return None


def simulate_jax_data_generation(N, M, D, batch_size, seed):
    """Simulate what JAX would generate with the same seed."""
    rng = np.random.RandomState(seed)

    if batch_size is not None:
        x_np = rng.randn(batch_size, N, D).astype(np.float32)
        y_np = rng.randn(batch_size, M, D).astype(np.float32)
        u_np = rng.randn(batch_size, N, D).astype(np.float32)
        v_np = rng.randn(batch_size, M, D).astype(np.float32)
        u_np = u_np / np.linalg.norm(u_np, axis=-1, keepdims=True)
        v_np = v_np / np.linalg.norm(v_np, axis=-1, keepdims=True)
    else:
        x_np = rng.randn(N, D).astype(np.float32)
        y_np = rng.randn(M, D).astype(np.float32)
        u_np = rng.randn(N, D).astype(np.float32)
        v_np = rng.randn(M, D).astype(np.float32)
        u_np = u_np / np.linalg.norm(u_np, axis=-1, keepdims=True)
        v_np = v_np / np.linalg.norm(v_np, axis=-1, keepdims=True)

    return x_np, y_np, u_np, v_np


def main():
    parser = argparse.ArgumentParser(description='Verify data consistency')
    parser.add_argument('--pytorch-results', default='debug_pytorch_results.npz',
                        help='PyTorch results file')
    parser.add_argument('--N', type=int, default=1000, help='Expected N')
    parser.add_argument('--M', type=int, default=1000, help='Expected M')
    parser.add_argument('--D', type=int, default=3, help='Expected D')
    parser.add_argument('--batch-size', type=int, default=4, help='Expected batch size')
    args = parser.parse_args()

    # Check saved results
    saved_data = check_saved_results(args.pytorch_results)

    if saved_data is None:
        return

    print("\n" + "=" * 70)
    print("Checking data generation consistency")
    print("=" * 70)

    # Check if batched results have expected shape
    if 'lazy_batched' in saved_data:
        expected_shape = (args.batch_size, args.N)
        actual_shape = saved_data['lazy_batched'].shape

        print(f"\nBatched results:")
        print(f"  Expected shape: {expected_shape}")
        print(f"  Actual shape:   {actual_shape}")

        if expected_shape == actual_shape:
            print(f"  ✓ Shape matches!")
        else:
            print(f"  ✗ Shape MISMATCH!")
            print(f"\n  This means PyTorch was run with different parameters:")
            if actual_shape[0] != expected_shape[0]:
                print(f"    - PyTorch batch_size={actual_shape[0]} vs expected={expected_shape[0]}")
            if actual_shape[1] != expected_shape[1]:
                print(f"    - PyTorch N={actual_shape[1]} vs expected={expected_shape[1]}")
            print(f"\n  Solution: Delete {args.pytorch_results} and re-run both scripts with same params")

    # Simulate what JAX would generate
    print(f"\n" + "=" * 70)
    print("Simulating JAX data generation (batched, seed=43)")
    print("=" * 70)

    x_jax, y_jax, u_jax, v_jax = simulate_jax_data_generation(
        args.N, args.M, args.D, args.batch_size, seed=43
    )

    print(f"\nJAX would generate:")
    print(f"  x shape: {x_jax.shape}")
    print(f"  x[0, 0, :] = {x_jax[0, 0, :]}")
    print(f"  x[0, -1, :] = {x_jax[0, -1, :]}")
    print(f"  x[-1, -1, :] = {x_jax[-1, -1, :]}")

    print(f"\n" + "=" * 70)
    print("Key Questions to Answer:")
    print("=" * 70)
    print("1. Did you run debug_varifold_torch.py with the SAME parameters as debug_varifold_jax.py?")
    print("2. Did you delete debug_pytorch_results.npz before re-running tests?")
    print("3. Are both scripts using the SAME seed for batched test (currently seed=43)?")
    print("\nIf any answer is 'no', that explains the mismatch!")


if __name__ == '__main__':
    main()