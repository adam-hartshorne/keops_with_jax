#!/usr/bin/env python3
import os

os.environ['PYKEOPS_JAX_MODE'] = '1'

import argparse
import numpy as np
import jax
import jax.numpy as jnp
from pykeops.jax import LazyTensor, Genred


class Colors:
    CYAN, GREEN, YELLOW, RED, END = '\033[96m', '\033[92m', '\033[93m', '\033[91m', '\033[0m'


def get_devices():
    """Get available JAX devices."""
    try:
        return jax.devices('gpu')
    except RuntimeError:
        return jax.devices('cpu')


def create_test_data(N: int, M: int, D: int, batch_size: int = None, dtype=jnp.float32, seed=42):
    """Create test data for varifold kernels."""
    rng = np.random.RandomState(seed)

    if batch_size is not None:
        # Batched: [B, N, D]
        x_np = rng.randn(batch_size, N, D).astype(np.float32)
        y_np = rng.randn(batch_size, M, D).astype(np.float32)
        u_np = rng.randn(batch_size, N, D).astype(np.float32)
        v_np = rng.randn(batch_size, M, D).astype(np.float32)
        u_np = u_np / np.linalg.norm(u_np, axis=-1, keepdims=True)
        v_np = v_np / np.linalg.norm(v_np, axis=-1, keepdims=True)
    else:
        # Unbatched: [N, D]
        x_np = rng.randn(N, D).astype(np.float32)
        y_np = rng.randn(M, D).astype(np.float32)
        u_np = rng.randn(N, D).astype(np.float32)
        v_np = rng.randn(M, D).astype(np.float32)
        u_np = u_np / np.linalg.norm(u_np, axis=-1, keepdims=True)
        v_np = v_np / np.linalg.norm(v_np, axis=-1, keepdims=True)

    return {
        'x': jnp.array(x_np, dtype=dtype),
        'y': jnp.array(y_np, dtype=dtype),
        'u': jnp.array(u_np, dtype=dtype),
        'v': jnp.array(v_np, dtype=dtype)
    }


def varifold_kernel_lazytensor(x, y, u, v, gamma_s, gamma_n):
    """Varifold kernel using LazyTensor."""
    if x.ndim == 3:
        # Batched (B, N, D)
        x_L = LazyTensor(x[:, :, None, :])
        y_L = LazyTensor(y[:, None, :, :])
        u_L = LazyTensor(u[:, :, None, :])
        v_L = LazyTensor(v[:, None, :, :])
        reduction_axis = 2
    else:
        # Unbatched (N, D)
        x_L = LazyTensor(x[:, None, :])
        y_L = LazyTensor(y[None, :, :])
        u_L = LazyTensor(u[:, None, :])
        v_L = LazyTensor(v[None, :, :])
        reduction_axis = 1

    sq_dist = x_L.sqdist(y_L)
    dot_prod = (u_L * v_L).sum(-1)
    K = (-sq_dist * gamma_s).exp() * (dot_prod * gamma_n).exp()
    result = K.sum(axis=reduction_axis)

    # Remove trailing dimension if present
    if result.ndim == 3 and result.shape[-1] == 1:
        result = result.squeeze(-1)
    elif result.ndim == 2 and result.shape[-1] == 1 and x.ndim == 2:
        result = result.squeeze(-1)

    return result


def varifold_kernel_genred(x, y, u, v, gamma_s, gamma_n):
    """Varifold kernel using Genred."""
    D = x.shape[-1]
    formula = "Exp(-g * SqDist(x, y)) * Exp(g1 * (u | v))"
    aliases = [
        f"x = Vi({D})", f"y = Vj({D})",
        f"u = Vi({D})", f"v = Vj({D})",
        "g = Pm(1)", "g1 = Pm(1)"
    ]
    genred_fn = Genred(formula, aliases, reduction_op='Sum', axis=1)
    g = jnp.atleast_1d(gamma_s).reshape(-1)
    g1 = jnp.atleast_1d(gamma_n).reshape(-1)
    result = genred_fn(x, y, u, v, g, g1)

    # Genred always returns [..., 1] trailing dimension, squeeze it
    if result.shape[-1] == 1:
        result = result.squeeze(-1)

    return result


def print_fingerprint(name, result, indices=None):
    """Print fingerprint values at specific indices."""
    result_np = np.array(result)

    print(f"\n  {name}:")
    print(f"    Shape: {result.shape}")
    print(f"    Mean:  {float(result_np.mean()):.6f}")
    print(f"    Std:   {float(result_np.std()):.6f}")

    if indices is None:
        # Default indices based on shape
        if result.ndim == 1:
            indices = [(0,), (-1,)]
        elif result.ndim == 2:
            indices = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
        else:
            # For higher dimensions, just show first and last
            indices = []

    if len(indices) > 0:
        print(f"    Values at indices:")
        for idx in indices:
            val = float(result_np[idx])
            print(f"      {str(idx):<12} {val:.6f}")


def compare_results(lazy_result, genred_result, name):
    """Compare LazyTensor and Genred results."""
    lazy_np = np.array(lazy_result)
    genred_np = np.array(genred_result)

    max_diff = float(np.abs(lazy_np - genred_np).max())
    mean_diff = float(np.abs(lazy_np - genred_np).mean())

    print(f"\n  {name} - LazyTensor vs Genred:")
    print(f"    Max diff:  {max_diff:.2e}")
    print(f"    Mean diff: {mean_diff:.2e}")

    if max_diff < 1e-4:
        print(f"    {Colors.GREEN}✓ PASSED{Colors.END}")
    else:
        print(f"    {Colors.RED}✗ FAILED{Colors.END}")

    return max_diff < 1e-4


def compare_with_pytorch(jax_result, pytorch_file, result_key):
    """Compare JAX results with PyTorch reference."""
    try:
        pt_data = np.load(pytorch_file)
        if result_key not in pt_data:
            print(f"    {Colors.YELLOW}Warning: '{result_key}' not found in PyTorch results{Colors.END}")
            return True

        pt_result = pt_data[result_key]
        jax_np = np.array(jax_result)

        # Try to match shapes
        if jax_np.shape != pt_result.shape:
            if jax_np.size == pt_result.size:
                jax_np = jax_np.reshape(pt_result.shape)
            else:
                print(
                    f"    {Colors.YELLOW}Warning: Shape mismatch - JAX: {jax_np.shape}, PyTorch: {pt_result.shape}{Colors.END}")
                return True

        max_diff = float(np.abs(jax_np - pt_result).max())
        mean_diff = float(np.abs(jax_np - pt_result).mean())

        print(f"\n  JAX vs PyTorch:")
        print(f"    Max diff:  {max_diff:.2e}")
        print(f"    Mean diff: {mean_diff:.2e}")

        if max_diff < 1e-4:
            print(f"    {Colors.GREEN}✓ PASSED{Colors.END}")
        else:
            print(f"    {Colors.RED}✗ FAILED{Colors.END}")

        return max_diff < 1e-4

    except FileNotFoundError:
        print(f"    {Colors.YELLOW}PyTorch results file not found: {pytorch_file}{Colors.END}")
        return True
    except Exception as e:
        print(f"    {Colors.YELLOW}Error comparing with PyTorch: {e}{Colors.END}")
        return True


def run_test(name, N, M, D, batch_size, gamma_s, gamma_n, pytorch_file=None, seed=42):
    """Run a single test configuration."""
    batch_str = "unbatched" if batch_size is None else f"batched (B={batch_size})"
    print(f"\n{Colors.CYAN}{'=' * 70}")
    print(f"Test: {name} - {batch_str}")
    print(f"Size: N={N}, M={M}, D={D}")
    print(f"{'=' * 70}{Colors.END}")

    # Create data with specific seed for this test
    data = create_test_data(N, M, D, batch_size, seed=seed)

    # Run LazyTensor
    lazy_result = varifold_kernel_lazytensor(
        data['x'], data['y'], data['u'], data['v'], gamma_s, gamma_n
    )

    # Run Genred
    genred_result = varifold_kernel_genred(
        data['x'], data['y'], data['u'], data['v'], gamma_s, gamma_n
    )

    # Print fingerprints
    print_fingerprint("LazyTensor", lazy_result)
    print_fingerprint("Genred", genred_result)

    # Compare JAX implementations
    passed = compare_results(lazy_result, genred_result, name)

    # Compare with PyTorch if available
    if pytorch_file:
        result_key = 'lazy_unbatched' if batch_size is None else 'lazy_batched'
        pt_passed = compare_with_pytorch(lazy_result, pytorch_file, result_key)
        passed = passed and pt_passed

    return lazy_result, genred_result, passed


def main():
    parser = argparse.ArgumentParser(description='Debug JAX KeOps varifold kernels')
    parser.add_argument('--N', type=int, default=1000, help='Number of source points')
    parser.add_argument('--M', type=int, default=1000, help='Number of target points')
    parser.add_argument('--D', type=int, default=3, help='Dimension')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for batched test')
    parser.add_argument('--skip-unbatched', action='store_true', help='Skip unbatched tests')
    parser.add_argument('--pytorch-results', default='debug_pytorch_results.npz', help='PyTorch results for comparison')
    parser.add_argument('--output', default='debug_jax_results.npz', help='Output file path')
    args = parser.parse_args()

    devices = get_devices()

    print(f"{Colors.CYAN}{'=' * 70}")
    print(f"JAX KeOps Varifold Debug")
    print(f"{'=' * 70}{Colors.END}")
    print(f"Device: {devices[0].platform}:{devices[0].id}")
    print(f"Configuration: N={args.N}, M={args.M}, D={args.D}, Batch={args.batch_size}")

    # Kernel parameters
    sigma_spatial, sigma_normal = 0.5, 1.0
    gamma_s = jnp.array(1.0 / (2 * sigma_spatial ** 2))
    gamma_n = jnp.array(1.0 / (sigma_normal ** 2))

    results = {}
    all_passed = True

    # Unbatched test (seed=42)
    if not args.skip_unbatched:
        lazy_ub, genred_ub, passed_ub = run_test(
            "Unbatched", args.N, args.M, args.D, None, gamma_s, gamma_n, args.pytorch_results, seed=42
        )
        results['lazy_unbatched'] = np.array(lazy_ub)
        results['genred_unbatched'] = np.array(genred_ub)
        all_passed = all_passed and passed_ub
    else:
        print(f"\n{Colors.YELLOW}Skipping unbatched tests (--skip-unbatched){Colors.END}")

    # Batched test (seed=43 to ensure different data)
    lazy_b, genred_b, passed_b = run_test(
        "Batched", args.N, args.M, args.D, args.batch_size, gamma_s, gamma_n, args.pytorch_results, seed=43
    )
    results['lazy_batched'] = np.array(lazy_b)
    results['genred_batched'] = np.array(genred_b)
    all_passed = all_passed and passed_b

    # Save results
    np.savez(args.output, **results)

    # Final summary
    print(f"\n{Colors.CYAN}{'=' * 70}")
    print(f"Summary")
    print(f"{'=' * 70}{Colors.END}")
    if all_passed:
        print(f"{Colors.GREEN}✓ All tests PASSED{Colors.END}")
    else:
        print(f"{Colors.RED}✗ Some tests FAILED{Colors.END}")
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()