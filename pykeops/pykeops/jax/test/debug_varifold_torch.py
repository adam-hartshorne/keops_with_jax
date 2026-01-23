#!/usr/bin/env python3
import argparse
import numpy as np
import torch
from pykeops.torch import LazyTensor, Genred


class Colors:
    CYAN, GREEN, YELLOW, RED, END = '\033[96m', '\033[92m', '\033[93m', '\033[91m', '\033[0m'


def create_test_data(N: int, M: int, D: int, batch_size: int = None, dtype=torch.float32, device='cuda:0', seed=42):
    """Create test data for varifold kernels."""
    rng = np.random.RandomState(seed)

    if batch_size is not None:
        # Batched: [B, N, D]
        x_np = rng.randn(batch_size, N, D).astype(np.float32)
        y_np = rng.randn(batch_size, M, D).astype(np.float32)
        u_np = rng.randn(batch_size, N, D).astype(np.float32)
        v_np = rng.randn(batch_size, M, D).astype(np.float32)
        u_np /= np.linalg.norm(u_np, axis=-1, keepdims=True)
        v_np /= np.linalg.norm(v_np, axis=-1, keepdims=True)
    else:
        # Unbatched: [N, D]
        x_np = rng.randn(N, D).astype(np.float32)
        y_np = rng.randn(M, D).astype(np.float32)
        u_np = rng.randn(N, D).astype(np.float32)
        v_np = rng.randn(M, D).astype(np.float32)
        u_np /= np.linalg.norm(u_np, axis=-1, keepdims=True)
        v_np /= np.linalg.norm(v_np, axis=-1, keepdims=True)

    to_dev = lambda arr: torch.from_numpy(arr).to(device=device, dtype=dtype)

    return {
        'x': to_dev(x_np), 'y': to_dev(y_np),
        'u': to_dev(u_np), 'v': to_dev(v_np)
    }


def varifold_kernel_lazytensor(x, y, u, v, gamma_s, gamma_n):
    """Varifold kernel using LazyTensor."""
    if x.dim() == 3:
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
    if result.dim() == 3 and result.shape[-1] == 1:
        result = result.squeeze(-1)
    elif result.dim() == 2 and result.shape[-1] == 1 and x.dim() == 2:
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
    g = gamma_s.view(1)
    g1 = gamma_n.view(1)
    result = genred_fn(x, y, u, v, g, g1)

    # Genred always returns [..., 1] trailing dimension, squeeze it
    if result.shape[-1] == 1:
        result = result.squeeze(-1)

    return result


def print_fingerprint(name, result, indices=None):
    """Print fingerprint values at specific indices."""
    result_np = result.detach().cpu().numpy()

    print(f"\n  {name}:")
    print(f"    Shape: {result.shape}")
    print(f"    Mean:  {result_np.mean():.6f}")
    print(f"    Std:   {result_np.std():.6f}")

    if indices is None:
        # Default indices based on shape
        if result.dim() == 1:
            indices = [(0,), (-1,)]
        elif result.dim() == 2:
            indices = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
        else:
            # For higher dimensions, just show first and last
            indices = []

    if len(indices) > 0:
        print(f"    Values at indices:")
        for idx in indices:
            val = result_np[idx]
            print(f"      {str(idx):<12} {val:.6f}")


def compare_results(lazy_result, genred_result, name):
    """Compare LazyTensor and Genred results."""
    lazy_np = lazy_result.detach().cpu().numpy()
    genred_np = genred_result.detach().cpu().numpy()

    max_diff = np.abs(lazy_np - genred_np).max()
    mean_diff = np.abs(lazy_np - genred_np).mean()

    print(f"\n  {name} - LazyTensor vs Genred:")
    print(f"    Max diff:  {max_diff:.2e}")
    print(f"    Mean diff: {mean_diff:.2e}")

    if max_diff < 1e-4:
        print(f"    {Colors.GREEN}✓ PASSED{Colors.END}")
    else:
        print(f"    {Colors.RED}✗ FAILED{Colors.END}")

    return max_diff < 1e-4


def run_test(name, N, M, D, batch_size, gamma_s, gamma_n, device, seed=42):
    """Run a single test configuration."""
    batch_str = "unbatched" if batch_size is None else f"batched (B={batch_size})"
    print(f"\n{Colors.CYAN}{'=' * 70}")
    print(f"Test: {name} - {batch_str}")
    print(f"Size: N={N}, M={M}, D={D}")
    print(f"{'=' * 70}{Colors.END}")

    # Create data with specific seed for this test
    data = create_test_data(N, M, D, batch_size, device=device, seed=seed)

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

    # Compare
    passed = compare_results(lazy_result, genred_result, name)

    return lazy_result, genred_result, passed


def main():
    parser = argparse.ArgumentParser(description='Debug PyTorch KeOps varifold kernels')
    parser.add_argument('--N', type=int, default=1000, help='Number of source points')
    parser.add_argument('--M', type=int, default=1000, help='Number of target points')
    parser.add_argument('--D', type=int, default=3, help='Dimension')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for batched test')
    parser.add_argument('--skip-unbatched', action='store_true', help='Skip unbatched tests')
    parser.add_argument('--output', default='debug_pytorch_results.npz', help='Output file path')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"{Colors.CYAN}{'=' * 70}")
    print(f"PyTorch KeOps Varifold Debug")
    print(f"{'=' * 70}{Colors.END}")
    print(f"Device: {device}")
    print(f"Configuration: N={args.N}, M={args.M}, D={args.D}, Batch={args.batch_size}")

    # Kernel parameters
    sigma_spatial, sigma_normal = 0.5, 1.0
    gamma_s = torch.tensor(1.0 / (2 * sigma_spatial ** 2), device=device)
    gamma_n = torch.tensor(1.0 / (sigma_normal ** 2), device=device)

    results = {}
    all_passed = True

    # Unbatched test (seed=42)
    if not args.skip_unbatched:
        lazy_ub, genred_ub, passed_ub = run_test(
            "Unbatched", args.N, args.M, args.D, None, gamma_s, gamma_n, device, seed=42
        )
        results['lazy_unbatched'] = lazy_ub.detach().cpu().numpy()
        results['genred_unbatched'] = genred_ub.detach().cpu().numpy()
        all_passed = all_passed and passed_ub
    else:
        print(f"\n{Colors.YELLOW}Skipping unbatched tests (--skip-unbatched){Colors.END}")

    # Batched test (seed=43 to ensure different data)
    lazy_b, genred_b, passed_b = run_test(
        "Batched", args.N, args.M, args.D, args.batch_size, gamma_s, gamma_n, device, seed=43
    )
    results['lazy_batched'] = lazy_b.detach().cpu().numpy()
    results['genred_batched'] = genred_b.detach().cpu().numpy()
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