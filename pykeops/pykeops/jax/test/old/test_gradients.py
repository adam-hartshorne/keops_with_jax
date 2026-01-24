"""
KeOps JAX Gradient Comparison Tests
====================================

Tests GIL-free gradient implementation by comparing:
- Pure JAX (ground truth)
- KeOps JAX (our GIL-free implementation)
- KeOps NumPy (reference KeOps implementation on CPU)
- KeOps PyTorch (reference KeOps implementation on GPU)
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

# Import KeOps backends
from pykeops.jax import Genred
from pykeops.numpy import Genred as Genred_numpy

# Try to import PyTorch
try:
    import torch
    from pykeops.torch import Genred as Genred_torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available, skipping PyTorch comparisons")

print("=" * 70)
print("  KeOps JAX vs Pure JAX vs KeOps NumPy vs KeOps PyTorch")
print("=" * 70)


def test_forward_pass_comparison():
    """Test that forward pass matches across all implementations."""
    print("\n" + "=" * 70)
    print("  Test: Forward Pass Comparison")
    print("=" * 70)

    # Test data
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    y = jnp.array([[0.0, 0.0]], dtype=jnp.float32)
    x_np = np.array(x)
    y_np = np.array(y)

    print(f"Input x: {x}")
    print(f"Input y: {y}")

    # Pure JAX (ground truth)
    def pure_jax_sqdist(x, y):
        # x is (N, D), y is (M, D)
        # Compute squared distance for each x[i] to y[j], sum over j
        diff = x[:, None, :] - y[None, :, :]  # (N, M, D)
        sqdist = jnp.sum(diff ** 2, axis=2)     # (N, M)
        return jnp.sum(sqdist, axis=1, keepdims=True)  # (N, 1)

    result_jax = pure_jax_sqdist(x, y)
    print(f"\nPure JAX result:\n{result_jax}")

    # KeOps JAX
    formula = "SqDist(x,y)"
    aliases = ["x=Vi(2)", "y=Vj(2)"]
    op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1)
    result_keops_jax = op_jax(x, y)
    print(f"\nKeOps JAX result:\n{result_keops_jax}")

    all_match = True

    # KeOps NumPy - use CPU backend
    print("\nKeOps NumPy result (CPU backend):")
    try:
        op_numpy = Genred_numpy(formula, aliases, reduction_op='Sum', axis=1)
        result_keops_numpy = op_numpy(x_np, y_np, backend='CPU')
        print(f"{result_keops_numpy}")

        jax_vs_keops_numpy = jnp.allclose(result_jax, result_keops_numpy, rtol=1e-5)
        print(f"✓ JAX vs KeOps NumPy match: {jax_vs_keops_numpy}")
        if not jax_vs_keops_numpy:
            all_match = False
    except Exception as e:
        print(f"NumPy backend failed: {e}")

    # KeOps PyTorch
    if HAS_TORCH:
        print("\nKeOps PyTorch result (GPU):")
        try:
            x_torch = torch.tensor(x_np, dtype=torch.float32).cuda()
            y_torch = torch.tensor(y_np, dtype=torch.float32).cuda()
            op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)
            result_keops_torch = op_torch(x_torch, y_torch).cpu().numpy()
            print(f"{result_keops_torch}")

            jax_vs_keops_torch = jnp.allclose(result_jax, result_keops_torch, rtol=1e-5)
            print(f"✓ JAX vs KeOps PyTorch match: {jax_vs_keops_torch}")
            if not jax_vs_keops_torch:
                all_match = False
        except Exception as e:
            print(f"PyTorch backend failed: {e}")

    # Compare JAX implementations
    jax_vs_keops_jax = jnp.allclose(result_jax, result_keops_jax, rtol=1e-5)
    print(f"\n✓ JAX vs KeOps JAX match: {jax_vs_keops_jax}")
    if not jax_vs_keops_jax:
        all_match = False

    if all_match:
        print("\n✓ PASS: All implementations match!")
    else:
        print("\n✗ FAIL: Some results don't match!")

    return all_match


def test_gradient_comparison():
    """Test that gradients match across all implementations."""
    print("\n" + "=" * 70)
    print("  Test: Gradient Comparison")
    print("=" * 70)

    # Test data
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    y = jnp.array([[0.0, 0.0]], dtype=jnp.float32)
    x_np = np.array(x)
    y_np = np.array(y)

    print(f"Input x: {x}")
    print(f"Input y: {y}")

    # Pure JAX gradient
    def pure_jax_sqdist(x, y):
        diff = x[:, None, :] - y[None, :, :]
        sqdist = jnp.sum(diff ** 2, axis=2)
        return jnp.sum(sqdist, axis=1, keepdims=True)

    def loss(x):
        return jnp.sum(pure_jax_sqdist(x, y))

    grad_jax = jax.grad(loss)(x)
    print(f"\nPure JAX gradient w.r.t. x:\n{grad_jax}")

    # KeOps JAX gradient
    formula = "SqDist(x,y)"
    aliases = ["x=Vi(2)", "y=Vj(2)"]
    op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1)

    def loss_keops_jax(x):
        return jnp.sum(op_jax(x, y))

    grad_keops_jax = jax.grad(loss_keops_jax)(x)
    print(f"\nKeOps JAX gradient w.r.t. x:\n{grad_keops_jax}")

    all_match = True

    # KeOps PyTorch gradient (most reliable reference for GPU)
    if HAS_TORCH:
        print("\nKeOps PyTorch gradient w.r.t. x (GPU):")
        try:
            # Create tensor on CPU with requires_grad, then move to GPU
            x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True, device='cuda')
            y_torch = torch.tensor(y_np, dtype=torch.float32, device='cuda')
            op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)

            result_torch = op_torch(x_torch, y_torch)
            loss_torch = result_torch.sum()
            loss_torch.backward()
            grad_keops_torch = x_torch.grad.cpu().numpy()
            print(f"{grad_keops_torch}")

            jax_vs_torch = jnp.allclose(grad_jax, grad_keops_torch, rtol=1e-5)
            keops_jax_vs_torch = jnp.allclose(grad_keops_jax, grad_keops_torch, rtol=1e-5)
            print(f"✓ JAX vs KeOps PyTorch match: {jax_vs_torch}")
            print(f"✓ KeOps JAX vs KeOps PyTorch match: {keops_jax_vs_torch}")
            if not jax_vs_torch or not keops_jax_vs_torch:
                all_match = False
        except Exception as e:
            print(f"PyTorch backend failed: {e}")
            import traceback
            traceback.print_exc()

    # KeOps NumPy gradient (manual using Grad operator, CPU backend)
    print("\nKeOps NumPy gradient w.r.t. x (CPU backend):")
    try:
        op_numpy = Genred_numpy(formula, aliases, reduction_op='Sum', axis=1)

        # Get output shape for eta
        output_shape = op_numpy(x_np, y_np, backend='CPU').shape
        eta_np = np.ones(output_shape, dtype=np.float32)

        # Compute gradient using KeOps Grad operator
        # For SqDist(x,y), the gradient is 2*(x-y)
        grad_formula = "Grad(SqDist(x,y), x, eta)"
        grad_aliases = ["x=Vi(2)", "y=Vj(2)", "eta=Vi(1)"]
        grad_op_numpy = Genred_numpy(grad_formula, grad_aliases, reduction_op='Sum', axis=1)

        # Use CPU backend
        grad_keops_numpy = grad_op_numpy(x_np, y_np, eta_np, backend='CPU')
        print(f"KeOps NumPy gradient: {grad_keops_numpy}")

        jax_vs_keops_numpy = jnp.allclose(grad_jax, grad_keops_numpy, rtol=1e-5)
        print(f"✓ JAX vs KeOps NumPy match: {jax_vs_keops_numpy}")
        if not jax_vs_keops_numpy:
            print(f"  Difference: {np.array(grad_jax) - grad_keops_numpy}")
            all_match = False
    except Exception as e:
        print(f"NumPy backend failed: {e}")
        import traceback
        traceback.print_exc()

    # Compare JAX implementations
    jax_vs_keops_jax = jnp.allclose(grad_jax, grad_keops_jax, rtol=1e-5)
    print(f"\n✓ JAX vs KeOps JAX match: {jax_vs_keops_jax}")
    if not jax_vs_keops_jax:
        all_match = False
        print(f"  Max diff: {jnp.max(jnp.abs(grad_jax - grad_keops_jax))}")

    if all_match:
        print("\n✓ PASS: All gradients match!")
    else:
        print("\n✗ FAIL: Some gradients don't match!")

    return all_match


def test_gradient_wrt_vj():
    """Test gradient w.r.t. y (Vj variable)."""
    print("\n" + "=" * 70)
    print("  Test: Gradient w.r.t. y (Vj variable)")
    print("=" * 70)

    # Test data
    x = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
    y = jnp.array([[0.0, 0.0]], dtype=jnp.float32)

    print(f"Input x: {x}")
    print(f"Input y: {y}")

    # Pure JAX gradient w.r.t. y
    def pure_jax_sqdist(x, y):
        diff = x[:, None, :] - y[None, :, :]
        sqdist = jnp.sum(diff ** 2, axis=2)
        return jnp.sum(sqdist, axis=1, keepdims=True)

    def loss(y):
        return jnp.sum(pure_jax_sqdist(x, y))

    grad_y_jax = jax.grad(loss)(y)
    print(f"\nPure JAX gradient w.r.t. y:\n{grad_y_jax}")

    # KeOps JAX gradient w.r.t. y
    formula = "SqDist(x,y)"
    aliases = ["x=Vi(2)", "y=Vj(2)"]
    op_jax = Genred(formula, aliases, reduction_op='Sum', axis=1)

    def loss_keops(y):
        return jnp.sum(op_jax(x, y))

    grad_y_keops_jax = jax.grad(loss_keops)(y)
    print(f"\nKeOps JAX gradient w.r.t. y:\n{grad_y_keops_jax}")

    all_match = True

    # KeOps PyTorch gradient w.r.t. y
    if HAS_TORCH:
        print("\nKeOps PyTorch gradient w.r.t. y (GPU):")
        try:
            x_torch = torch.tensor(np.array(x), dtype=torch.float32, device='cuda')
            y_torch = torch.tensor(np.array(y), dtype=torch.float32, requires_grad=True, device='cuda')
            op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)

            result_torch = op_torch(x_torch, y_torch)
            loss_torch = result_torch.sum()
            loss_torch.backward()
            grad_y_torch = y_torch.grad.cpu().numpy()
            print(f"{grad_y_torch}")

            jax_vs_torch = jnp.allclose(grad_y_jax, grad_y_torch, rtol=1e-5)
            print(f"✓ JAX vs KeOps PyTorch match: {jax_vs_torch}")
            if not jax_vs_torch:
                all_match = False
        except Exception as e:
            print(f"PyTorch backend failed: {e}")

    # Compare
    match = jnp.allclose(grad_y_jax, grad_y_keops_jax, rtol=1e-5)
    print(f"\n✓ JAX vs KeOps JAX match: {match}")
    if not match:
        all_match = False
        print(f"  Max difference: {jnp.max(jnp.abs(grad_y_jax - grad_y_keops_jax)):.6f}")

    # Also compute gradient w.r.t. x for comparison
    def loss_x(x):
        return jnp.sum(op_jax(x, y))

    grad_x_keops_jax = jax.grad(loss_x)(x)

    print(f"\nFor comparison, gradient w.r.t. x:\n{grad_x_keops_jax}")
    print(f"Sum of grad_x and grad_y (should be ~0): {grad_x_keops_jax + grad_y_keops_jax}")

    return all_match


def test_different_formulas():
    """Test various formulas to ensure gradient implementation is robust."""
    print("\n" + "=" * 70)
    print("  Test: Different Formulas")
    print("=" * 70)

    x = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    y = jnp.array([[0.0, 0.0]], dtype=jnp.float32)

    formulas = [
        ("SqDist(x, y)", "Squared distance"),
        ("(x-y)**2", "Elementwise squared diff"),
        ("Exp(-SqDist(x, y))", "Gaussian kernel"),
    ]

    all_passed = True

    for formula, description in formulas:
        print(f"\n{description}: {formula}")
        print("-" * 50)

        # KeOps JAX
        aliases = ["x=Vi(2)", "y=Vj(2)"]
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)

        def loss(x):
            return jnp.sum(op(x, y))

        # Compute gradient
        grad_keops = jax.grad(loss)(x)

        # Compare with pure JAX
        if "Exp" in formula:
            # Pure JAX Gaussian
            def pure_jax_gaussian(x):
                diff = x[:, None, :] - y[None, :, :]
                sqdist = jnp.sum(diff ** 2, axis=2)
                return jnp.sum(jnp.exp(-sqdist))

            grad_jax = jax.grad(pure_jax_gaussian)(x)
        elif "SqDist" in formula or "(x-y)**2" in formula:
            # Pure JAX version
            def pure_jax_loss(x):
                diff = x[:, None, :] - y[None, :, :]
                sqdist = jnp.sum(diff ** 2, axis=2)
                return jnp.sum(sqdist)

            grad_jax = jax.grad(pure_jax_loss)(x)
        else:
            print("Skipping pure JAX comparison for this formula")
            continue

        match = jnp.allclose(grad_jax, grad_keops, rtol=1e-4)

        print(f"JAX gradient:\n{grad_jax}")
        print(f"KeOps gradient:\n{grad_keops}")
        print(f"Match: {match}")

        # Also compare with PyTorch if available
        if HAS_TORCH:
            try:
                x_torch = torch.tensor(np.array(x), dtype=torch.float32, requires_grad=True, device='cuda')
                y_torch = torch.tensor(np.array(y), dtype=torch.float32, device='cuda')
                op_torch = Genred_torch(formula, aliases, reduction_op='Sum', axis=1)

                result_torch = op_torch(x_torch, y_torch)
                loss_torch = result_torch.sum()
                loss_torch.backward()
                grad_torch = x_torch.grad.cpu().numpy()

                torch_match = jnp.allclose(grad_jax, grad_torch, rtol=1e-4)
                print(f"PyTorch gradient:\n{grad_torch}")
                print(f"JAX vs PyTorch match: {torch_match}")
            except Exception as e:
                print(f"PyTorch comparison failed: {e}")

        if not match:
            print(f"Max diff: {jnp.max(jnp.abs(grad_jax - grad_keops)):.6f}")
            all_passed = False

    return all_passed


def run_all_comparison_tests():
    """Run all comparison tests."""

    tests = [
        ("Forward Pass", test_forward_pass_comparison),
        ("Gradient Comparison", test_gradient_comparison),
        ("Gradient w.r.t. y", test_gradient_wrt_vj),
        ("Different Formulas", test_different_formulas),
    ]

    results = {}

    for test_name, test_fn in tests:
        try:
            passed = test_fn()
            results[test_name] = "PASS" if passed else "FAIL"
            print(f"\n{'✓ PASS' if passed else '✗ FAIL'}: {test_name}")
        except Exception as e:
            results[test_name] = f"Exception: {e}"
            print(f"\n✗ {test_name} - Exception: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("  Comparison Test Summary")
    print("=" * 70)

    for test_name, result in results.items():
        status = "✓ PASS" if result == "PASS" else f"✗ FAIL"
        print(f"{status}: {test_name}")

    passed_count = sum(1 for r in results.values() if r == "PASS")
    total_count = len(results)

    print(f"\nPassed {passed_count}/{total_count} tests")

    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_comparison_tests()
    exit(0 if success else 1)