import numpy as np
import jax
import jax.numpy as jnp
import torch
from pykeops.jax import LazyTensor as LazyTensor_jax
from pykeops.torch import LazyTensor as LazyTensor_torch


def test_at_scale(B, N, M, name=""):
    """Test JAX vs PyTorch KeOps at specific scale."""
    print(f"\n{'=' * 60}")
    print(f"Testing: {name} (B={B}, N={N}, M={M})")
    print(f"{'=' * 60}")

    np.random.seed(42)

    # Create data
    x_centers_np = np.random.randn(B, N, 3).astype(np.float32) * 0.1
    x_normals_np = np.random.randn(B, N, 3).astype(np.float32)
    x_normals_np = x_normals_np / np.linalg.norm(x_normals_np, axis=-1, keepdims=True)

    y_centers_np = np.random.randn(B, M, 3).astype(np.float32) * 0.1
    y_normals_np = np.random.randn(B, M, 3).astype(np.float32)
    y_normals_np = y_normals_np / np.linalg.norm(y_normals_np, axis=-1, keepdims=True)

    areas_x_np = np.abs(np.random.randn(B, N, 1).astype(np.float32)) * 0.01 + 0.001
    areas_y_np = np.abs(np.random.randn(B, M, 1).astype(np.float32)) * 0.01 + 0.001
    kernel_params_np = np.array([1.0, 1.0], dtype=np.float32)

    # PyTorch version
    def torch_forward_backward(x_centers_np):
        x_centers = torch.tensor(x_centers_np, device='cuda', requires_grad=True)
        x_normals = torch.tensor(x_normals_np, device='cuda')
        y_centers = torch.tensor(y_centers_np, device='cuda')
        y_normals = torch.tensor(y_normals_np, device='cuda')
        areas_x = torch.tensor(areas_x_np, device='cuda')
        areas_y = torch.tensor(areas_y_np, device='cuda')
        kp = torch.tensor(kernel_params_np, device='cuda')

        gamma, gamma_1 = kp[0], kp[1]

        # vv kernel
        x_i = LazyTensor_torch(x_centers[:, :, None, :])
        x_j = LazyTensor_torch(x_centers[:, None, :, :])
        u_i = LazyTensor_torch(x_normals[:, :, None, :])
        u_j = LazyTensor_torch(x_normals[:, None, :, :])
        D2 = x_i.sqdist(x_j)
        ss = (u_i * u_j).sum()
        vv = (-D2 * gamma).exp() * (ss * gamma_1).exp()

        # ww kernel
        y_i = LazyTensor_torch(y_centers[:, :, None, :])
        y_j = LazyTensor_torch(y_centers[:, None, :, :])
        v_i = LazyTensor_torch(y_normals[:, :, None, :])
        v_j = LazyTensor_torch(y_normals[:, None, :, :])
        D2_w = y_i.sqdist(y_j)
        ss_w = (v_i * v_j).sum()
        ww = (-D2_w * gamma).exp() * (ss_w * gamma_1).exp()

        # vw kernel
        x_i2 = LazyTensor_torch(x_centers[:, :, None, :])
        y_j2 = LazyTensor_torch(y_centers[:, None, :, :])
        u_i2 = LazyTensor_torch(x_normals[:, :, None, :])
        v_j2 = LazyTensor_torch(y_normals[:, None, :, :])
        D2_vw = x_i2.sqdist(y_j2)
        ss_vw = (u_i2 * v_j2).sum()
        vw = (-D2_vw * gamma).exp() * (ss_vw * gamma_1).exp()

        term1 = ((vv @ areas_x) * areas_x).sum(1)
        term2 = ((ww @ areas_y) * areas_y).sum(1)
        term3 = -2.0 * ((vw @ areas_y) * areas_x).sum(1)

        loss = (term1 + term2 + term3).sum()
        loss.backward()

        return loss.item(), x_centers.grad.cpu().numpy()

    # JAX version
    def jax_loss_fn(x_centers):
        x_normals = jnp.array(x_normals_np)
        y_centers = jnp.array(y_centers_np)
        y_normals = jnp.array(y_normals_np)
        areas_x = jnp.array(areas_x_np)
        areas_y = jnp.array(areas_y_np)
        kp = jnp.array(kernel_params_np)

        gamma, gamma_1 = kp[0], kp[1]

        # vv kernel
        x_i = LazyTensor_jax(x_centers[:, :, None, :])
        x_j = LazyTensor_jax(x_centers[:, None, :, :])
        u_i = LazyTensor_jax(x_normals[:, :, None, :])
        u_j = LazyTensor_jax(x_normals[:, None, :, :])
        D2 = x_i.sqdist(x_j)
        ss = (u_i * u_j).sum()
        vv = (-D2 * gamma).exp() * (ss * gamma_1).exp()

        # ww kernel
        y_i = LazyTensor_jax(y_centers[:, :, None, :])
        y_j = LazyTensor_jax(y_centers[:, None, :, :])
        v_i = LazyTensor_jax(y_normals[:, :, None, :])
        v_j = LazyTensor_jax(y_normals[:, None, :, :])
        D2_w = y_i.sqdist(y_j)
        ss_w = (v_i * v_j).sum()
        ww = (-D2_w * gamma).exp() * (ss_w * gamma_1).exp()

        # vw kernel
        x_i2 = LazyTensor_jax(x_centers[:, :, None, :])
        y_j2 = LazyTensor_jax(y_centers[:, None, :, :])
        u_i2 = LazyTensor_jax(x_normals[:, :, None, :])
        v_j2 = LazyTensor_jax(y_normals[:, None, :, :])
        D2_vw = x_i2.sqdist(y_j2)
        ss_vw = (u_i2 * v_j2).sum()
        vw = (-D2_vw * gamma).exp() * (ss_vw * gamma_1).exp()

        term1 = ((vv @ areas_x) * areas_x).sum(1)
        term2 = ((ww @ areas_y) * areas_y).sum(1)
        term3 = -2.0 * ((vw @ areas_y) * areas_x).sum(1)

        return (term1 + term2 + term3).sum()

    # Run tests
    try:
        loss_torch, grad_torch = torch_forward_backward(x_centers_np)
        print(f"PyTorch - Loss: {loss_torch:.6f}, Grad norm: {np.linalg.norm(grad_torch):.6f}")
    except Exception as e:
        print(f"PyTorch FAILED: {e}")
        return

    try:
        loss_jax = float(jax_loss_fn(jnp.array(x_centers_np)))
        grad_jax = np.array(jax.grad(jax_loss_fn)(jnp.array(x_centers_np)))
        print(f"JAX     - Loss: {loss_jax:.6f}, Grad norm: {np.linalg.norm(grad_jax):.6f}")
    except Exception as e:
        print(f"JAX FAILED: {e}")
        return

    # Compare
    loss_diff = abs(loss_torch - loss_jax)
    grad_diff = np.abs(grad_torch - grad_jax).max()
    grad_rel_diff = grad_diff / (np.abs(grad_torch).max() + 1e-10)

    print(f"\nComparison:")
    print(f"  Loss diff: {loss_diff:.2e}")
    print(f"  Grad max diff: {grad_diff:.2e}")
    print(f"  Grad relative diff: {grad_rel_diff:.2e}")
    print(f"  Any NaN (torch): {np.any(np.isnan(grad_torch))}")
    print(f"  Any NaN (jax): {np.any(np.isnan(grad_jax))}")

    if loss_diff < 1e-3 and grad_rel_diff < 1e-2:
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")


# Run at different scales
test_at_scale(2, 50, 40, "Small")
test_at_scale(2, 500, 400, "Medium")
test_at_scale(2, 5000, 4000, "Large")
test_at_scale(4, 10000, 5000, "Very Large")
test_at_scale(21, 98776, 25000, "Your actual size")  # This might OOM or take long