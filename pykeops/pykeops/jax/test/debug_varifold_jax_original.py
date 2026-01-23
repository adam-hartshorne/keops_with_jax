import os

os.environ['PYKEOPS_JAX_MODE'] = '1'

import jax
import jax.numpy as jnp
import numpy as np
from pykeops.jax import LazyTensor, Genred


def run_single_batch(B, data):
    print(f"\n{'-' * 20} Testing Batch Size B={B} {'-' * 20}")

    try:
        x = jnp.array(data[f'x_B{B}'])
        y = jnp.array(data[f'y_B{B}'])
        u = jnp.array(data[f'u_B{B}'])
        v = jnp.array(data[f'v_B{B}'])
        torch_ref = data[f'ref_B{B}']
        torch_lazy = data[f'lazy_B{B}']
    except KeyError:
        print(f"❌ Missing data for Batch {B}")
        return

    gamma_s = 0.25
    gamma_n = 0.75

    # ------------------------------------------------------------------
    # JAX LazyTensor
    # ------------------------------------------------------------------
    x_L = LazyTensor(x[:, :, None, :])
    y_L = LazyTensor(y[:, None, :, :])
    u_L = LazyTensor(u[:, :, None, :])
    v_L = LazyTensor(v[:, None, :, :])

    sq_dist_L = x_L.sqdist(y_L)
    dot_prod_L = (u_L * v_L).sum(-1)
    K_L = (-sq_dist_L * gamma_s).exp() * (dot_prod_L * gamma_n).exp()

    # axis=2 for (Batch, N, M)
    jax_lazy = K_L.sum(axis=2)

    if jax_lazy.ndim == 3 and jax_lazy.shape[-1] == 1:
        jax_lazy = jax_lazy.squeeze(-1)

    jax_lazy_np = np.array(jax_lazy)

    # ------------------------------------------------------------------
    # Multi-Point Fingerprint Check
    # ------------------------------------------------------------------
    print(f"  {'Index':<12} {'JAX Value':<15} {'PyTorch Value':<15} {'Diff':<15}")

    indices = [(0, 0), (-1, -1)]
    if B > 1:
        indices.insert(1, (1, 0))

    for b_idx, n_idx in indices:
        val_jax = jax_lazy_np[b_idx, n_idx]
        val_torch = torch_lazy[b_idx, n_idx]
        diff = abs(val_jax - val_torch)

        # Color code the diff
        if diff < 1e-4:
            status = ""
        else:
            status = "❌"

        print(f"  [{b_idx},{n_idx:<3}]      {val_jax:<15.5f} {val_torch:<15.5f} {diff:<15.1e} {status}")

    total_diff = np.abs(jax_lazy_np - torch_ref).max()
    if total_diff < 1e-4:
        print(f"\n✅ JAX Batch={B} PASSED (Max Diff: {total_diff:.1e})")
    else:
        print(f"\n❌ JAX Batch={B} FAILED (Max Diff: {total_diff:.1e})")


def run_debug():
    print("=" * 60)
    print("DEBUG: JAX Batched Varifold (Multi-Point Check)")
    print("=" * 60)

    try:
        data = np.load('debug_workspace.npz')
        print("Loaded 'debug_workspace.npz'")
    except FileNotFoundError:
        print("❌ 'debug_workspace.npz' not found.")
        return

    run_single_batch(2, data)
    run_single_batch(3, data)


if __name__ == "__main__":
    run_debug()