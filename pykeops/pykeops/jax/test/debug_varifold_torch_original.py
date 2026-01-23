import torch
import numpy as np
from pykeops.torch import LazyTensor, Genred


def run_single_batch(B, save_dict):
    print(f"\n{'-' * 20} Generating Batch Size B={B} {'-' * 20}")

    # 1. Setup Deterministic Data
    N, M, D = 5, 10, 3

    np.random.seed(42)
    x_np = np.random.randn(B, N, D).astype(np.float32)
    y_np = np.random.randn(B, M, D).astype(np.float32)
    u_np = np.random.randn(B, N, D).astype(np.float32)
    v_np = np.random.randn(B, M, D).astype(np.float32)
    u_np = u_np / np.linalg.norm(u_np, axis=-1, keepdims=True)
    v_np = v_np / np.linalg.norm(v_np, axis=-1, keepdims=True)

    x = torch.from_numpy(x_np).to(device='cuda', dtype=torch.float32).requires_grad_(True)
    y = torch.from_numpy(y_np).to(device='cuda', dtype=torch.float32).requires_grad_(True)
    u = torch.from_numpy(u_np).to(device='cuda', dtype=torch.float32).requires_grad_(True)
    v = torch.from_numpy(v_np).to(device='cuda', dtype=torch.float32).requires_grad_(True)

    gamma_s = torch.tensor(0.25, device='cuda')
    gamma_n = torch.tensor(0.75, device='cuda')

    # ------------------------------------------------------------------
    # 2. Reference (Pure PyTorch)
    # ------------------------------------------------------------------
    x_i, y_j = x.unsqueeze(2), y.unsqueeze(1)
    u_i, v_j = u.unsqueeze(2), v.unsqueeze(1)

    sq_dist = torch.sum((x_i - y_j) ** 2, dim=-1)
    dot_prod = torch.sum(u_i * v_j, dim=-1)
    K = torch.exp(-sq_dist * gamma_s) * torch.exp(dot_prod * gamma_n)

    # K is shape (B, N, M). Sum over M (dim 2).
    ref_res = K.sum(dim=2)

    # ------------------------------------------------------------------
    # 3. KeOps LazyTensor
    # ------------------------------------------------------------------
    x_L = LazyTensor(x.view(B, N, 1, D))
    y_L = LazyTensor(y.view(B, 1, M, D))
    u_L = LazyTensor(u.view(B, N, 1, D))
    v_L = LazyTensor(v.view(B, 1, M, D))

    sq_dist_L = x_L.sqdist(y_L)
    dot_prod_L = (u_L * v_L).sum(-1)
    K_L = (-sq_dist_L * gamma_s).exp() * (dot_prod_L * gamma_n).exp()

    # FIX: Use dim=2 for batched tensors (B, N, M)
    lazy_res = K_L.sum(dim=2)

    if lazy_res.dim() == 3 and lazy_res.shape[-1] == 1:
        lazy_res = lazy_res.squeeze(-1)

    # ------------------------------------------------------------------
    # 4. Detailed Fingerprints
    # ------------------------------------------------------------------
    print(f"  PyTorch Reference Shape: {ref_res.shape}")

    # Check indices: (0,0), (1,0) [if B>1], and (-1,-1)
    indices = [(0, 0), (-1, -1)]
    if B > 1:
        indices.insert(1, (1, 0))

    print(f"  {'Index':<12} {'Ref Value':<15} {'Lazy Value':<15}")
    for b_idx, n_idx in indices:
        val_ref = ref_res[b_idx, n_idx].item()
        val_lazy = lazy_res[b_idx, n_idx].item()
        print(f"  [{b_idx},{n_idx:<3}]      {val_ref:<15.5f} {val_lazy:<15.5f}")

    # Store for JAX
    save_dict[f'x_B{B}'] = x.detach().cpu().numpy()
    save_dict[f'y_B{B}'] = y.detach().cpu().numpy()
    save_dict[f'u_B{B}'] = u.detach().cpu().numpy()
    save_dict[f'v_B{B}'] = v.detach().cpu().numpy()
    save_dict[f'ref_B{B}'] = ref_res.detach().cpu().numpy()
    save_dict[f'lazy_B{B}'] = lazy_res.detach().cpu().numpy()


def run_debug():
    print("=" * 60)
    print("DEBUG: PyTorch Batched Varifold (Multi-Point Check)")
    print("=" * 60)

    final_data = {}
    run_single_batch(2, final_data)
    run_single_batch(3, final_data)

    np.savez('debug_workspace.npz', **final_data)
    print("\n✅ All results saved to 'debug_workspace.npz'")


if __name__ == "__main__":
    run_debug()