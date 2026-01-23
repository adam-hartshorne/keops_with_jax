import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
from pykeops.jax import Genred, LazyTensor



def run_full_comparison():
    B, N, M, D = 2, 3, 2, 3
    np.random.seed(0)

    # Generate data: Batch 1 significantly larger to catch offset errors
    x_np = np.random.randn(B, N, D).astype('float32')
    x_np[1, :, :] *= 10.0
    y_np = np.zeros((B, M, D), dtype='float32')

    x, y = jnp.array(x_np), jnp.array(y_np)

    # 1. Lazy Definition
    xi, yj = LazyTensor(x[:, :, None, :]), LazyTensor(y[:, None, :, :])
    res_lazy = ((xi - yj) ** 2).sum(-1).sum(axis=2)

    # 2. Genred Definition
    formula = "Sum(SqDist(a, b))"
    aliases = [f"a=Vi({D})", f"b=Vj({D})"]
    op_genred = Genred(formula, aliases, reduction_op='Sum', axis=1)
    res_genred = op_genred(x, y)

    print("=" * 60)
    print("FULL MATRIX COMPARISON")
    print("=" * 60)

    for b in range(B):
        print(f"\n--- Batch {b} ---")
        lazy_b = np.array(res_lazy[b])
        genred_b = np.array(res_genred[b])

        print(f"LazyTensor:\n{lazy_b}")
        print(f"Genred:\n{genred_b}")

        if np.allclose(lazy_b, genred_b, atol=1e-4):
            print(f"Result: ✅ Batch {b} MATCH")
        else:
            print(f"Result: ❌ Batch {b} MISMATCH")

    diff = np.abs(np.array(res_lazy) - np.array(res_genred)).max()
    print(f"\nGlobal Max Difference: {diff:.6e}")


if __name__ == "__main__":
    run_full_comparison()