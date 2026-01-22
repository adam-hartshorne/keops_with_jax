import os
os.environ['PYKEOPS_JAX_MODE'] = '1'
import sys
import jax
import jax.numpy as jnp
import numpy as np
from pykeops.jax import Genred, LazyTensor


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
np.set_printoptions(precision=4, suppress=True)

def run_jax_full_comparison():
    print("=" * 80)
    print("JAX PLATINUM VERIFICATION: GENRED VS LAZYTENSOR")
    print("=" * 80)

    B, N, M, D = 2, 3, 2, 3
    np.random.seed(0)
    x_np = np.random.randn(B, N, D).astype('float32')
    y_np = np.random.randn(B, M, D).astype('float32')
    x, y = jnp.array(x_np), jnp.array(y_np)

    # --- 1. GENRED ---
    op_genred = Genred("Sum(SqDist(a, b))", [f"a=Vi({D})", f"b=Vj({D})"], 'Sum', 1)
    res_genred, vjp_genred = jax.vjp(lambda x_in: op_genred(x_in, y), x)
    grad_genred = vjp_genred(jnp.ones_like(res_genred))[0]

    # --- 2. LAZYTENSOR ---
    def lazy_op(x_in):
        xi, yj = LazyTensor(x_in[:, :, None, :]), LazyTensor(y[:, None, :, :])
        return ((xi - yj)**2).sum(-1).sum(axis=2)
    res_lazy, vjp_lazy = jax.vjp(lazy_op, x)
    grad_lazy = vjp_lazy(jnp.ones_like(res_lazy))[0]

    for b in range(B):
        print(f"\n--- Batch {b} ---")
        print(f"Fwd (Genred):\n{np.array(res_genred[b])}")
        print(f"Fwd (Lazy):  \n{np.array(res_lazy[b])}")
        print(f"Grad (Genred):\n{np.array(grad_genred[b])}")
        print(f"Grad (Lazy):  \n{np.array(grad_lazy[b])}")

if __name__ == '__main__':
    run_jax_full_comparison()