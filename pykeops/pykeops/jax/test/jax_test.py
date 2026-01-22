import os
os.environ['PYKEOPS_JAX_MODE'] = '1'
import sys
import jax
import jax.numpy as jnp
import numpy as np
from pykeops.jax import Genred


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
np.set_printoptions(threshold=np.inf, precision=5, suppress=True)

def run_jax_test():
    N, M, D, B = 10, 5, 3, 2  # D is 3
    np.random.seed(0)
    x_np = np.random.randn(N, D).astype('float32')
    y_np = np.random.randn(M, D).astype('float32')
    xb_np = np.random.randn(B, N, D).astype('float32')
    yb_np = np.random.randn(B, M, D).astype('float32')

    # MATCHING FORMULA AND ALIASES (D=3)
    formula = "Sum((a-b)**2)"
    aliases = [f"a=Vi({D})", f"b=Vj({D})"]
    op = Genred(formula, aliases, reduction_op='Sum', axis=1)

    print("=" * 60)
    print("TEST 1: NON-BATCHED (2D) - JAX")
    print("=" * 60)
    res_jax_2d = op(jnp.array(x_np), jnp.array(y_np))
    print(f"Full Result:\n{np.array(res_jax_2d).flatten()}")

    print("\n" + "=" * 60)
    print("TEST 2: BATCHED (3D) - JAX")
    print("=" * 60)
    res_jax_3d = op(jnp.array(xb_np), jnp.array(yb_np))
    print(f"Full Result:\n{np.array(res_jax_3d)}")

if __name__ == "__main__":
    run_jax_test()