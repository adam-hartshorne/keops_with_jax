import os
# Configuration
os.environ['PYKEOPS_JAX_MODE'] = '1'
import sys
import jax
import jax.numpy as jnp
import numpy as np
from pykeops.jax import LazyTensor, Genred


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Force full output
np.set_printoptions(threshold=np.inf, precision=4, suppress=True)

def run_full_comparison():
    B, N, M, D = 2, 3, 2, 3
    np.random.seed(0)

    x_np = np.random.randn(B, N, D).astype('float32')
    x_np[1, :, :] *= 10.0
    y_np = np.zeros((B, M, D), dtype='float32')

    x, y = jnp.array(x_np), jnp.array(y_np)

    # TEST: See if Genred behaves differently if we pass 2D vs 3D
    formula = "Sum(SqDist(a, b))"
    aliases = [f"a=Vi({D})", f"b=Vj({D})"]
    op_genred = Genred(formula, aliases, reduction_op='Sum', axis=1)

    # Try calling it
    res_genred = op_genred(x, y)

    print(f"Input shape: {x.shape}") # Should be (2, 3, 3)
    for b in range(B):
        print(f"Batch {b} Result:\n{res_genred[b]}")

if __name__ == "__main__":
    run_full_comparison()