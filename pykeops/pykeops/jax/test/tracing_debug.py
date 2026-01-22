import os

os.environ['PYKEOPS_JAX_MODE'] = '1'
import jax
import jax.numpy as jnp
from pykeops.jax import Genred


def trace_batching():
    # Use Primes to catch stride errors
    B, N, M, D = 2, 3, 5, 1

    # Batch 0: x=1, y=0 -> Res = 1*M = 5
    # Batch 1: x=10, y=0 -> Res = 100*M = 500
    x = jnp.array([[[1.0]] * N, [[10.0]] * N], dtype=jnp.float32)
    y = jnp.zeros((B, M, D), dtype=jnp.float32)

    formula = "SqDist(x, y)"
    aliases = ["x=Vi(1)", "y=Vj(1)"]
    op = Genred(formula, aliases, reduction_op='Sum', axis=1)

    print(f"Executing B={B}, N={N}, M={M}")
    res = op(x, y)

    print("\n--- RESULTS ---")
    print(f"Batch 0 (Expect 5.0):   {res[0, 0, 0]}")
    print(f"Batch 1 (Expect 500.0): {res[1, 0, 0]}")

    if res[0, 0, 0] == 500.0:
        print("\nDIAGNOSIS: Offsets are shifted. Batch 0 is reading Batch 1's memory.")
    elif res[1, 0, 0] == 0.0:
        print("\nDIAGNOSIS: Stride error. Batch 1 is pointing to out-of-bounds/zero memory.")


if __name__ == "__main__":
    trace_batching()