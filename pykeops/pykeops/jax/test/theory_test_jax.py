import os
import sys

# Set JAX mode for the Platinum FFI
os.environ['PYKEOPS_JAX_MODE'] = '1'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jax.numpy as jnp
import numpy as np
from pykeops.jax import Genred


def run_jax_theory_test():
    # Matches the standard test dimensions exactly
    B, N, M, D = 2, 2, 3, 3

    x_np = np.zeros((B, N, D), dtype='float32')
    x_np[0, :, :] = 1.0  # SqDist vs 0 = 3.0
    x_np[1, :, :] = 10.0  # SqDist vs 0 = 300.0
    y_np = np.zeros((B, M, D), dtype='float32')

    x, y = jnp.array(x_np), jnp.array(y_np)

    # USE THE STRING FORMULA - Matches theory_test_standard.py
    formula = "Sum((x-y)**2)"
    aliases = [f"x=Vi({D})", f"y=Vj({D})"]

    # In JAX Genred, axis=1 reduces over Vj (the M dimension)
    op = Genred(formula, aliases, reduction_op='Sum', axis=1)
    res_jax = op(x, y)

    print("=== JAX KEOPS RESULTS (PLATINUM FFI) ===")
    print(np.array(res_jax))

    # Verification logic
    # Batch 0 should be (1^2 + 1^2 + 1^2) * M(3) = 9.0
    # Batch 1 should be (10^2 + 10^2 + 10^2) * M(3) = 900.0
    val_b1 = np.array(res_jax)[1, 0, 0]
    if abs(val_b1 - 900.0) < 1e-3:
        print("\n✅ JAX Match: Theory DISPROVED (Offsets are correct)")
    else:
        print(f"\n❌ JAX Mismatch: Theory PROVED (Got {val_b1}, expected 900.0)")


if __name__ == "__main__":
    run_jax_theory_test()