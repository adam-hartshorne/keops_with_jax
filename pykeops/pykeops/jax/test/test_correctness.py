import os
os.environ['PYKEOPS_JAX_MODE'] = '1'

import unittest
import jax
import jax.numpy as jnp
import numpy as np
from pykeops.jax import Genred, LazyTensor

# --- Ground Truth Implementation ---
def pure_jax_rbf(x, y, b, sigma=1.0):
    """
    Standard RBF: sum_j exp(-|x_i - y_j|^2 / 2sigma^2) * b_j
    Supports both 2D (N, D) and 3D (B, N, D) inputs via broadcasting.
    """
    # x: (..., N, D) -> (..., N, 1, D)
    # y: (..., M, D) -> (..., 1, M, D)
    diff = jnp.expand_dims(x, -2) - jnp.expand_dims(y, -3)
    sq_dist = jnp.sum(diff ** 2, axis=-1)
    K = jnp.exp(-sq_dist / (2 * sigma**2))
    # b: (..., M, 1) or (..., M, E)
    return K @ b

class TestKeOpsCorrectness(unittest.TestCase):

    def setUp(self):
        # Deterministic keys
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)

        self.N, self.M, self.D, self.E = 100, 50, 3, 2
        self.B = 5 # Batch size for 3D tests
        self.sigma = 1.5

        # Unbatched Data (2D)
        self.x2d = jax.random.normal(k1, (self.N, self.D))
        self.y2d = jax.random.normal(k2, (self.M, self.D))
        self.b2d = jax.random.normal(k3, (self.M, self.E))

        # Batched Data (3D)
        self.x3d = jax.random.normal(k1, (self.B, self.N, self.D))
        self.y3d = jax.random.normal(k2, (self.B, self.M, self.D))
        self.b3d = jax.random.normal(k3, (self.B, self.M, self.E))

    def assert_close(self, a, b, atol=1e-5, msg=""):
        diff = jnp.max(jnp.abs(a - b))
        if not jnp.allclose(a, b, atol=atol):
            self.fail(f"{msg} | Max diff: {diff:.2e}")

    # --- Test 1: Genred Basic ---
    def test_genred_forward(self):
        print("\n[Genred] Testing Forward Pass...")
        formula = "Sum(Exp(-SqDist(x, y) * s) * b)"
        aliases = [
            f"x=Vi({self.D})",
            f"y=Vj({self.D})",
            f"b=Vj({self.E})",
            "s=Pm(1)"
        ]
        op = Genred(formula, aliases, axis=1)

        s_val = jnp.array([1.0 / (2 * self.sigma**2)])
        res_keops = op(self.x2d, self.y2d, self.b2d, s_val)
        res_jax = pure_jax_rbf(self.x2d, self.y2d, self.b2d, self.sigma)

        self.assert_close(res_keops, res_jax, msg="Genred Forward Failed")

    # --- Test 2: LazyTensor Basic ---
    def test_lazytensor_forward(self):
        print("[LazyTensor] Testing Forward Pass...")
        # x_i: (N, 1, D)
        xi = LazyTensor(self.x2d[:, None, :])
        # y_j: (1, M, D)
        yj = LazyTensor(self.y2d[None, :, :])
        # b_j: (1, M, E)
        bj = LazyTensor(self.b2d[None, :, :])

        s = 1.0 / (2 * self.sigma**2)

        # Symbolic computation
        d_ij = ((xi - yj) ** 2).sum(-1)
        K_ij = (-d_ij * s).exp()
        res_lazy = (K_ij * bj).sum(1)

        res_jax = pure_jax_rbf(self.x2d, self.y2d, self.b2d, self.sigma)

        # LazyTensor output might need squeeze depending on internal broadcasting
        # The sum(1) reduces axis 1, leaving (N, E)
        self.assert_close(res_lazy, res_jax, msg="LazyTensor Forward Failed")

    # --- Test 3: Gradients ---
    def test_gradients(self):
        print("[Gradients] Testing jax.grad on Genred...")
        formula = "SqDist(x, y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)

        def loss_fn(x, y):
            # Sum output to scalar
            return jnp.sum(op(x, y))

        def loss_fn_jax(x, y):
            diff = x[:, None, :] - y[None, :, :]
            sq_dist = jnp.sum(diff ** 2, axis=-1)
            return jnp.sum(sq_dist)

        grad_keops = jax.grad(loss_fn)(self.x2d, self.y2d)
        grad_jax = jax.grad(loss_fn_jax)(self.x2d, self.y2d)

        self.assert_close(grad_keops, grad_jax, msg="Gradient Failed")

    # --- Test 4: JIT Compilation ---
    def test_jit(self):
        print("[JIT] Testing jax.jit...")
        formula = "SqDist(x, y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)

        @jax.jit
        def compiled_op(x, y):
            return op(x, y)

        # First run (compilation)
        res1 = compiled_op(self.x2d, self.y2d)
        # Second run (cached)
        res2 = compiled_op(self.x2d, self.y2d)

        self.assert_close(res1, res2, msg="JIT Result Inconsistent")

    # --- Test 5: Batched Inputs (3D) ---
    def test_batched_3d(self):
        print("[Batching] Testing 3D Input (B, N, D)...")
        formula = "SqDist(x, y)"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})"]
        # Note: KeOps automatically detects batch dimensions
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)

        res_keops = op(self.x3d, self.y3d)

        # JAX Ground Truth
        diff = self.x3d[:, :, None, :] - self.y3d[:, None, :, :]
        res_jax = jnp.sum(jnp.sum(diff**2, axis=-1), axis=2, keepdims=True)

        # KeOps output for Sum reduction is usually (B, N, 1)
        self.assert_close(res_keops, res_jax, msg="3D Batching Failed")

if __name__ == '__main__':
    unittest.main()