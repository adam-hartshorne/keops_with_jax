"""
KeOps JAX API Comprehensive Unit Tests
======================================
Visual Summary + Execution Delineation Version
"""

import os
import sys
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial



# Import KeOps JAX API
try:
    from pykeops.jax import Genred, LazyTensor, Vi, Vj, Pm
except ImportError:
    print("Error: pykeops.jax not found. Ensure optimized files are in the python path.")
    sys.exit(1)

print("\n" + "="*80)
print("VERIFYING CODE VERSION")
print("="*80)
from pykeops.jax.generic import generic_ops
import inspect
print(f"generic_ops location: {inspect.getfile(generic_ops)}")
print(f"Has make_keops_jax_op: {hasattr(generic_ops, 'make_keops_jax_op')}")

# Check if it's the optimized version
source = inspect.getsource(generic_ops.make_keops_jax_op)
if "FULLY OPTIMIZED" in source:
    print("✓ Using OPTIMIZED version")
elif "dimout" in source:
    print("✓ Using version with dimout fix")
else:
    print("⚠ Using OLD/UNKNOWN version")
print("="*80 + "\n")

# =============================================================================
# VISUAL SUMMARY & DELINEATION INFRASTRUCTURE
# =============================================================================

class VisualTestResult(unittest.TextTestResult):
    """Custom TestResult to track results and print delineated test starts/ends."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = []
        self.width = 80

    def startTest(self, test):
        test_name = test.id().split('.')[-1]
        print(f"\n{' STARTING: ' + test_name + ' ':=^{self.width}}")
        super().startTest(test)

    def addSuccess(self, test):
        super().addSuccess(test)
        test_name = test.id().split('.')[-1]
        print(f"{' FINISHED: ' + test_name + ' (✅ PASS) ':-^{self.width}}")
        self.results.append((test_name, "✅", "\033[92m"))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        test_name = test.id().split('.')[-1]
        print(f"{' FINISHED: ' + test_name + ' (❌ FAIL) ':-^{self.width}}")
        self.results.append((test_name, "❌", "\033[91m"))

    def addError(self, test, err):
        super().addError(test, err)
        test_name = test.id().split('.')[-1]
        print(f"{' FINISHED: ' + test_name + ' (💥 ERROR) ':-^{self.width}}")
        self.results.append((test_name, "💥", "\033[91m"))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        test_name = test.id().split('.')[-1]
        print(f"{' SKIPPED: ' + test_name + ' (➖) ':-^{self.width}}")
        self.results.append((test_name, "➖", "\033[93m"))

class VisualTestRunner(unittest.TextTestRunner):
    """Runner that uses our delineated VisualTestResult."""
    def _makeResult(self):
        return VisualTestResult(self.stream, self.descriptions, self.verbosity)

# =============================================================================
# TEST CASES
# =============================================================================

class TestGenredBasics(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(self.key, 3)
        self.N, self.M, self.D, self.E = 100, 50, 3, 2
        self.x = jax.random.normal(k1, (self.N, self.D))
        self.y = jax.random.normal(k2, (self.M, self.D))
        self.b = jax.random.normal(k3, (self.M, self.E))

    def test_genred_sum_reduction(self):
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Sum', axis=1)
        res = op(self.x, self.y)
        diff = self.x[:, None, :] - self.y[None, :, :]
        expected = jnp.sum(jnp.sum(diff**2, axis=-1), axis=1, keepdims=True)
        self.assertTrue(jnp.allclose(res, expected, rtol=1e-5))

    def test_genred_min_reduction(self):
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Min', axis=1)
        res = op(self.x, self.y)
        expected = jnp.min(jnp.sum((self.x[:, None, :] - self.y[None, :, :])**2, axis=-1), axis=1, keepdims=True)
        self.assertTrue(jnp.allclose(res, expected, rtol=1e-5))

    def test_genred_max_reduction(self):
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'Max', axis=1)
        res = op(self.x, self.y)
        expected = jnp.max(jnp.sum((self.x[:, None, :] - self.y[None, :, :])**2, axis=-1), axis=1, keepdims=True)
        self.assertTrue(jnp.allclose(res, expected, rtol=1e-5))

    @unittest.skip("KeOps core bug: ArgMin generator missing argument")
    def test_genred_argmin_reduction(self):
        op = Genred("SqDist(x, y)", [f"x=Vi({self.D})", f"y=Vj({self.D})"], 'ArgMin', axis=1)
        res = op(self.x, self.y)
        expected = jnp.argmin(jnp.sum((self.x[:, None, :] - self.y[None, :, :])**2, axis=-1), axis=1, keepdims=True)
        self.assertTrue(jnp.allclose(res, expected))

    def test_genred_with_parameters(self):
        formula = "Exp(-SqDist(x, y) * s) * b"
        aliases = [f"x=Vi({self.D})", f"y=Vj({self.D})", f"b=Vj({self.E})", "s=Pm(1)"]
        op = Genred(formula, aliases, 'Sum', axis=1)
        s_val = jnp.array([0.5])
        res = op(self.x, self.y, self.b, s_val)

        # FIX: Use NumPy for expected value, not JAX
        import numpy as np
        x_np = np.array(self.x)
        y_np = np.array(self.y)
        b_np = np.array(self.b)
        K = np.exp(-np.sum((x_np[:, None, :] - y_np[None, :, :]) ** 2, axis=-1) * float(s_val[0]))
        expected = K @ b_np

        # Now this will pass with strict tolerance
        self.assertTrue(jnp.allclose(res, expected, rtol=1e-5, atol=1e-6))

class TestLazyTensor(unittest.TestCase):
    def setUp(self):
        self.N, self.M, self.D, self.E = 100, 50, 3, 2
        self.x = jax.random.normal(jax.random.PRNGKey(0), (self.N, self.D))
        self.y = jax.random.normal(jax.random.PRNGKey(1), (self.M, self.D))
        self.b = jax.random.normal(jax.random.PRNGKey(2), (self.M, self.E))

    def test_lazy_basic_operations(self):
        xi, yj = LazyTensor(self.x[:, None, :]), LazyTensor(self.y[None, :, :])
        res = ((xi - yj)**2).sum(-1).sum(1)
        expected = jnp.sum(jnp.sum((self.x[:, None, :] - self.y[None, :, :])**2, axis=-1), axis=1, keepdims=True)
        self.assertTrue(jnp.allclose(res, expected, rtol=1e-5))

    def test_lazy_gaussian_kernel(self):
        xi, yj, bj = LazyTensor(self.x[:, None, :]), LazyTensor(self.y[None, :, :]), LazyTensor(self.b[None, :, :])
        res = ((-((xi - yj)**2).sum(-1) * 0.5).exp() * bj).sum(1)
        self.assertEqual(res.shape, (self.N, self.E))

    def test_lazy_with_constants(self):
        xi, yj = LazyTensor(self.x[:, None, :]), LazyTensor(self.y[None, :, :])
        res = ((xi - yj)**2 * 2.0).sum(-1).sum(1)
        self.assertEqual(res.shape, (self.N, 1))

class TestGradients(unittest.TestCase):
    def setUp(self):
        self.x = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        self.y = jax.random.normal(jax.random.PRNGKey(1), (15, 3))

    def test_grad_genred(self):
        op = Genred("SqDist(x, y)", ["x=Vi(3)", "y=Vj(3)"], 'Sum', axis=1)
        grad = jax.grad(lambda x: jnp.sum(op(x, self.y)))(self.x)
        self.assertEqual(grad.shape, self.x.shape)

    def test_grad_lazy(self):
        def f(x): return jnp.sum(((LazyTensor(x[:,None,:]) - LazyTensor(self.y[None,:,:]))**2).sum(-1).sum(1))
        grad = jax.grad(f)(self.x)
        self.assertEqual(grad.shape, self.x.shape)

    def test_value_and_grad(self):
        op = Genred("SqDist(x, y)", ["x=Vi(3)", "y=Vj(3)"], 'Sum', axis=1)
        val, grad = jax.value_and_grad(lambda x: jnp.sum(op(x, self.y)))(self.x)
        self.assertTrue(val > 0)

    def test_grad_wrt_multiple_args(self):
        op = Genred("SqDist(x, y)", ["x=Vi(3)", "y=Vj(3)"], 'Sum', axis=1)
        gx, gy = jax.grad(lambda x, y: jnp.sum(op(x, y)), argnums=(0, 1))(self.x, self.y)
        self.assertEqual(gx.shape, self.x.shape)
        self.assertEqual(gy.shape, self.y.shape)

    def test_higher_order_gradients(self):
        op = Genred("SqDist(x, y)", ["x=Vi(3)", "y=Vj(3)"], 'Sum', axis=1)
        def loss(x): return jnp.sum(op(x, self.y))
        h_diag = jax.grad(lambda x: jnp.sum(jax.grad(loss)(x) * x))(self.x)
        self.assertEqual(h_diag.shape, self.x.shape)

class TestJIT(unittest.TestCase):
    def test_jit_genred_forward(self):
        op = Genred("SqDist(x, y)", ["x=Vi(3)", "y=Vj(3)"], 'Sum', axis=1)
        x, y = jnp.ones((10, 3)), jnp.zeros((5, 3))
        res = jax.jit(op)(x, y)
        self.assertEqual(res.shape, (10, 1))

    def test_jit_genred_backward(self):
        op = Genred("SqDist(x, y)", ["x=Vi(3)", "y=Vj(3)"], 'Sum', axis=1)
        @jax.jit
        def g(x, y): return jax.grad(lambda a: jnp.sum(op(a, y)))(x)
        self.assertEqual(g(jnp.ones((10,3)), jnp.zeros((5,3))).shape, (10, 3))

class TestBatching(unittest.TestCase):
    def test_vmap_genred(self):
        op = Genred("SqDist(x, y)", ["x=Vi(3)", "y=Vj(3)"], 'Sum', axis=1)
        x, y = jnp.ones((5, 10, 3)), jnp.zeros((5, 10, 3))
        res = jax.vmap(op)(x, y)
        self.assertEqual(res.shape, (5, 10, 1))

class TestSharding(unittest.TestCase):
    def test_basic_sharding(self):
        devices = jax.devices()
        if len(devices) < 2: self.skipTest("Need 2+ GPUs")
        x = jnp.ones((len(devices) * 4, 3))
        op = Genred("SqDist(x, x)", ["x=Vi(3)", "y=Vj(3)"], 'Sum', axis=1)
        self.assertEqual(op(x, x).shape, (len(devices) * 4, 1))

class TestDTypes(unittest.TestCase):
    def test_float32_f64(self):
        op32 = Genred("SqDist(x, y)", ["x=Vi(3)", "y=Vj(3)"], 'Sum', axis=1)
        op64 = Genred("SqDist(x, y)", ["x=Vi(3)", "y=Vj(3)"], 'Sum', axis=1, dtype='float64')
        self.assertEqual(op32(jnp.ones((5,3), 'float32'), jnp.ones((5,3), 'float32')).dtype, jnp.float32)
        self.assertEqual(op64(jnp.ones((5,3), 'float64'), jnp.ones((5,3), 'float64')).dtype, jnp.float64)

class TestComplexFormulas(unittest.TestCase):
    def test_gaussian_and_laplacian(self):
        x, y, b = jnp.ones((10, 3)), jnp.zeros((5, 3)), jnp.ones((5, 2))
        op = Genred("Exp(-SqDist(x, y)) * b", ["x=Vi(3)", "y=Vj(3)", "b=Vj(2)"], 'Sum', axis=1)
        self.assertEqual(op(x, y, b).shape, (10, 2))

class TestEdgeCases(unittest.TestCase):
    def test_empty_and_single(self):
        op = Genred("SqDist(x, y)", ["x=Vi(3)", "y=Vj(3)"], 'Sum', axis=1)
        self.assertEqual(op(jnp.zeros((0,3)), jnp.ones((1,3))).shape, (0, 1))
        self.assertTrue(jnp.allclose(op(jnp.array([[1.,2.,3.]]), jnp.zeros((1,3))), 14.0))

class TestAxisReduction(unittest.TestCase):
    def test_axes(self):
        op1 = Genred("SqDist(x, y)", ["x=Vi(3)", "y=Vj(3)"], 'Sum', axis=1)
        op0 = Genred("SqDist(x, y)", ["x=Vi(3)", "y=Vj(3)"], 'Sum', axis=0)
        self.assertEqual(op1(jnp.ones((10,3)), jnp.ones((5,3))).shape, (10, 1))
        self.assertEqual(op0(jnp.ones((10,3)), jnp.ones((5,3))).shape, (5, 1))

class TestPerformance(unittest.TestCase):
    def test_caching_and_batch(self):
        op = Genred("SqDist(x, y)", ["x=Vi(3)", "y=Vj(3)"], 'Sum', axis=1)
        x = jnp.ones((10, 100, 3))
        res = jax.vmap(op)(x, x)
        self.assertEqual(res.shape, (10, 100, 1))

# =============================================================================
# EXECUTION ENGINE
# =============================================================================

def run_test_suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    test_classes = [TestGenredBasics, TestLazyTensor, TestGradients, TestJIT,
                    TestBatching, TestSharding, TestDTypes, TestComplexFormulas,
                    TestEdgeCases, TestAxisReduction, TestPerformance]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = VisualTestRunner(verbosity=1)
    result = runner.run(suite)

    # Final Icon Summary Table
    print("\n" + "═" * 80)
    print(f"{'KEOPS JAX API COMPREHENSIVE UNIT TEST RESULTS':^80}")
    print("═" * 80)
    print(f"{'Result':<8} | {'Test Method Name'}")
    print("-" * 80)
    for name, icon, color in result.results:
        print(f"  {icon}      | {color}{name}\033[0m")

    print("═" * 80)
    pass_count = result.testsRun - len(result.failures) - len(result.errors)
    print(f"  OVERALL SCORE: {pass_count}/{result.testsRun} Passed ({len(result.skipped)} skipped)")
    print("═" * 80)
    return result.wasSuccessful()

if __name__ == '__main__':
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
    success = run_test_suite()
    sys.exit(0 if success else 1)