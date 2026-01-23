#!/usr/bin/env python3
"""
Verify the C++ extension is properly loaded and can be called.
"""
import os

os.environ['PYKEOPS_JAX_MODE'] = '1'

print("=" * 80)
print("KeOps JAX C++ Extension Test")
print("=" * 80)

# Import and check version
try:
    import keops_jax_ext

    print("\n✓ keops_jax_ext module imported successfully")

    # Check available functions
    print("\nAvailable functions:")
    for attr in dir(keops_jax_ext):
        if not attr.startswith('_'):
            print(f"  - {attr}")

    # Test registry functions
    print(f"\nRegistry size: {keops_jax_ext.get_registry_size()}")

except ImportError as e:
    print(f"\n✗ Failed to import keops_jax_ext: {e}")
    print("\nThis means the C++ extension isn't compiled or in the wrong location.")
    print("You need to rebuild it:")
    print("  cd /path/to/pykeops")
    print("  python setup.py build_ext --inplace")
    exit(1)

print("\n" + "=" * 80)
print("Now testing with actual KeOps call...")
print("=" * 80)

import numpy as np
import jax.numpy as jnp
from pykeops.jax import Genred

# Small test
rng = np.random.RandomState(42)
B, N, M, D = 2, 10, 10, 3

x_np = rng.randn(B, N, D).astype(np.float32)
y_np = rng.randn(B, M, D).astype(np.float32)

x = jnp.array(x_np)
y = jnp.array(y_np)

formula = "Exp(-g * SqDist(x, y))"
aliases = [f"x = Vi({D})", f"y = Vj({D})", "g = Pm(1)"]
genred = Genred(formula, aliases, reduction_op='Sum', axis=1)

g = jnp.array([1.0])

print(f"\nCalling Genred with N={N}, M={M}...")
print("If C++ debug is working, you should see '[KEOPS DEBUG]' messages below:")
print("-" * 80)

result = genred(x, y, g)

print("-" * 80)
print(f"\nResult shape: {result.shape}")
print(f"Result computed successfully!")

print("\n" + "=" * 80)
print("DIAGNOSTIC:")
print("=" * 80)
print("If you see NO '[KEOPS DEBUG]' messages above, then:")
print("1. The C++ file wasn't properly replaced, OR")
print("2. The extension wasn't rebuilt, OR")
print("3. Python is loading an old cached version")
print("\nTry:")
print("  1. Verify keops_jax.cpp was replaced with the debug version")
print("  2. Run: python setup.py build_ext --inplace --force")
print("  3. Restart Python (to clear any cached imports)")
print("  4. Check the .so file timestamp to verify it was rebuilt")