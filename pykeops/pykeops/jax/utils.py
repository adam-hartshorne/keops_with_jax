"""
JAX-specific utilities for KeOps
"""
import jax
import jax.numpy as jnp
from jax import Array as JaxArray


class jaxtools:
    """Utility functions for JAX arrays in KeOps"""

    # Class attributes
    norm = jnp.linalg.norm
    arraysum = jnp.sum
    exp = jnp.exp
    log = jnp.log
    arraytype = JaxArray
    float_types = [float, jnp.float16, jnp.float32, jnp.float64]

    # These will be set by JAX's Genred/KernelSolve when available
    Genred = None
    KernelSolve = None

    @staticmethod
    def swap_axes(x, ax1, ax2):
        return jnp.swapaxes(x, ax1, ax2)

    @staticmethod
    def is_tensor(x):
        """Check if x is a JAX array"""
        return isinstance(x, JaxArray)

    @staticmethod
    def copy(x):
        return jnp.copy(x)

    @staticmethod
    def eq(x, y):
        return jnp.equal(x, y)

    @staticmethod
    def transpose(x):
        return x.T

    @staticmethod
    def permute(x, *args):
        return jnp.transpose(x, args)

    @staticmethod
    def contiguous(x):
        """JAX arrays are already contiguous"""
        return x

    @staticmethod
    def numpy(x):
        """Convert JAX array to NumPy"""
        return jnp.asarray(x)

    @staticmethod
    def tile(*args):
        return jnp.tile(*args)

    @staticmethod
    def solve(A, b):
        return jnp.linalg.solve(A, b)

    @staticmethod
    def size(x):
        return x.size

    @staticmethod
    def view(x, *args):
        return x.reshape(*args)

    @staticmethod
    def long(x):
        # Use int32 by default since JAX doesn't enable X64 by default
        # This avoids the "int64 will be truncated to int32" warning
        if jax.config.x64_enabled:
            return x.astype(jnp.int64)
        else:
            return x.astype(jnp.int32)

    @staticmethod
    def dtype(x):
        return x.dtype

    @staticmethod
    def detect_complex(x):
        if type(x) == list:
            return any(type(v) == complex for v in x)
        elif isinstance(x, JaxArray):
            return jnp.iscomplexobj(x)
        else:
            return type(x) == complex

    @staticmethod
    def view_as_complex(x):
        if x.dtype == jnp.float32:
            return x.view(jnp.complex64)
        elif x.dtype == jnp.float64:
            return x.view(jnp.complex128)
        return x

    @staticmethod
    def view_as_real(x):
        return x.view(jnp.float32 if x.dtype == jnp.complex64 else jnp.float64)

    @staticmethod
    def dtypename(dtype):
        return str(dtype)

    @staticmethod
    def rand(shape, dtype=jnp.float32):
        key = jax.random.PRNGKey(0)
        return jax.random.uniform(key, shape, dtype=dtype)

    @staticmethod
    def randn(shape, dtype=jnp.float32):
        key = jax.random.PRNGKey(0)
        return jax.random.normal(key, shape, dtype=dtype)

    @staticmethod
    def zeros(shape, dtype, device=None):
        return jnp.zeros(shape, dtype=dtype)

    @staticmethod
    def empty(shape, dtype, device=None):
        return jnp.empty(shape, dtype=dtype)

    @staticmethod
    def eye(n, dtype):
        return jnp.eye(n).astype(dtype)

    @staticmethod
    def array(x, dtype, device=None):
        return jnp.array(x).astype(dtype)

    @staticmethod
    def get_pointer(x):
        """Get GPU device pointer from JAX array"""
        if not isinstance(x, JaxArray):
            raise TypeError(f"Expected JAX array, got {type(x)}")
        return x.unsafe_buffer_pointer()

    @staticmethod
    def device(x):
        """Get device string from JAX array or tracer"""
        from jax import core

        # Handle JAX tracers during jit/grad/vmap
        if isinstance(x, core.Tracer):
            return 'cuda:0'

        # Handle concrete JAX arrays - .devices() returns a set
        if hasattr(x, 'devices'):
            devs = x.devices()
            if devs:
                return str(list(devs)[0])
            return 'cpu'

        # Fallback
        if hasattr(x, 'device'):
            return str(x.device)
        return 'cpu'

    @staticmethod
    def device_type_index(x):
        """Get device type and index"""
        from jax import core

        # Handle tracers
        if isinstance(x, core.Tracer):
            return "cuda", 0

        device_str = jaxtools.device(x)
        if 'gpu' in device_str.lower() or 'cuda' in device_str.lower():
            # Extract device index if present
            try:
                idx = int(device_str.split(':')[-1])
                return "cuda", idx
            except:
                return "cuda", 0
        return "cpu", None

    @staticmethod
    def device_dict(x):
        """Get device as dictionary"""
        device_type, device_id = jaxtools.device_type_index(x)
        return {"type": device_type, "index": device_id}