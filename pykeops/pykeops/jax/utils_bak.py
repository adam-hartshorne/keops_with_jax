"""JAX utility functions for KeOps"""
import jax.numpy as jnp
import jax
import numpy as np

class JaxTools:
    arraytype = type(jnp.array([0]))
    arrayname = "JAX array"
    float_types = [float, np.float32, np.float64]

    @staticmethod
    def array(data, dtype=None, device=None):
        if dtype is None: return jnp.array(data)
        return jnp.array(data, dtype=dtype)

    @staticmethod
    def empty(shape, dtype, device):
        return jnp.empty(shape, dtype=dtype)

    @staticmethod
    def get_pointer(arr):
        if hasattr(arr, '__cuda_array_interface__'): return arr.__cuda_array_interface__['data'][0]
        elif hasattr(arr, '__array_interface__'): return arr.__array_interface__['data'][0]
        return id(arr)

    @staticmethod
    def dtype(arr): return arr.dtype

    @staticmethod
    def dtypename(dtype): return str(dtype)

    @staticmethod
    def device(arr): return None

    @staticmethod
    def view(arr, shape): return jnp.reshape(arr, shape)

    @staticmethod
    def is_tensor(x):
        return hasattr(x, 'shape') and hasattr(x, 'dtype') and hasattr(x, 'ndim')

    @staticmethod
    def detect_complex(x):
        if x is None: return False
        if isinstance(x, complex): return True
        if hasattr(x, 'dtype'): return jnp.issubdtype(x.dtype, jnp.complexfloating)
        return False

    @staticmethod
    def view_as_real(x):
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            real = jnp.real(x)
            imag = jnp.imag(x)
            return jnp.stack([real, imag], axis=-1).reshape(x.shape[:-1] + (-1,))
        return x

    @staticmethod
    def view_as_complex(x):
        shape = x.shape
        new_shape = shape[:-1] + (shape[-1] // 2, 2)
        x_reshaped = x.reshape(new_shape)
        return x_reshaped[..., 0] + 1j * x_reshaped[..., 1]

    @staticmethod
    def eq(x, y): return jnp.equal(x, y)

    @staticmethod
    def swap_axes(ranges):
        if ranges is None: return None
        if len(ranges) == 6:
            return (ranges[3], ranges[4], ranges[5], ranges[0], ranges[1], ranges[2])
        return ranges

jaxtools = JaxTools()