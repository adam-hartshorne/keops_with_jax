"""
KeOps JAX LazyTensor implementation

FIXES APPLIED:
1. Use monotonic counter instead of id() for variable uniqueness
   (id() can be reused when objects are deallocated)
"""

import jax.numpy as jnp
import threading

from pykeops.common.lazy_tensor import GenericLazyTensor, ComplexGenericLazyTensor
from pykeops.jax.utils import jaxtools
from pykeops.jax.generic import Genred

# Thread-safe monotonic counter for unique variable IDs
_var_counter = 0
_var_counter_lock = threading.Lock()


def _get_unique_var_id():
    """Get a unique variable ID that won't be reused."""
    global _var_counter
    with _var_counter_lock:
        _var_counter += 1
        # Use a large offset to distinguish from actual indices
        return _var_counter + 1000000


class SafeTools:
    # CRITICAL: Use object to accept JAX tracers (needed for gradients)
    arraytype = object
    arrayname = "JAX array"
    float_types = jaxtools.float_types

    @staticmethod
    def is_tensor(x):
        return hasattr(x, 'shape') and hasattr(x, 'dtype')

    @staticmethod
    def dtype(x):
        return x.dtype if hasattr(x, 'dtype') else None

    @staticmethod
    def dtypename(dtype):
        s = str(dtype)
        if 'float32' in s: return 'float32'
        if 'float64' in s: return 'float64'
        return s

    @staticmethod
    def view(x, shape):
        if isinstance(shape, int): shape = (shape,)
        return jnp.reshape(x, shape)

    @staticmethod
    def contiguous(x):
        return x

    @staticmethod
    def get_pointer(arr):
        return arr.__cuda_array_interface__['data'][0] if hasattr(arr, '__cuda_array_interface__') else id(arr)

    def __getattr__(self, name):
        return getattr(jaxtools, name)


safe_tools = SafeTools()


def Var(x_or_ind, dim=None, cat=None):
    if dim is None:
        return LazyTensor(x_or_ind, axis=cat)
    else:
        return LazyTensor((x_or_ind, dim, cat))


def Vi(x_or_ind, dim=None):
    if dim is None and hasattr(x_or_ind, 'shape') and len(x_or_ind.shape) == 2:
        x_or_ind = x_or_ind[:, None, :]
        return LazyTensor(x_or_ind)
    return Var(x_or_ind, dim, 0)


def Vj(x_or_ind, dim=None):
    if dim is None and hasattr(x_or_ind, 'shape') and len(x_or_ind.shape) == 2:
        x_or_ind = x_or_ind[None, :, :]
        return LazyTensor(x_or_ind)
    return Var(x_or_ind, dim, 1)


def Pm(x_or_ind, dim=None):
    return Var(x_or_ind, dim, 2)


class LazyTensor(GenericLazyTensor):
    def __new__(self, x=None, axis=None, is_complex=False):
        if is_complex or safe_tools.detect_complex(x):
            return ComplexLazyTensor(x, axis)
        return object.__new__(self)

    def __init__(self, x=None, axis=None, is_complex=False):
        self.tools = safe_tools
        self.Genred = Genred
        self.KernelSolve = None

        # Handle scalars FIRST
        if x is not None and not isinstance(x, tuple):
            if isinstance(x, (int, float)):
                x = jnp.array(x)
            elif hasattr(x, 'dtype') and not hasattr(x, 'shape'):
                x = jnp.array(x)
            if hasattr(x, 'shape') and len(x.shape) == 0:
                x = jnp.reshape(x, (1,))

        # Try base class first
        try:
            super().__init__(x=x, axis=axis)
        except TypeError as e:
            # Base class rejected JAX array - handle manually
            if x is not None and safe_tools.is_tensor(x):
                self.batchdims = ()
                self.ni = None
                self.nj = None

                # Handle 3D tensors
                if len(x.shape) >= 3:
                    if len(x.shape) > 3:
                        self.batchdims = tuple(x.shape[:-3])

                    dim_i = x.shape[-3]
                    dim_j = x.shape[-2]

                    # Determine axis from shape if not provided
                    if axis is None:
                        if dim_i > 1 and dim_j == 1:
                            axis = 0  # Vi
                        elif dim_i == 1 and dim_j > 1:
                            axis = 1  # Vj
                        elif dim_i == 1 and dim_j == 1:
                            axis = 1  # Default to Vj for (1,1,D)

                    # Squeeze dimensions
                    if dim_i == 1 and dim_j == 1:
                        x = jnp.squeeze(x, axis=-3)
                        x = jnp.squeeze(x, axis=-2)
                    elif dim_i == 1:
                        x = jnp.squeeze(x, axis=-3)
                    elif dim_j == 1:
                        x = jnp.squeeze(x, axis=-2)

                # Set basic attributes - Use monotonic counter for uniqueness!
                self.variables = (x,)
                self.ndim = x.shape[-1]
                self.axis = axis if axis is not None else (2 if len(x.shape) == 1 else None)

                # FIX: Use monotonic counter instead of id(x) to avoid reuse issues
                unique_id = _get_unique_var_id()
                self.formula = f"Var({unique_id},{self.ndim},{self.axis})"

                # Set ni/nj
                if len(x.shape) >= 2:
                    if self.axis == 0:
                        self.ni = x.shape[-2]
                    elif self.axis == 1:
                        self.nj = x.shape[-2]

                self._dtype = safe_tools.dtypename(safe_tools.dtype(x))
            else:
                raise

    def get_tools(self):
        pass

    def lt_constructor(self, x=None, axis=None, is_complex=False):
        return LazyTensor(x=x, axis=axis, is_complex=is_complex)

    def fixvariables(self):
        """
        Convert id-based variable names to index-based names.

        When we manually initialize LazyTensors (in except block), we use
        Var(unique_id, dim, cat) for uniqueness. But Genred needs Var(0, dim, cat),
        Var(1, dim, cat), etc. This method does the conversion.
        """
        import re

        combined = (self.formula or "") + (self.formula2 or "")

        # Build mapping from unique_id to index based on variable order
        id_to_idx = {}
        for i, v in enumerate(self.variables):
            # Find the unique ID used for this variable in the formula
            # We search for Var(large_number, dim, cat) patterns
            pass  # Will be handled below

        # Find all Var IDs in formulas
        var_pattern = r'Var\((\d+),\s*(\d+),\s*(\d+|None)\)'

        # Collect all unique IDs in order of appearance
        seen_ids = []
        for match in re.finditer(var_pattern, combined):
            var_id = int(match.group(1))
            if var_id not in seen_ids:
                seen_ids.append(var_id)

        # Build mapping: large IDs (>1000000) are our unique IDs, map to indices
        id_to_idx = {}
        idx = 0
        for var_id in seen_ids:
            if var_id >= 1000000:  # Our unique IDs start at 1000001
                id_to_idx[var_id] = idx
                idx += 1

        # Replace Var(unique_id, dim, cat) with Var(idx, dim, cat)
        def replace_var(match):
            var_id = int(match.group(1))
            dim = match.group(2)
            cat = match.group(3)

            # If small number (<1000000), assume it's already an index
            if var_id < 1000000:
                return match.group(0)

            # Look up index
            idx = id_to_idx.get(var_id, 0)

            # Convert None to 2 (Pm)
            if cat == "None":
                cat = "2"

            return f"Var({idx},{dim},{cat})"

        # Fix both formulas
        if self.formula:
            self.formula = re.sub(var_pattern, replace_var, self.formula)
        if self.formula2:
            self.formula2 = re.sub(var_pattern, replace_var, self.formula2)

        return self

    def __call__(self, *args, **kwargs):
        """
        Call the compiled KeOps kernel.

        This method is called when a reduction operation is applied.
        It creates the Genred operation and executes it.
        """
        import re

        # Convert id-based formulas to index-based
        self.fixvariables()

        # Build aliases from variables
        combined = (self.formula or "") + (self.formula2 or "")
        aliases = []

        for i, v in enumerate(self.variables):
            # Try to find variable definition in formula
            m = re.search(r"Var\({},\s*(\d+),\s*(\d+)\)".format(i), combined)
            if m:
                aliases.append(f"Var({i},{m.group(1)},{m.group(2)})")
            else:
                # Fallback: infer from shape
                dim = v.shape[-1]
                if len(v.shape) == 1:
                    c = 2  # Parameter
                elif len(v.shape) == 2:
                    c = 2  # Ambiguous, default to parameter
                elif len(v.shape) >= 3:
                    if v.shape[-2] == 1:
                        c = 0  # Vi
                    elif v.shape[-3] == 1:
                        c = 1  # Vj
                    else:
                        c = 2  # Parameter
                else:
                    c = 2
                aliases.append(f"Var({i},{dim},{c})")

        # Get dtype
        dtype_str = getattr(self, '_dtype', 'float32')

        # Create Genred operation
        op = self.Genred(
            self.formula,
            aliases,
            self.reduction_op,
            self.axis,
            dtype=dtype_str,
            opt_arg=self.opt_arg,
            formula2=self.formula2
        )

        # Pass variables directly (not as a list)
        return op(*self.variables)


class ComplexLazyTensor(ComplexGenericLazyTensor):
    def __init__(self, x=None, axis=None):
        self.tools = safe_tools
        self.Genred = Genred
        self.KernelSolve = None

        if x is not None:
            if isinstance(x, (int, float)):
                x = jnp.array(x)
            elif hasattr(x, 'dtype') and not hasattr(x, 'shape'):
                x = jnp.array(x)
            if hasattr(x, 'shape') and len(x.shape) == 0:
                x = jnp.reshape(x, (1,))

        super().__init__(x=x, axis=axis)

    def get_tools(self):
        pass

    def lt_constructor(self, x=None, axis=None, is_complex=True):
        return LazyTensor(x=x, axis=axis, is_complex=is_complex)