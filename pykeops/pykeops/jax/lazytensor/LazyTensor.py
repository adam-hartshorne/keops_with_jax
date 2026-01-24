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

        # Initialize _var_ids
        self._var_ids = ()

        # Handle scalars FIRST
        if x is not None and not isinstance(x, tuple):
            if isinstance(x, (int, float)):
                x = jnp.array(x)
            elif hasattr(x, 'dtype') and not hasattr(x, 'shape'):
                x = jnp.array(x)
            if hasattr(x, 'shape') and len(x.shape) == 0:
                x = jnp.reshape(x, (1,))

        # For JAX arrays (but NOT LazyTensors), we handle them ourselves to ensure consistent ID scheme
        # The base class uses id(x) which doesn't work well with JAX tracers
        is_lazy_tensor = hasattr(x, '__GenericLazyTensor__')
        if x is not None and not isinstance(x, tuple) and not is_lazy_tensor and safe_tools.is_tensor(x):
            self._init_from_jax_array(x, axis)
        elif x is not None:
            # Non-tensor types (tuple for symbolic, int, list) - use base class
            try:
                super().__init__(x=x, axis=axis)
                # Base class may have created variables with id()-based IDs
                # We need to track these
                if hasattr(self, 'variables') and self.variables:
                    self._var_ids = tuple(id(v) for v in self.variables)
            except TypeError:
                raise
        else:
            # x is None - initialize empty
            super().__init__(x=None, axis=axis)

    def _init_from_jax_array(self, x, axis):
        """Initialize LazyTensor from a JAX array with consistent ID scheme."""
        # Set the duck typing attribute (required for base class compatibility)
        self.__GenericLazyTensor__ = True

        # Initialize attributes that base class would set
        self.batchdims = ()
        self.ni = None
        self.nj = None
        self.symbolic_variables = ()
        self.ranges = None
        self.backend = None
        self.formula2 = None
        self.is_complex = False

        # Handle 3D+ tensors
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

        # Set basic attributes
        self.variables = (x,)
        self.ndim = x.shape[-1]
        self.axis = axis if axis is not None else (2 if len(x.shape) == 1 else None)

        # Use monotonic counter for unique ID
        unique_id = _get_unique_var_id()
        self.formula = f"Var({unique_id},{self.ndim},{self.axis})"

        # Store the unique_id for this variable
        self._var_ids = (unique_id,)

        # Set ni/nj
        if len(x.shape) >= 2:
            if self.axis == 0:
                self.ni = x.shape[-2]
            elif self.axis == 1:
                self.nj = x.shape[-2]

        self._dtype = safe_tools.dtypename(safe_tools.dtype(x))

    def get_tools(self):
        pass

    def lt_constructor(self, x=None, axis=None, is_complex=False):
        return LazyTensor(x=x, axis=axis, is_complex=is_complex)

    def init(self, is_complex=False):
        """Override to propagate _var_ids."""
        res = super().init(is_complex=is_complex)
        # Propagate _var_ids if we have it
        if hasattr(self, '_var_ids'):
            res._var_ids = self._var_ids
        return res

    def join(self, other, is_complex=False):
        """Override to merge _var_ids from both operands, deduplicating by ID."""
        res = super().join(other, is_complex=is_complex)

        # Get _var_ids from both operands
        self_ids = getattr(self, '_var_ids', ())
        other_ids = getattr(other, '_var_ids', ())

        # The base class concatenates variables: res.variables = self.variables + other.variables
        # We need to deduplicate by _var_id, keeping only the first occurrence

        # Build deduplicated variables and _var_ids
        seen_ids = set()
        new_variables = []
        new_var_ids = []

        # Process self's variables first
        for i, var_id in enumerate(self_ids):
            if var_id not in seen_ids:
                seen_ids.add(var_id)
                new_variables.append(self.variables[i])
                new_var_ids.append(var_id)

        # Then other's variables (skip duplicates)
        for i, var_id in enumerate(other_ids):
            if var_id not in seen_ids:
                seen_ids.add(var_id)
                new_variables.append(other.variables[i])
                new_var_ids.append(var_id)

        # Update result with deduplicated variables
        res.variables = tuple(new_variables)
        res._var_ids = tuple(new_var_ids)

        return res

    def fixvariables(self):
        """
        Convert id-based variable names to index-based names.

        When we manually initialize LazyTensors (in except block), we use
        Var(unique_id, dim, cat) for uniqueness. But Genred needs Var(0, dim, cat),
        Var(1, dim, cat), etc. This method does the conversion.

        Uses _var_ids to correctly map each variable to its unique_id in the formula.
        """
        import re

        combined = (self.formula or "") + (self.formula2 or "")
        var_pattern = r'Var\((\d+),\s*(\d+),\s*(\d+|None)\)'

        # Get the _var_ids mapping (unique_id -> variable index in tuple)
        var_ids = getattr(self, '_var_ids', ())

        # Build mapping: unique_id -> index in self.variables tuple
        # Use FIRST occurrence of each unique_id
        id_to_tuple_idx = {}
        for tuple_idx, unique_id in enumerate(var_ids):
            if unique_id not in id_to_tuple_idx:  # Only keep first occurrence
                id_to_tuple_idx[unique_id] = tuple_idx

        # Replace Var(unique_id, dim, cat) with Var(tuple_idx, dim, cat)
        def replace_var(match):
            var_id = int(match.group(1))
            dim = match.group(2)
            cat = match.group(3)

            # If small number (<1000000), assume it's already an index
            if var_id < 1000000:
                return match.group(0)

            # Look up the tuple index for this unique_id
            tuple_idx = id_to_tuple_idx.get(var_id)
            if tuple_idx is None:
                # Fallback: not found in _var_ids, use old approach (for base class LazyTensors)
                # This shouldn't happen for JAX LazyTensors we create
                return match.group(0)

            # Convert None to 2 (Pm)
            if cat == "None":
                cat = "2"

            return f"Var({tuple_idx},{dim},{cat})"

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