"""
KeOps JAX Genred class with lazy FFI registration
"""

import jax.numpy as jnp
from .generic_ops import make_keops_jax_op


class Genred:
    """
    JAX interface for KeOps generic reductions.

    Example:
        formula = "SqDist(x, y)"
        aliases = ["x = Vi(3)", "y = Vj(3)"]
        op = Genred(formula, aliases, reduction_op='Sum', axis=1)

        x = jnp.array(...)  # shape (N, 3)
        y = jnp.array(...)  # shape (M, 3)
        result = op(x, y)   # shape (N, 1)
    """

    def __init__(
        self,
        formula,
        aliases,
        reduction_op='Sum',
        axis=0,
        dtype='float32',
        opt_arg=None,
        formula2=None,
        cuda_type=None,
        dtype_acc="auto",
        use_double_acc=False,
        sum_scheme="auto",
        enable_chunks=True,
        rec_multVar_highdim=False,
        use_fast_math=True,
    ):
        """
        Initialize KeOps reduction operator.

        Args:
            formula: KeOps formula string (e.g., "SqDist(x, y)")
            aliases: List of variable definitions (e.g., ["x = Vi(3)", "y = Vj(3)"])
            reduction_op: Reduction operation ('Sum', 'Max', 'Min', etc.)
            axis: Reduction axis (0 for reducing over i, 1 for reducing over j)
            dtype: Data type ('float32' or 'float64')
            opt_arg: Optional argument (for compatibility with NumPy Genred)
            formula2: Optional second formula (for compatibility)
            cuda_type: CUDA type (for compatibility)
            dtype_acc: Accumulator dtype (for compatibility)
            use_double_acc: Use double precision accumulator (for compatibility)
            sum_scheme: Summation scheme (for compatibility)
            enable_chunks: Enable chunked mode (for compatibility)
            rec_multVar_highdim: Recursive multVar highdim (for compatibility)
            use_fast_math: Use fast math (for compatibility)
        """
        self.formula = formula
        self.aliases = aliases
        self.reduction_op = reduction_op
        self.axis = axis
        self.dtype_str = dtype

        # Store optional args for compatibility (currently unused)
        self.opt_arg = opt_arg
        self.formula2 = formula2
        self.dtype_acc = dtype_acc
        self.sum_scheme = sum_scheme
        self.enable_chunks = enable_chunks

        # Compile the kernel and get the JAX operator
        # Note: This creates the myconv object but doesn't register FFI yet
        self._jax_op = make_keops_jax_op(
            formula=self.formula,
            aliases=tuple(self.aliases),
            reduction_op=self.reduction_op,
            axis=self.axis,
            dtype_str=self.dtype_str
        )

    def __call__(self, *args):
        """
        Execute the KeOps kernel.

        FFI registration happens here on first call, not at init time!
        This avoids JAX's test validation calls.
        """
        # Flatten list of arguments if passed as a list
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            return self._jax_op(*args[0])
        return self._jax_op(*args)