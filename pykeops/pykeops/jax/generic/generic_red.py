"""
KeOps JAX Genred class with lazy FFI registration
"""

import jax.numpy as jnp
from .generic_ops import make_keops_jax_op
from pykeops.common.operations import preprocess, postprocess


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
            reduction_op: Reduction operation ('Sum', 'Max', 'Min', 'LogSumExp', 'KMin', 'ArgKMin', etc.)
            axis: Reduction axis (0 for reducing over i, 1 for reducing over j)
            dtype: Data type ('float32' or 'float64')
            opt_arg: Optional argument (e.g., K for KMin reduction)
            formula2: Optional second formula (for weighted reductions)
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
        self.reduction_op = reduction_op  # Keep original for postprocess
        self.axis = axis
        self.dtype_str = dtype
        self.opt_arg = opt_arg
        self.formula2 = formula2
        self.dtype_acc = dtype_acc
        self.sum_scheme = sum_scheme
        self.enable_chunks = enable_chunks

        # Preprocess reduction_op and formula2 for special reductions
        # (e.g., LogSumExp -> Max_SumShiftExp)
        reduction_op_internal, formula2_internal = preprocess(reduction_op, formula2)
        
        self.reduction_op_internal = reduction_op_internal
        self.formula2_internal = formula2_internal

        # Compile the kernel and get the JAX operator
        # Note: This creates the myconv object but doesn't register FFI yet
        self._jax_op = make_keops_jax_op(
            formula=self.formula,
            aliases=tuple(self.aliases),
            reduction_op=self.reduction_op_internal,
            axis=self.axis,
            dtype_str=self.dtype_str,
            opt_arg=self.opt_arg,
            formula2=self.formula2_internal
        )

    def __call__(self, *args):
        """
        Execute the KeOps kernel.

        FFI registration happens here on first call, not at init time!
        This avoids JAX's test validation calls.
        """
        # Flatten list of arguments if passed as a list
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        
        # Call the kernel
        out = self._jax_op(*args)
        
        # Determine output dimension for postprocess
        nout = out.shape[-1] if len(out.shape) > 0 else 1
        
        # Postprocess output for special reductions (e.g., LogSumExp finalization)
        out = postprocess(out, "jax", self.reduction_op, nout, self.opt_arg, self.dtype_str)
        
        return out