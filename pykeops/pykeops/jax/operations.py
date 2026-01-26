"""
KeOps JAX KernelSolve - Conjugate Gradient Solver for Kernel Linear Systems

Solves optimization problems of the form:
    a* = argmin_a  (1/2) <a, (αI + K_xx) a> - <a, b>

which is equivalent to solving the linear system:
    (αI + K_xx) a = b

where K_xx is a kernel matrix defined by a KeOps formula.
"""

import jax
import jax.numpy as jnp
from functools import partial

from .generic import Genred
from pykeops.common.parse_type import get_type, get_sizes, complete_aliases
from pykeops.common.utils import axis2cat


def conjugate_gradient_jax(linop, b, eps=1e-6, max_iter=1000):
    """
    Conjugate gradient solver for JAX.
    
    Solves Ma = b where linop computes M @ v.
    
    Args:
        linop: Function that computes matrix-vector product M @ v
        b: Right-hand side vector
        eps: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        Solution vector a
    """
    delta = b.size * eps**2
    
    # Initialize
    a = jnp.zeros_like(b)
    r = b.copy()
    nr2 = jnp.sum(r**2)
    
    # Early exit if b is nearly zero
    if nr2 < delta:
        return a
    
    p = r.copy()
    
    for k in range(max_iter):
        Mp = linop(p)
        pMp = jnp.sum(p * Mp)
        
        # Avoid division by zero
        alp = jnp.where(pMp > 1e-30, nr2 / pMp, 0.0)
        
        a = a + alp * p
        r = r - alp * Mp
        nr2_new = jnp.sum(r**2)
        
        if nr2_new < delta:
            break
            
        beta = nr2_new / nr2
        p = r + beta * p
        nr2 = nr2_new
    
    return a


def conjugate_gradient_jax_lax(linop, b, eps=1e-6, max_iter=1000):
    """
    JAX-native conjugate gradient using lax.while_loop for JIT compatibility.
    
    Solves Ma = b where linop computes M @ v.
    
    Args:
        linop: Function that computes matrix-vector product M @ v
        b: Right-hand side vector
        eps: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        Solution vector a
    """
    delta = b.size * eps**2
    
    def cond_fn(state):
        a, r, p, nr2, k = state
        return jnp.logical_and(nr2 >= delta, k < max_iter)
    
    def body_fn(state):
        a, r, p, nr2, k = state
        
        Mp = linop(p)
        pMp = jnp.sum(p * Mp)
        alp = jnp.where(pMp > 1e-30, nr2 / pMp, 0.0)
        
        a_new = a + alp * p
        r_new = r - alp * Mp
        nr2_new = jnp.sum(r_new**2)
        
        beta = jnp.where(nr2 > 1e-30, nr2_new / nr2, 0.0)
        p_new = r_new + beta * p
        
        return (a_new, r_new, p_new, nr2_new, k + 1)
    
    # Initialize
    a0 = jnp.zeros_like(b)
    r0 = b
    p0 = b
    nr2_0 = jnp.sum(b**2)
    
    # Early exit check
    init_state = (a0, r0, p0, nr2_0, 0)
    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    
    return final_state[0]


class KernelSolve:
    r"""
    Creates a conjugate gradient solver for kernel linear systems.
    
    Supporting the same generic syntax as :class:`Genred`, this class allows 
    you to solve optimization problems of the form:

    .. math::
        a^* = \operatorname*{argmin}_a \frac{1}{2} \langle a, (\alpha I + K_{xx}) a \rangle - \langle a, b \rangle

    which is equivalent to solving:

    .. math::
        (\alpha I + K_{xx}) a = b

    where :math:`K_{xx}` is a kernel matrix defined by the KeOps formula.

    Example:
        >>> # Define a Gaussian kernel solve
        >>> formula = "Exp(-SqDist(x,y)) * a"
        >>> aliases = ["x=Vi(3)", "y=Vj(3)", "a=Vj(3)"]
        >>> solver = KernelSolve(formula, aliases, "a", axis=1)
        >>> 
        >>> # Solve (αI + K)a = b
        >>> x = jnp.randn(1000, 3)
        >>> b = jnp.randn(1000, 3)
        >>> a_star = solver(x, x, b, alpha=0.1)
    
    Args:
        formula (string): The KeOps formula defining the kernel K(x,y) * a.
            Should be linear in the variable specified by `varinvalias`.
        aliases (list of strings): Variable definitions like ["x=Vi(3)", "y=Vj(3)", "a=Vj(3)"]
        varinvalias (string): The alias of the variable to solve for (must appear linearly in formula)
        axis (int): Reduction axis (0 or 1). Default 1.
        dtype (string): Data type, 'float32' or 'float64'. Default 'float32'.
        
    Keyword Args:
        dtype_acc: Accumulator dtype for improved accuracy
        sum_scheme: Summation scheme ('auto', 'block_sum', 'direct_sum', 'kahan_scheme')
        enable_chunks: Enable chunked computation mode
        use_fast_math: Use fast math optimizations
    """

    def __init__(
        self,
        formula,
        aliases,
        varinvalias,
        axis=1,
        dtype='float32',
        dtype_acc="auto",
        sum_scheme="auto",
        enable_chunks=True,
        use_fast_math=True,
    ):
        self.formula = formula
        self.aliases = complete_aliases(formula, list(aliases))
        self.axis = axis
        self.dtype = dtype
        self.dtype_acc = dtype_acc
        self.sum_scheme = sum_scheme
        self.enable_chunks = enable_chunks
        self.use_fast_math = use_fast_math
        
        # Find the position of the variable we're solving for
        if varinvalias.startswith("Var("):
            # Direct Var specification
            self.varinvpos = int(varinvalias[4:varinvalias.find(",")])
        else:
            # Find by alias name
            alias_names = []
            for s in self.aliases:
                name = s[:s.find("=")].strip()
                alias_names.append(name)
            self.varinvpos = alias_names.index(varinvalias)
        
        # Create the underlying Genred operator
        self._genred = Genred(
            formula=self.formula,
            aliases=self.aliases,
            reduction_op='Sum',
            axis=self.axis,
            dtype=self.dtype,
        )

    def __call__(self, *args, alpha=1e-10, eps=1e-6, max_iter=1000):
        """
        Solve the kernel linear system (αI + K)a = b.

        This method is JIT-compatible and can be used inside jax.jit, jax.grad,
        and jax.vmap transformations.

        Args:
            *args: Input arrays matching the aliases. The array at position
                   `varinvpos` is the right-hand side `b`.
            alpha (float): Ridge regularization parameter. Default 1e-10.
            eps (float): Convergence tolerance. Default 1e-6.
            max_iter (int): Maximum CG iterations. Default 1000.

        Returns:
            Solution array `a` with same shape as `b`.
        """
        # Get the right-hand side b (at varinvpos)
        b = args[self.varinvpos]

        # Define the linear operator: linop(v) = K @ v + alpha * v
        def linop(v):
            # Replace the varinv argument with v
            new_args = args[:self.varinvpos] + (v,) + args[self.varinvpos + 1:]
            Kv = self._genred(*new_args)
            return Kv + alpha * v

        # Solve using JAX-native conjugate gradient (JIT-compatible)
        solution = conjugate_gradient_jax_lax(linop, b, eps=eps, max_iter=max_iter)

        return solution
