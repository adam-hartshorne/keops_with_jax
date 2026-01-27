"""
KeOps JAX Helper Functions
==========================

Convenience wrappers around Genred for common reduction operations.
These provide a simpler API matching the PyTorch KeOps helpers.
"""

from pykeops.common.parse_type import get_type
from pykeops.common.utils import cat2axis
from .generic_red import Genred


def generic_sum(formula, output, *aliases, **kwargs):
    """Alias for Genred with a "Sum" reduction.

    Args:
        formula (str): Symbolic KeOps expression.
        output (str): Output specification of the form "name = Vi(dim)" or "name = Vj(dim)".
            - Vi: indexation by i along axis 0; reduction performed along axis 1.
            - Vj: indexation by j along axis 1; reduction performed along axis 0.
        *aliases (str): Variable specifications like "x = Vi(3)", "y = Vj(3)".
        **kwargs: Additional arguments passed to Genred (e.g., dtype).

    Returns:
        A callable Genred operation.

    Example:
        >>> my_conv = generic_sum(
        ...     'Exp(-SqNorm2(x - y))',  # Formula
        ...     'a = Vi(1)',              # Output: 1 scalar per line
        ...     'x = Vi(3)',              # 1st input: dim-3 vector per line
        ...     'y = Vj(3)')              # 2nd input: dim-3 vector per line
        >>> x = jnp.randn(1000, 3)
        >>> y = jnp.randn(2000, 3)
        >>> a = my_conv(x, y)  # a_i = sum_j exp(-|x_i-y_j|^2)
    """
    _, cat, _, _ = get_type(output)
    axis = cat2axis(cat)
    return Genred(formula, aliases, reduction_op="Sum", axis=axis, **kwargs)


def generic_logsumexp(formula, output, *aliases, **kwargs):
    """Alias for Genred with a "LogSumExp" reduction.

    Args:
        formula (str): Scalar-valued symbolic KeOps expression.
        output (str): Output specification of the form "name = Vi(1)" or "name = Vj(1)".
        *aliases (str): Variable specifications.
        **kwargs: Additional arguments passed to Genred.

    Returns:
        A callable Genred operation.

    Example:
        >>> log_likelihood = generic_logsumexp(
        ...     '(-(g * SqNorm2(x - y))) + b',
        ...     'a = Vi(1)',
        ...     'x = Vi(3)',
        ...     'y = Vj(3)',
        ...     'g = Pm(1)',
        ...     'b = Vj(1)')
    """
    _, cat, _, _ = get_type(output)
    axis = cat2axis(cat)
    return Genred(formula, aliases, reduction_op="LogSumExp", axis=axis, **kwargs)


def generic_argmin(formula, output, *aliases, **kwargs):
    """Alias for Genred with an "ArgMin" reduction.

    Args:
        formula (str): Scalar-valued symbolic KeOps expression.
        output (str): Output specification of the form "name = Vi(1)" or "name = Vj(1)".
        *aliases (str): Variable specifications.
        **kwargs: Additional arguments passed to Genred.

    Returns:
        A callable Genred operation that returns indices of minimum values.

    Example:
        >>> nearest_neighbor = generic_argmin(
        ...     'SqDist(x, y)',
        ...     'a = Vi(1)',
        ...     'x = Vi(100)',
        ...     'y = Vj(100)')
        >>> x = jnp.randn(5, 100)
        >>> y = jnp.randn(20000, 100)
        >>> a = nearest_neighbor(x, y)  # Indices of nearest neighbors
    """
    _, cat, _, _ = get_type(output)
    axis = cat2axis(cat)
    return Genred(formula, aliases, reduction_op="ArgMin", axis=axis, **kwargs)


def generic_argkmin(formula, output, *aliases, **kwargs):
    """Alias for Genred with an "ArgKMin" reduction (K nearest neighbors).

    Args:
        formula (str): Scalar-valued symbolic KeOps expression.
        output (str): Output specification of the form "name = Vi(K)" or "name = Vj(K)",
            where K is the number of nearest neighbors to find.
        *aliases (str): Variable specifications.
        **kwargs: Additional arguments passed to Genred.

    Returns:
        A callable Genred operation that returns indices of K minimum values.

    Example:
        >>> knn = generic_argkmin(
        ...     'SqDist(x, y)',
        ...     'a = Vi(5)',   # Find 5 nearest neighbors
        ...     'x = Vi(100)',
        ...     'y = Vj(100)')
        >>> x = jnp.randn(5, 100)
        >>> y = jnp.randn(20000, 100)
        >>> a = knn(x, y)  # Shape: (5, 5) - indices of 5 nearest neighbors for each point
    """
    _, cat, k, _ = get_type(output)
    axis = cat2axis(cat)
    return Genred(formula, aliases, reduction_op="ArgKMin", axis=axis, opt_arg=k, **kwargs)


def generic_min(formula, output, *aliases, **kwargs):
    """Alias for Genred with a "Min" reduction.

    Args:
        formula (str): Symbolic KeOps expression.
        output (str): Output specification of the form "name = Vi(dim)" or "name = Vj(dim)".
        *aliases (str): Variable specifications.
        **kwargs: Additional arguments passed to Genred.

    Returns:
        A callable Genred operation that returns minimum values.
    """
    _, cat, _, _ = get_type(output)
    axis = cat2axis(cat)
    return Genred(formula, aliases, reduction_op="Min", axis=axis, **kwargs)


def generic_max(formula, output, *aliases, **kwargs):
    """Alias for Genred with a "Max" reduction.

    Args:
        formula (str): Symbolic KeOps expression.
        output (str): Output specification of the form "name = Vi(dim)" or "name = Vj(dim)".
        *aliases (str): Variable specifications.
        **kwargs: Additional arguments passed to Genred.

    Returns:
        A callable Genred operation that returns maximum values.
    """
    _, cat, _, _ = get_type(output)
    axis = cat2axis(cat)
    return Genred(formula, aliases, reduction_op="Max", axis=axis, **kwargs)
