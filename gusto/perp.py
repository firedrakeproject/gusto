"""
Defines a new UFL perp operator. This is necessary to give a symbolic
represenation of the perp operation.
"""

from ufl.core.expr import ufl_err_str
from ufl.core.ufl_type import ufl_type
from ufl.constantvalue import Zero
from ufl.tensoralgebra import CompoundTensorOperator
from ufl import as_ufl

__all__ = ["perp"]


def perp(v):
    "UFL operator: Take the perp of *v*, i.e. :math:`(-v_1, +v_0)`."
    v = as_ufl(v)
    if v.ufl_shape != (2,):
        raise ValueError("Expecting a 2D vector expression.")
    return Perp(v)


@ufl_type(is_index_free=True, num_ops=1)
class Perp(CompoundTensorOperator):
    __slots__ = ()

    def __new__(cls, A):
        sh = A.ufl_shape

        # Checks
        if not len(sh) == 1:
            raise ValueError(f"Perp requires arguments of rank 1, got {ufl_err_str(A)}")
        if not sh[0] == 2:
            raise ValueError(f"Perp can only work on 2D vectors, got {ufl_err_str(A)}")

        # Simplification
        if isinstance(A, Zero):
            return Zero((), A.ufl_free_indices, A.ufl_index_dimensions)

        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        CompoundTensorOperator.__init__(self, (A,))

    ufl_shape = (2,)

    def __str__(self):
        return "perp(%s)" % self.ufl_operands[0]
