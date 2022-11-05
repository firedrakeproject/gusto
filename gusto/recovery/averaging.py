"""
This provides an operator for restoring continuity for discontinuous fields.

The :class:`Averager` provided in this module is an operator that transforms
fields from (partially-)discontinuous function spaces to their (partially-)
continuous counterparts. It does this by simply averaging the values at the DoFs
from the discontiuous space that correspond to those from the continuous space.
"""

from firedrake import Function
from firedrake.utils import cached_property
from gusto.recovery import recovery_kernels as kernels
import ufl

__all__ = ["Averager"]


class Averager(object):
    """
    Computes a continuous field from a broken space through averaging.

    This object restores the continuity from a field in a discontinuous or
    broken function space. The target function space must have the same DoFs per
    cell as the source function space. Then the value of the continuous field
    at a particular DoF is the average of the corresponding DoFs from the
    discontinuous space.
    """

    def __init__(self, v, v_out):
        """
        Args:
            v (:class:`Function`): the (discontinuous) field to average. Can
                also be a :class:`ufl.Expr`.
            v_out (:class:`Function`): the (continuous) field to compute.

        Raises:
            RuntimeError: the geometric shape of the two fields must be equal.
            RuntimeError: the number of DoFs per cell must be equal.
        """

        if not isinstance(v, (ufl.core.expr.Expr, Function)):
            raise ValueError("Can only recover UFL expression or Functions not '%s'" % type(v))

        # Check shape values
        if v.ufl_shape != v_out.ufl_shape:
            raise RuntimeError('Shape mismatch between source %s and target function spaces %s in project' % (v.ufl_shape, v_out.ufl_shape))

        self._same_fspace = (isinstance(v, Function) and v.function_space() == v_out.function_space())
        self.v = v
        self.v_out = v_out
        self.V = v_out.function_space()

        # Check the number of local dofs
        if self.v_out.function_space().finat_element.space_dimension() != self.v.function_space().finat_element.space_dimension():
            raise RuntimeError("Number of local dofs for each field must be equal.")

        self.average_kernel = kernels.AverageKernel(self.V)

    @cached_property
    def _weighting(self):
        """Generate the weights to be used in the averaging."""
        w = Function(self.V)

        weight_kernel = kernels.AverageWeightings(self.V)
        weight_kernel.apply(w)

        return w

    def project(self):
        """Apply the recovery."""
        # Ensure that the function being populated is zeroed out
        self.v_out.dat.zero()
        self.average_kernel.apply(self.v_out, self._weighting, self.v)
        return self.v_out
