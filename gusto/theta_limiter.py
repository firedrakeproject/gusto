from __future__ import absolute_import, print_function, division
from firedrake import dx, assemble, LinearSolver
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.parloops import par_loop, READ, RW
from firedrake.ufl_expr import TrialFunction, TestFunction
from firedrake.slope_limiter.limiter import Limiter, VertexBasedLimiter
__all__ = ("ThetaLimiter",)


class ThetaLimiter(Limiter):
    """
    A vertex based limiter for P1DG fields.

    This limiter implements the vertex-based limiting scheme described in
    Dmitri Kuzmin, "A vertex-based hierarchical slope limiter for p-adaptive
    discontinuous Galerkin methods". J. Comp. Appl. Maths (2010)
    http://dx.doi.org/10.1016/j.cam.2009.05.028
    """

    def __init__(self, space):
        """
        Initialise limiter

        :param space : FunctionSpace instance
        """

        self.Vt = space
        self.Q1DG = FunctionSpace(self.Vt.mesh(), 'DG', 1) # space with only vertex DOF
        self.vertex_limiter = VertexBasedLimiter(self.Q1DG)
        self.theta_hat = Function(self.Q1DG) # theta function with only vertex DOF

    def copy_vertex_values(self, field):
        """
        Copies the vertex values from temperature space to
        Q1DG space which only has vertices.
        """

    def copy_vertex_values_back(self, field):
        """
        Copies the vertex values back from the Q1DG space to
        the original temperature space.
        """

    def check_midpoint_values(self, field):
        """
        Checks the midpoint field values are less than the maximum
        and more than the minimum values. Amends them to the average
        if they are not.
        """

    def remap_to_embedded_space(self, field):
        """
        Not entirely sure yet. Remap to embedded space?
        """

    def apply(self, field):
        """
        Re-computes centroids and applies limiter to given field
        """
        assert field.function_space() == self.Vt, \
            'Given field does not belong to this objects function space'

        self.copy_vertex_values(field)
        self.vertex_limiter.apply(self.Q1DG)
        self.copy_vertex_values_back(field)
        self.check_midpoint_values(field)
        self.remap_to_embedded_space(field)
