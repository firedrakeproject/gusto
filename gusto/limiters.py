from __future__ import absolute_import, print_function, division
from gusto.configuration import logger
from firedrake import dx
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.parloops import par_loop, READ, RW, INC
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter

__all__ = ["ThetaLimiter", "NoLimiter"]

_copy_into_Q1DG_loop = """
theta_hat[0][0] = theta[0][0];
theta_hat[1][0] = theta[1][0];
theta_hat[2][0] = theta[3][0];
theta_hat[3][0] = theta[4][0];
"""
_copy_from_Q1DG_loop = """
theta[0][0] = theta_hat[0][0];
theta[1][0] = theta_hat[1][0];
theta[3][0] = theta_hat[2][0];
theta[4][0] = theta_hat[3][0];
"""

_check_midpoint_values_loop = """
if (theta[2][0] > fmax(theta[0][0], theta[1][0]))
    theta[2][0] = 0.5 * (theta[0][0] + theta[1][0]);
else if (theta[2][0] < fmin(theta[0][0], theta[1][0]))
    theta[2][0] = 0.5 * (theta[0][0] + theta[1][0]);
if (theta[5][0] > fmax(theta[3][0], theta[4][0]))
    theta[5][0] = 0.5 * (theta[3][0] + theta[4][0]);
else if (theta[5][0] < fmin(theta[3][0], theta[4][0]))
    theta[5][0] = 0.5 * (theta[3][0] + theta[4][0]);
"""

_weight_kernel = """
for (int i=0; i<weight.dofs; ++i) {
    weight[i][0] += 1.0;
    }"""

_average_kernel = """
for (int i=0; i<vrec.dofs; ++i) {
        vrec[i][0] += v_b[i][0]/weight[i][0];
        }"""


class ThetaLimiter(object):
    """
    A vertex based limiter for fields in the DG1xCG2 space,
    i.e. temperature variables. This acts like the vertex-based
    limiter implemented in Firedrake, but in addition corrects
    the central nodes to prevent new maxima or minima forming.
    """

    def __init__(self, equation):
        """
        Initialise limiter

        :param space : equation, as we need the broken space attached to it
        """

        self.Vt = equation.space
        # check this is the right space, only currently working for 2D extruded mesh
        if self.Vt.extruded and self.Vt.mesh().topological_dimension() == 2:
            # check that horizontal degree is 1 and vertical degree is 2
            if self.Vt.ufl_element().degree()[0] is not 1 or \
               self.Vt.ufl_element().degree()[1] is not 2:
                raise ValueError('This is not the right limiter for this space.')
            # check that continuity of the spaces is correct
            # this will fail if the space does not use broken elements
            if self.Vt.ufl_element()._element.sobolev_space()[0].name is not 'L2' or \
               self.Vt.ufl_element()._element.sobolev_space()[1].name is not 'H1':
                raise ValueError('This is not the right limiter for this space.')
        else:
            logger.warning('This limiter may not work for the space you are using.')

        self.Q1DG = FunctionSpace(self.Vt.mesh(), 'DG', 1)  # space with only vertex DOFs
        self.vertex_limiter = VertexBasedLimiter(self.Q1DG)
        self.theta_hat = Function(self.Q1DG)  # theta function with only vertex DOFs
        self.w = Function(self.Vt)
        self.result = Function(self.Vt)
        par_loop(_weight_kernel, dx, {"weight": (self.w, INC)})

    def copy_vertex_values(self, field):
        """
        Copies the vertex values from temperature space to
        Q1DG space which only has vertices.
        """
        par_loop(_copy_into_Q1DG_loop, dx,
                 {"theta": (field, READ),
                  "theta_hat": (self.theta_hat, RW)})

    def copy_vertex_values_back(self, field):
        """
        Copies the vertex values back from the Q1DG space to
        the original temperature space.
        """
        par_loop(_copy_from_Q1DG_loop, dx,
                 {"theta": (field, RW),
                  "theta_hat": (self.theta_hat, READ)})

    def check_midpoint_values(self, field):
        """
        Checks the midpoint field values are less than the maximum
        and more than the minimum values. Amends them to the average
        if they are not.
        """
        par_loop(_check_midpoint_values_loop, dx,
                 {"theta": (field, RW)})

    def remap_to_embedded_space(self, field):
        """
        Remap from DG space to embedded DG space.
        """

        self.result.assign(0.)
        par_loop(_average_kernel, dx, {"vrec": (self.result, INC),
                                       "v_b": (field, READ),
                                       "weight": (self.w, READ)})
        field.assign(self.result)

    def apply(self, field):
        """
        The application of the limiter to the theta-space field.
        """
        assert field.function_space() == self.Vt, \
            'Given field does not belong to this objects function space'

        self.copy_vertex_values(field)
        self.vertex_limiter.apply(self.theta_hat)
        self.copy_vertex_values_back(field)
        self.check_midpoint_values(field)
        self.remap_to_embedded_space(field)


class NoLimiter(object):
    """
    A blank limiter that does nothing.
    """

    def __init__(self):
        """
        Initialise the blank limiter.
        """
        pass

    def apply(self, field):
        """
        The application of the blank limiter.
        """
        pass
