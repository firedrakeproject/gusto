"""
The recovery operators used for lowest-order advection schemes.
"""
from gusto.configuration import logger
from firedrake import expression, function, Function, FunctionSpace, Projector, \
    VectorFunctionSpace, SpatialCoordinate, as_vector, Constant, dx, Interpolator
from firedrake.utils import cached_property
from firedrake.parloops import par_loop, READ, INC, RW
from pyop2 import ON_TOP, ON_BOTTOM
import ufl
import numpy as np

__all__ = ["Averager", "Boundary_Recoverer", "Recoverer"]


class Averager(object):
    """
    An object that 'recovers' a low order field (e.g. in DG0)
    into a higher order field (e.g. in CG1).
    The code is essentially that of the Firedrake Projector
    object, using the "average" method, and could possibly
    be replaced by it if it comes into the master branch.

    :arg v: the :class:`ufl.Expr` or
         :class:`.Function` to project.
    :arg v_out: :class:`.Function` to put the result in.
    """

    def __init__(self, v, v_out):

        if isinstance(v, expression.Expression) or not isinstance(v, (ufl.core.expr.Expr, function.Function)):
            raise ValueError("Can only recover UFL expression or Functions not '%s'" % type(v))

        # Check shape values
        if v.ufl_shape != v_out.ufl_shape:
            raise RuntimeError('Shape mismatch between source %s and target function spaces %s in project' % (v.ufl_shape, v_out.ufl_shape))

        self._same_fspace = (isinstance(v, function.Function) and v.function_space() == v_out.function_space())
        self.v = v
        self.v_out = v_out
        self.V = v_out.function_space()

        # Check the number of local dofs
        if self.v_out.function_space().finat_element.space_dimension() != self.v.function_space().finat_element.space_dimension():
            raise RuntimeError("Number of local dofs for each field must be equal.")

        # NOTE: Any bcs on the function self.v should just work.
        # Loop over node extent and dof extent
        self._shapes = (self.V.finat_element.space_dimension(), np.prod(self.V.shape))
        self._average_kernel = """
        for (int i=0; i<%d; ++i) {
        for (int j=0; j<%d; ++j) {
        vo[i][j] += v[i][j]/w[i][j];
        }}""" % self._shapes

    @cached_property
    def _weighting(self):
        """
        Generates a weight function for computing a projection via averaging.
        """
        w = Function(self.V)
        weight_kernel = """
        for (int i=0; i<%d; ++i) {
        for (int j=0; j<%d; ++j) {
        w[i][j] += 1.0;
        }}""" % self._shapes

        par_loop(weight_kernel, ufl.dx, {"w": (w, INC)})
        return w

    def project(self):
        """
        Apply the recovery.
        """

        # Ensure that the function being populated is zeroed out
        self.v_out.dat.zero()
        par_loop(self._average_kernel, ufl.dx, {"vo": (self.v_out, INC),
                                                "w": (self._weighting, READ),
                                                "v": (self.v, READ)})
        return self.v_out


class Boundary_Recoverer(object):
    """
    An object that performs a `recovery` process at the domain
    boundaries that has second order accuracy. This is necessary
    because the :class:`Averager` object does not recover a field
    with sufficient accuracy at the boundaries.

    The strategy is to minimise the curvature of the function in
    the boundary cells, subject to the constraints of conserved
    mass and continuity on the interior facets. The quickest way
    to perform this is by using the analytic solution and a parloop.

    Currently this is only implemented for the (DG0, DG1, CG1)
    set of spaces, and only on a `PeriodicIntervalMesh` or
    'PeriodicUnitIntervalMesh` that has been extruded.

    :arg v0: the function providing the mass conservation
             constraints. Should be in DG0 and the initial
             function before the recovery process.
    :arg v1: the continuous function providing the continuity
             constraints. Should be in CG1 and is the field
             output by the initial recovery process.
    :arg v_out: the function to be output. Should be in DG1.
    """

    def __init__(self, v0, v1, v_out):

        self.v_out = v_out
        self.v0 = v0
        self.v1 = v1
        VDG0 = FunctionSpace(v_out.function_space().mesh(), "DG", 0)

        # check function spaces of functions -- this only works for a particular set
        if v0.function_space() != VDG0:
            raise NotImplementedError("We can currently only do boundary recovery when v0 is in DG0.")
        if v1.function_space() != FunctionSpace(v1.function_space().mesh(), "CG", 1):
            raise NotImplementedError("We can currently only do boundary recovery when v1 is in CG1.")
        if v_out.function_space() != FunctionSpace(v_out.function_space().mesh(), "DG", 1):
            raise NotImplementedError("We can currently only do boundary recovery when v_out is in DG1.")

        VuDG1 = VectorFunctionSpace(VDG0.mesh(), "DG", 1)
        x, z = SpatialCoordinate(VDG0.mesh())
        self.coords = Function(VuDG1).project(as_vector([x, z]))
        self.interpolator = Interpolator(self.v1, self.v_out)

        # check that we're using quads on extruded mesh -- otherwise it will fail!
        if not VDG0.extruded:
            raise NotImplementedError("This code only works on extruded quadrilateral meshes.")

        logger.warning('This boundary recovery method is bespoke: it should only be used extruded meshes based on a periodic interval in 2D.')

        # make DG0 field that is one in rightmost cells, but zero otherwise
        # this is done as the DOF numbering is different in the rightmost cells
        max_coord = Function(VDG0).interpolate(Constant(np.max(self.coords.dat.data[:, 0])))
        self.right = Function(VDG0)
        right_kernel = """
        if (fmax(COORDS[0][0], fmax(COORDS[1][0], COORDS[2][0])) == MAX[0][0])
            RIGHT[0][0] = 1.0;
        """
        par_loop(right_kernel, dx,
                 args={"COORDS": (self.coords, READ),
                       "MAX": (max_coord, READ),
                       "RIGHT": (self.right, RW)})

        self.bottom_kernel = """
        if (RIGHT[0][0] == 1.0)
        {
        float x = COORDS[2][0] - COORDS[0][0];
        float y = COORDS[1][1] - COORDS[0][1];
        float a = CG1[3][0];
        float b = CG1[1][0];
        float c = DG0[0][0];
        DG1[1][0] = a;
        DG1[3][0] = b;
        DG1[2][0] = (1.0 / (pow(x, 2.0) + 4.0 * pow(y, 2.0))) * (-3.0 * a * pow(y, 2.0) - b * pow(x, 2.0) - b * pow(y, 2.0) + 2.0 * c * pow(x, 2.0) + 8.0 * c * pow(y, 2.0));
        DG1[0][0] = 4.0 * c - b - a - DG1[2][0];
        }
        else
        {
        float x = COORDS[1][0] - COORDS[3][0];
        float y = COORDS[3][1] - COORDS[2][1];
        float a = CG1[1][0];
        float b = CG1[3][0];
        float c = DG0[0][0];
        DG1[3][0] = a;
        DG1[1][0] = b;
        DG1[0][0] = (1.0 / (pow(x, 2.0) + 4.0 * pow(y, 2.0))) * (-3.0 * a * pow(y, 2.0) - b * pow(x, 2.0) - b * pow(y, 2.0) + 2.0 * c * pow(x, 2.0) + 8.0 * c * pow(y, 2.0));
        DG1[2][0] = 4.0 * c - b - a - DG1[0][0];
        }
        """

        self.top_kernel = """
        if (RIGHT[0][0] == 1.0)
        {
        float x = COORDS[2][0] - COORDS[0][0];
        float y = COORDS[1][1] - COORDS[0][1];
        float a = CG1[2][0];
        float b = CG1[0][0];
        float c = DG0[0][0];
        DG1[2][0] = a;
        DG1[0][0] = b;
        DG1[3][0] = (1.0 / (pow(x, 2.0) + 4.0 * pow(y, 2.0))) * (-3.0 * a * pow(y, 2.0) - b * pow(x, 2.0) - b * pow(y, 2.0) + 2.0 * c * pow(x, 2.0) + 8.0 * c * pow(y, 2.0));
        DG1[1][0] = 4.0 * c - b - a - DG1[3][0];
        }
        else
        {
        float x = COORDS[0][0] - COORDS[2][0];
        float y = COORDS[3][1] - COORDS[2][1];
        float a = CG1[2][0];
        float b = CG1[0][0];
        float c = DG0[0][0];
        DG1[0][0] = a;
        DG1[2][0] = b;
        DG1[3][0] = (1.0 / (pow(x, 2.0) + 4.0 * pow(y, 2.0))) * (-3.0 * a * pow(y, 2.0) - b * pow(x, 2.0) - b * pow(y, 2.0) + 2.0 * c * pow(x, 2.0) + 8.0 * c * pow(y, 2.0));
        DG1[1][0] = 4.0 * c - b - a - DG1[3][0];
        }
        """

    def apply(self):

        self.interpolator.interpolate()
        par_loop(self.bottom_kernel, dx,
                 args={"DG1": (self.v_out, RW),
                       "CG1": (self.v1, READ),
                       "DG0": (self.v0, READ),
                       "COORDS": (self.coords, READ),
                       "RIGHT": (self.right, READ)},
                 iterate=ON_BOTTOM)

        par_loop(self.top_kernel, dx,
                 args={"DG1": (self.v_out, RW),
                       "CG1": (self.v1, READ),
                       "DG0": (self.v0, READ),
                       "COORDS": (self.coords, READ),
                       "RIGHT": (self.right, READ)},
                 iterate=ON_TOP)


class Recoverer(object):
    """
    An object that 'recovers' a field from a low order space
    (e.g. DG0) into a higher order space (e.g. CG1). This encompasses
    the process of interpolating first to a the right space before
    using the :class:`Averager` object, and also automates the
    boundary recovery process. If no boundary method is specified,
    this simply performs the action of the :class: `Averager`.

    :arg v_in: the :class:`ufl.Expr` or
         :class:`.Function` to project. (e.g. a VDG0 function)
    :arg v_out: :class:`.Function` to put the result in. (e.g. a CG1 function)
    :arg VDG: optional :class:`.FunctionSpace`. If not None, v_in is interpolated
         to this space first before recovery happens.
    :arg boundary_method: a string defining which type of method needs to be
         used at the boundaries. Valid options are 'density' or 'velocity'.
    """

    def __init__(self, v_in, v_out, VDG=None, boundary_method=None):

        # check if v_in is valid
        if isinstance(v_in, expression.Expression) or not isinstance(v_in, (ufl.core.expr.Expr, function.Function)):
            raise ValueError("Can only recover UFL expression or Functions not '%s'" % type(v_in))

        self.v_in = v_in
        self.v_out = v_out
        self.V = v_out.function_space()
        if VDG is not None:
            self.v = Function(VDG)
            self.interpolator = Interpolator(v_in, self.v)
        else:
            self.v = v_in
            self.interpolator = None

        self.VDG = VDG
        self.boundary_method = boundary_method
        self.averager = Averager(self.v, self.v_out)

        # check boundary method options are valid
        if boundary_method is not None:
            if boundary_method != 'density' and boundary_method != 'velocity':
                raise ValueError("Specified boundary_method % not valid" % boundary_method)
            if VDG is None:
                raise ValueError("If boundary_method is specified, VDG also needs specifying.")

            # now specify things that we'll need if we are doing boundary recovery
            if boundary_method == 'density':
                # check dimensions
                if self.V.value_size != 1:
                    raise ValueError('This method only works for scalar functions.')
                self.boundary_recoverer = Boundary_Recoverer(self.v_in, self.v_out, self.v)
            if boundary_method == 'velocity':
                # check dimensions
                if self.V.value_size != 2:
                    raise ValueError('This method only works for 2D vector functions.')
                # declare spaces and functions manually for the scalar field
                self.VDG0 = FunctionSpace(self.VDG.mesh(), "DG", 0)
                self.VCG1 = FunctionSpace(self.VDG.mesh(), "CG", 1)
                self.VDG1 = FunctionSpace(self.VDG.mesh(), "DG", 1)
                self.v_in_scalar = Function(self.VDG0)
                self.v_scalar = Function(self.VDG1)
                self.v_out_scalar = Function(self.VCG1)

                self.boundary_recoverer = Boundary_Recoverer(self.v_in_scalar, self.v_out_scalar, self.v_scalar)
                # the boundary recoverer needs to be done on a scalar fields
                # so need to extract component and restore it after the boundary recovery is done
                self.project_to_vector = Projector(as_vector([self.v_out_scalar, self.v_out[1]]), self.v_out)
                self.project_to_scalar_DG = Projector(self.v_in[0], self.v_in_scalar)
                self.project_to_scalar_CG = Projector(self.v_out[0], self.v_out_scalar)
                # need an extra averager that works on the scalar fields rather than the vector one
                self.extra_averager = Averager(self.v_scalar, self.v_out_scalar)

    def extract_scalar(self):
        self.project_to_scalar_DG.project()
        self.project_to_scalar_CG.project()

    def restore_vector(self):
        self.project_to_vector.project()

    def project(self):
        """
        Perform the fully specified recovery.
        """

        if self.interpolator is not None:
            self.interpolator.interpolate()
        self.averager.project()
        if self.boundary_method is not None:
            if self.boundary_method == 'velocity':
                self.extract_scalar()
                self.boundary_recoverer.apply()
                self.extra_averager.project()
                self.restore_vector()
            elif self.boundary_method == 'density':
                self.boundary_recoverer.apply()
                self.averager.project()
        return self.v_out
