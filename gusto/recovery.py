"""
The recovery operators used for lowest-order advection schemes.
"""
from firedrake import (expression, function, Function, FunctionSpace, Projector,
                       VectorFunctionSpace, SpatialCoordinate, as_vector,
                       dx, Interpolator, BrokenElement, interval, Constant,
                       TensorProductElement, FiniteElement, DirichletBC,
                       VectorElement)
from firedrake.utils import cached_property
from firedrake.parloops import par_loop, READ, INC, RW
from gusto.transport_equation import is_dg
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

    :arg v_CG1: the continuous function after the first recovery
             is performed. Should be in CG1. This is correct
             on the interior of the domain.
    :arg v_DG1: the function to be output. Should be in DG1.
    :arg method: string giving the method used for the recovery.
             Valid options are 'dynamics' and 'physics'.
    :arg coords_to_adjust: a DG1 field containing 1 at locations of
                           coords that must be adjusted to give they
                           effective coords.
    """

    def __init__(self, v_CG1, v_DG1, method='physics', coords_to_adjust=None):

        self.v_DG1 = v_DG1
        self.v_CG1 = v_CG1
        self.coords_to_adjust = coords_to_adjust

        self.method = method
        mesh = v_CG1.function_space().mesh()
        VDG0 = FunctionSpace(mesh, "DG", 0)
        VCG1 = FunctionSpace(mesh, "CG", 1)
        VDG1 = FunctionSpace(mesh, "DG", 1)

        # check function spaces of functions
        if self.method == 'dynamics':
            if v_CG1.function_space() != VCG1:
                raise NotImplementedError("This boundary recovery method requires v1 to be in CG1.")
            if v_DG1.function_space() != VDG1:
                raise NotImplementedError("This boundary recovery method requires v_out to be in DG1.")
            # check whether mesh is valid
            if mesh.topological_dimension() == 2:
                # if mesh is extruded then we're fine, but if not needs to be quads
                if not VDG0.extruded and mesh.ufl_cell().cell_name() != 'quadrilateral':
                    raise NotImplementedError('For 2D meshes this recovery method requires that elements are quadrilaterals')
            elif mesh.topological_dimension() == 3:
                # assume that 3D mesh is extruded
                if mesh._base_mesh.ufl_cell().cellname() != 'quadrilateral':
                    raise NotImplementedError('For 3D extruded meshes this recovery method requires a base mesh with quadrilateral elements')
            elif mesh.topological_dimension() != 1:
                raise NotImplementedError('This boundary recovery is implemented only on certain classes of mesh.')
            if coords_to_adjust is None:
                raise ValueError('Need coords_to_adjust field for dynamics boundary methods')

        elif self.method == 'physics':
            # check that mesh is valid -- must be an extruded mesh
            if not VDG0.extruded:
                raise NotImplementedError('The physics boundary method only works on extruded meshes')
            # base spaces
            cell = mesh._base_mesh.ufl_cell().cellname()
            w_hori = FiniteElement("DG", cell, 0)
            w_vert = FiniteElement("CG", interval, 1)
            # build element
            theta_element = TensorProductElement(w_hori, w_vert)
            # spaces
            Vtheta = FunctionSpace(mesh, theta_element)
            Vtheta_broken = FunctionSpace(mesh, BrokenElement(theta_element))
            if v_CG1.function_space() != Vtheta:
                raise ValueError("This boundary recovery method requires v_CG1 to be in DG0xCG1 TensorProductSpace.")
            if v_DG1.function_space() != Vtheta_broken:
                raise ValueError("This boundary recovery method requires v_DG1 to be in the broken DG0xCG1 TensorProductSpace.")
        else:
            raise ValueError("Specified boundary_method % not valid" % self.method)

        VuDG1 = VectorFunctionSpace(VDG0.mesh(), "DG", 1)
        x = SpatialCoordinate(VDG0.mesh())
        self.interpolator = Interpolator(self.v_CG1, self.v_DG1)

        if self.method == 'dynamics':

            # STRATEGY
            # obtain a coordinate field for all the nodes
            VuDG1 = VectorFunctionSpace(mesh, "DG", 1)
            self.act_coords = Function(VuDG1).project(x)  # actual coordinates
            self.eff_coords = Function(VuDG1).project(x)  # effective coordinates

            find_effective_coords_kernel = """
            int dim = %d;

            /* find num of DOFs to adjust in this cell, DG1 */
            int sum_V1_ext = 0;
            for (int i=0; i<%d; ++i) {
            sum_V1_ext += round(EXT_V1[i][0]);}

            /* only do adjustment in cells with at least one DOF to adjust */
            if (sum_V1_ext > 0) {

            /* find max dist */
            float max_dist = 0;
            for (int i=0; i<%d; ++i){
            for (int j=0; j<%d; ++j) {
            float dist = 0;
            for (int k=0; k<dim; ++k) {
            dist += pow(ACT_COORDS[i][k] - ACT_COORDS[j][k], 2);}
            dist = pow(dist, 0.5);
            max_dist = fmax(dist, max_dist);}}

            /* loop through DOFs in cell and find which ones to adjust */
            for (int i=0; i<%d; ++i) {
            if (round(EXT_V1[i][0]) == 1){

            /* find closest interior node */
            float min_dist = max_dist;
            int index = -1;
            for (int j=0; j<%d; ++j) {
            if (round(EXT_V1[j][0]) == 0) {
            float dist = 0;
            for (int k=0; k<dim; ++k) {
            dist += pow(ACT_COORDS[i][k] - ACT_COORDS[j][k], 2);}
            dist = pow(dist, 0.5);
            if (dist <= min_dist) {
            min_dist = dist;
            index = j;}}}

            /* adjust coordinate */
            for (int j=0; j<dim; ++j) {
            EFF_COORDS[i][j] = 0.5 * (ACT_COORDS[i][j] + ACT_COORDS[index][j]);
            }
            }
            }
            }

            /* else do nothing */

            """ % ((np.prod(VuDG1.shape),) * 1 + (self.v_DG1.function_space().finat_element.space_dimension(), ) * 5)

            par_loop(find_effective_coords_kernel, dx,
                     args={"EXT_V1": (self.coords_to_adjust, READ),
                           "ACT_COORDS": (self.act_coords, READ),
                           "EFF_COORDS": (self.eff_coords, RW)})

            self.gaussian_elimination_kernel = """
            int n = %d;
            /* do gaussian elimination to find constants in linear expansion */
            /* trying to solve A*a = f for a, where A is a matrix */
            double A[%d][%d], a[%d], f[%d], c;
            double A_max, temp_A, temp_f;
            int i_max, i, j, k;

            /* find number of exterior nodes per cell */
            int sum_V1_ext = 0;
            for (int i=0; i<%d; i++) {
            sum_V1_ext += round(EXT_V1[i][0]);}

            /* ask if there are any exterior nodes */
            if (sum_V1_ext > 0) {

            /* fill A and f with their values */
            for (i=0; i<%d; i++) {
            f[i] = DG1[i][0];
            A[i][0] = 1.0;
            A[i][1] = EFF_COORDS[i][0];
            if (n == 4 || n == 8){
            A[i][2] = EFF_COORDS[i][1];
            A[i][3] = EFF_COORDS[i][0] * EFF_COORDS[i][1];
            if (n == 8){
            A[i][4] = EFF_COORDS[i][2];
            A[i][5] = EFF_COORDS[i][0] * EFF_COORDS[i][2];
            A[i][6] = EFF_COORDS[i][1] * EFF_COORDS[i][2];
            A[i][7] = EFF_COORDS[i][0] * EFF_COORDS[i][1] * EFF_COORDS[i][2];}}}

            /* do Gaussian elimination */
            for (i=0; i<%d-1; i++) {
            /* loop through rows and columns */
            A_max = fabs(A[i][i]);
            i_max = i;

            /* find max value in ith column */
            for (j=i+1; j<%d; j++){
            if (fabs(A[j][i]) > A_max) { /* loop through rows below ith row */
            A_max = fabs(A[j][i]);
            i_max = j;}}

            /* swap rows to get largest value in ith column on top */
            if (i_max != i){
            temp_f = f[i];
            f[i] = f[i_max];
            f[i_max] = temp_f;
            for (k=i; k<%d; k++) {
            temp_A = A[i][k];
            A[i][k] = A[i_max][k];
            A[i_max][k] = temp_A;}}

            /* now scale rows below to eliminate lower diagonal values */
            for (j=i+1; j<%d; j++) {
            c = -A[j][i] / A[i][i];
            for (k=i; k<%d; k++){
            A[j][k] += c * A[i][k];}
            f[j] += c * f[i];}}

            /* do back-substitution to acquire solution */
            for (i=0; i<%d; i++){
            j = n-i-1;
            a[j] = f[j];
            for(k=j+1; k<=%d; ++k) {
            a[j] -= A[j][k] * a[k];}
            a[j] = a[j] / A[j][j];
            }

            /* extrapolate solution using new coordinates */
            for (i=0; i<%d; i++) {
            if (n == 2){
            DG1[i][0] = a[0] + a[1]*ACT_COORDS[i][0];
            }
            else if (n == 4){
            DG1[i][0] = a[0] + a[1]*ACT_COORDS[i][0] + a[2]*ACT_COORDS[i][1] + a[3]*ACT_COORDS[i][0]*ACT_COORDS[i][1];
            }
            else if (n == 8){
            DG1[i][0] = a[0] + a[1]*ACT_COORDS[i][0] + a[2]*ACT_COORDS[i][1] + a[4]*ACT_COORDS[i][2] + a[3]*ACT_COORDS[i][0]*ACT_COORDS[i][1] + a[5]*ACT_COORDS[i][0]*ACT_COORDS[i][2] + a[6]*ACT_COORDS[i][1]*ACT_COORDS[i][2] + a[7]*ACT_COORDS[i][0]*ACT_COORDS[i][1]*ACT_COORDS[i][2];
            }}
            }
            /* do nothing if there are no exterior nodes */
            """ % ((self.v_DG1.function_space().finat_element.space_dimension(), )*15)

        elif self.method == 'physics':
            self.bottom_kernel = """
            DG1[0][0] = CG1[1][0] - 2 * (CG1[1][0] - CG1[0][0]);
            DG1[1][0] = CG1[1][0];
            """

            self.top_kernel = """
            DG1[1][0] = CG1[0][0] - 2 * (CG1[0][0] - CG1[1][0]);
            DG1[0][0] = CG1[0][0];
            """

    def apply(self):
        self.interpolator.interpolate()
        if self.method == 'physics':
            par_loop(self.bottom_kernel, dx,
                     args={"DG1": (self.v_DG1, RW),
                           "CG1": (self.v_CG1, READ)},
                     iterate=ON_BOTTOM)

            par_loop(self.top_kernel, dx,
                     args={"DG1": (self.v_DG1, RW),
                           "CG1": (self.v_CG1, READ)},
                     iterate=ON_TOP)
        else:
            par_loop(self.gaussian_elimination_kernel, dx,
                     args={"DG1": (self.v_DG1, RW),
                           "ACT_COORDS": (self.act_coords, READ),
                           "EFF_COORDS": (self.eff_coords, READ),
                           "EXT_V1": (self.coords_to_adjust, READ)})


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
         used at the boundaries. Valid options are 'density', 'velocity' or 'physics'.
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
            if boundary_method != 'scalar' and boundary_method != 'vector' and boundary_method != 'physics':
                raise ValueError("Specified boundary_method %s not valid" % boundary_method)
            if VDG is None:
                raise ValueError("If boundary_method is specified, VDG also needs specifying.")

            # now specify things that we'll need if we are doing boundary recovery
            if boundary_method == 'physics':
                # check dimensions
                if self.V.value_size != 1:
                    raise ValueError('This method only works for scalar functions.')
                self.boundary_recoverer = Boundary_Recoverer(self.v_out, self.v, method='physics')
            else:

                mesh = self.V.mesh()
                V0_brok = FunctionSpace(mesh, BrokenElement(self.v_in.ufl_element()))
                VDG1 = FunctionSpace(mesh, "DG", 1)
                VCG1 = FunctionSpace(mesh, "CG", 1)

                if boundary_method == 'scalar':
                    # check dimensions
                    if self.V.value_size != 1:
                        raise ValueError('This method only works for scalar functions.')

                    VCG2 = FunctionSpace(mesh, "CG", 2)
                    coords_to_adjust = find_coords_to_adjust(V0_brok, VDG1, VCG1, VCG2)

                    self.boundary_recoverer = Boundary_Recoverer(self.v_out, self.v,
                                                                 coords_to_adjust=coords_to_adjust,
                                                                 method='dynamics')
                elif boundary_method == 'vector':
                    # check dimensions
                    if self.V.value_size != 2 and self.V.value_size != 3:
                        raise NotImplementedError('This method only works for 2D or 3D vector functions.')

                    VuCG1 = VectorFunctionSpace(mesh, "CG", 1)
                    VuCG2 = VectorFunctionSpace(mesh, "CG", 2)
                    VuDG1 = VectorFunctionSpace(mesh, "DG", 1)
                    coords_to_adjust = find_coords_to_adjust(V0_brok, VuDG1, VuCG1, VuCG2)

                    # now, break the problem down into components
                    v_scalars = []
                    v_out_scalars = []
                    self.boundary_recoverers = []
                    self.project_to_scalars_CG = []
                    self.extra_averagers = []
                    coords_to_adjust_list = []
                    for i in range(self.V.value_size):
                        v_scalars.append(Function(VDG1))
                        v_out_scalars.append(Function(VCG1))
                        coords_to_adjust_list.append(Function(VDG1).project(coords_to_adjust[i]))
                        self.project_to_scalars_CG.append(Projector(self.v_out[i], v_out_scalars[i]))
                        self.boundary_recoverers.append(Boundary_Recoverer(v_out_scalars[i], v_scalars[i],
                                                                           method='dynamics',
                                                                           coords_to_adjust=coords_to_adjust_list[i]))
                        # need an extra averager that works on the scalar fields rather than the vector one
                        self.extra_averagers.append(Averager(v_scalars[i], v_out_scalars[i]))

                    # the boundary recoverer needs to be done on a scalar fields
                    # so need to extract component and restore it after the boundary recovery is done
                    self.project_to_vector = Projector(as_vector(v_out_scalars), self.v_out)

    def project(self):
        """
        Perform the fully specified recovery.
        """

        if self.interpolator is not None:
            self.interpolator.interpolate()
        self.averager.project()
        if self.boundary_method is not None:
            if self.boundary_method == 'vector':
                for i in range(self.V.value_size):
                    self.project_to_scalars_CG[i].project()
                    self.boundary_recoverers[i].apply()
                    self.extra_averagers[i].project()
                self.project_to_vector.project()
            elif self.boundary_method == 'scalar' or self.boundary_method == 'physics':
                self.boundary_recoverer.apply()
                self.averager.project()
        return self.v_out


def find_number_of_exterior_DOFs_per_cell(field, output):
    """
    Finds the number of DOFs on the domain exterior
    per cell and stores it in a DG0 field.

    :arg field: the input field, containing a 1 at each
                exterior DOF and a 0 at each interior DOF.
    :arg output: a DG0 field to be output to.
    """

    shapes = field.function_space().finat_element.space_dimension()
    kernel = """
    for (int i=0; i<%d; ++i) {
    DG0[0][0] += ext[0][i];}
    """ % shapes

    par_loop(kernel, dx,
             args={"DG0": (output, RW),
                   "ext": (field, READ)})


def find_coords_to_adjust(V0_brok, DG1, CG1, CG2):
    """
    This function finds the coordinates that need to be adjusted
    for the recovery at the boundary. These are assigned by a 1,
    while all coordinates to be left unchanged are assigned a 0.
    This field is returned as a DG1 field.
    Fields can be scalar or vector.

    :arg V0_brok: the broken space of the original field (before recovery).
    :arg DG1: a DG1 space, in which the boundary recovery will happen.
    :arg CG1: a CG1 space, in which the recovered field lies.
    :arg CG2: a CG2 space.
    """

    # check that spaces are correct
    mesh = DG1.mesh()
    # check V0_brok is fully discontinuous
    if not is_dg(V0_brok):
        raise ValueError('Need V0_brok to be a fully discontinuous space.')
    # check DG1, CG1 and CG2 fields are correct
    for space, family, degree in zip((DG1, CG1, CG2), ("DG", "CG", "CG"), (1, 1, 2)):
        if type(space.ufl_element()) == VectorElement:
            if space != VectorFunctionSpace(mesh, family, degree):
                raise ValueError('The function space entered as vector %s%s is not vector %s%s.' % (family, degree, family, degree))
        elif space != FunctionSpace(mesh, family, degree):
            raise ValueError('The function space entered as %s%s is not %s%s.' % (family, degree, family, degree))

    # STRATEGY
    # We need to pass the boundary recoverer a field denoting the location
    # of nodes on the boundary, which their coordinates adjusting to new effective
    # coords. This field will be 1 for these coords and 0 otherwise.
    # How do we do this?
    # 1. Obtain a DG1 field which is 1 at all exterior DOFs by applying Dirichlet
    #    boundary conditions.
    # 2. Obtain a field in DG1 that is 1 at exterior DOFs neighbouring the exterior
    #    values of V0 (i.e. the original space). For V0=DG0 there will be no exterior
    #    values, but could be if velocity is in RT or if there is a temperature space.
    #    This requires a couple of steps:
    #    a) Obtain a CG2 field with all the correct exterior values. CG2 is chosen because
    #       all DOFs in V0 should have a DOF at the same position in CG2.
    #    b) Project this down into V0,  which should give a good approximation to
    #       having an exterior field in V0, although values will not be 1 necessarily.
    #       Projection is required because interpolation is not supported for fields
    #       whose values aren't pointwise evaluations (i.e. velocity spaces).
    #    c) Interpolate these values into DG1, so this field is correctly represented
    #       at the same points as the exterior field.
    #    d) Because of the projection step, values will not necessarily be 1.
    #       Values that should be 1 *should* be over 1/2 (found by trial and error!),
    #       and we correct each point individually so that they become 1 or 0.
    # 3. Obtain a field that is 1 at corners in 2D or along edges in 3D.
    #    We do this by using that corners in 2D and edges in 3D are intersections
    #    of edges/faces respectively. In 2D, this means that if a field which is 1 on a
    #    horizontal edge is summed with a field that is 1 on a vertical edge, the
    #    corner value will be 2. Subtracting the exterior DG1 field from step 1 leaves
    #    a field that is 1 in the corner. This is generalised to 3D.
    # 4. The field of coords to be adjusted is then found by the following formula:
    #                            f1 + f3 - f2
    #    where f1, f2 and f3 are the DG1 fields obtained from steps 1, 2 and 3.

    # make DG1 field with 1 at all exterior coords
    all_ext_in_DG1 = Function(DG1)
    bcs = [DirichletBC(DG1, Constant(1.0), "on_boundary", method="geometric")]

    if DG1.extruded:
        bcs.append(DirichletBC(DG1, Constant(1.0), "top", method="geometric"))
        bcs.append(DirichletBC(DG1, Constant(1.0), "bottom", method="geometric"))

    for bc in bcs:
        bc.apply(all_ext_in_DG1)

    # make DG1 field with 1 at coords surrounding exterior coords of V0
    # first do it in CG2, as this should contain all DOFs of V0 and DG1
    all_ext_in_CG2 = Function(CG2)
    bcs = [DirichletBC(CG2, Constant(1.0), "on_boundary", method="geometric")]

    if CG2.extruded:
        bcs.append(DirichletBC(CG2, Constant(1.0), "top", method="geometric"))
        bcs.append(DirichletBC(CG2, Constant(1.0), "bottom", method="geometric"))

    for bc in bcs:
        bc.apply(all_ext_in_CG2)

    approx_ext_in_V0 = Function(V0_brok).project(all_ext_in_CG2)
    approx_V0_ext_in_DG1 = Function(DG1).interpolate(approx_ext_in_V0)

    # now do horrible hack to get back to 1s and 0s
    for (i, point) in enumerate(approx_V0_ext_in_DG1.dat.data[:]):
        if type(DG1.ufl_element()) == VectorElement:
            for (j, p) in enumerate(point):
                if p > 0.5:
                    approx_V0_ext_in_DG1.dat.data[i][j] = 1
                else:
                    approx_V0_ext_in_DG1.dat.data[i][j] = 0
        else:
            if point > 0.5:
                approx_V0_ext_in_DG1.dat.data[i] = 1
            else:
                approx_V0_ext_in_DG1.dat.data[i] = 0

    corners_in_DG1 = Function(DG1)
    if DG1.mesh().topological_dimension() == 2:
        if DG1.extruded:
            DG1_ext_hori = Function(DG1)
            DG1_ext_vert = Function(DG1)
            hori_bcs = [DirichletBC(DG1, Constant(1.0), "top", method="geometric"),
                        DirichletBC(DG1, Constant(1.0), "bottom", method="geometric")]
            vert_bc = DirichletBC(DG1, Constant(1.0), "on_boundary", method="geometric")
            for bc in hori_bcs:
                bc.apply(DG1_ext_hori)

            vert_bc.apply(DG1_ext_vert)
            corners_in_DG1.assign(DG1_ext_hori + DG1_ext_vert - all_ext_in_DG1)

        else:
            # we don't know whether its periodic or in how many directions
            DG1_ext_x = Function(DG1)
            DG1_ext_y = Function(DG1)
            x_bcs = [DirichletBC(DG1, Constant(1.0), 1, method="geometric"),
                     DirichletBC(DG1, Constant(1.0), 2, method="geometric")]
            y_bcs = [DirichletBC(DG1, Constant(1.0), 3, method="geometric"),
                     DirichletBC(DG1, Constant(1.0), 4, method="geometric")]

            # there is no easy way to know if the mesh is periodic or in which
            # directions, so we must use a try here
            # LookupError is the error for asking for a boundary number that doesn't exist
            try:
                for bc in x_bcs:
                    bc.apply(DG1_ext_x)
            except LookupError:
                pass
            try:
                for bc in y_bcs:
                    bc.apply(DG1_ext_y)
            except LookupError:
                pass

            corners_in_DG1.assign(DG1_ext_x + DG1_ext_y - all_ext_in_DG1)

    elif DG1.mesh().topological_dimension() == 3:
        DG1_vert_x = Function(DG1)
        DG1_vert_y = Function(DG1)
        DG1_hori = Function(DG1)
        x_bcs = [DirichletBC(DG1, Constant(1.0), 1, method="geometric"),
                 DirichletBC(DG1, Constant(1.0), 2, method="geometric")]
        y_bcs = [DirichletBC(DG1, Constant(1.0), 3, method="geometric"),
                 DirichletBC(DG1, Constant(1.0), 4, method="geometric")]
        hori_bcs = [DirichletBC(DG1, Constant(1.0), "top", method="geometric"),
                    DirichletBC(DG1, Constant(1.0), "bottom", method="geometric")]

        # there is no easy way to know if the mesh is periodic or in which
        # directions, so we must use a try here
        # LookupError is the error for asking for a boundary number that doesn't exist
        try:
            for bc in x_bcs:
                bc.apply(DG1_vert_x)
        except LookupError:
            pass

        try:
            for bc in y_bcs:
                bc.apply(DG1_vert_y)
        except LookupError:
            pass

        for bc in hori_bcs:
            bc.apply(DG1_hori)

        corners_in_DG1.assign(DG1_vert_x + DG1_vert_y + DG1_hori - all_ext_in_DG1)

    coords_to_correct = Function(DG1).assign(corners_in_DG1 + all_ext_in_DG1 - approx_V0_ext_in_DG1)
    for (i, point) in enumerate(coords_to_correct.dat.data[:]):
        if type(DG1.ufl_element()) == VectorElement:
            for (j, p) in enumerate(point):
                if p > 0.5:
                    coords_to_correct.dat.data[i][j] = 1
                else:
                    coords_to_correct.dat.data[i][j] = 0
        else:
            if point > 0.5:
                coords_to_correct.dat.data[i] = 1
            else:
                coords_to_correct.dat.data[i] = 0

    return coords_to_correct
