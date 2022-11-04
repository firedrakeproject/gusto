"""
Operators to recover fields from a discontinuous to a continuous function space.

The recovery operators provided in this module are used to restore continuity
in a discontinuous field, or to reconstruct a higher-order field from a lower-
order field, which can be used to improve the accuracy of lowest-order spatial
discretisations.
"""
from enum import Enum

import ufl
from firedrake import (BrokenElement, Constant, DirichletBC, FiniteElement,
                       Function, FunctionSpace, Interpolator, Projector,
                       SpatialCoordinate, TensorProductElement,
                       VectorFunctionSpace, as_vector, function, interval,
                       VectorElement, BrokenElement)
from gusto.recovery import Averager
from .recovery_kernels import (BoundaryRecoveryExtruded, BoundaryRecoveryHCurl,
                               BoundaryGaussianElimination)


__all__ = ["BoundaryMethod", "BoundaryRecoverer", "Recoverer"]


class BoundaryMethod(Enum):
    """
    Method for correcting the recovery at the domain boundaries.

    An enumerator object encoding methods for correcting boundary recovery:
    extruded: which corrects a scalar field on an extruded mesh at the top and
              bottom boundaries.
    hcurl: this corrects the recovery of a HDiv field into a HCurl space at the
           top and bottom boundaries of an extruded mesh.
    taylor: uses a Taylor expansion to correct the field at all the boundaries
            of the domain. Should only be used in Cartesian domains.
    """

    extruded = 0
    hcurl = 1
    taylor = 2


class BoundaryRecoverer(object):
    """
    Corrects values in domain boundary cells that have been recovered.

    An object that performs a `recovery` process at the domain boundaries that
    has second-order accuracy. This is necessary because the :class:`Averager`
    object does not recover a field with sufficient accuracy at the boundaries.

    The strategy is to expand the function at the boundary using a Taylor
    expansion. The quickest way to perform this is by using the analytic
    solution and a parloop.

    This is only implemented to recover to the CG1 function space.
    """

    def __init__(self, x_inout, method=BoundaryMethod.extruded, eff_coords=None):
        """
        Args:
            v_CG1 (:class:`Function`): the continuous function after the first
                recovery is performed. Should be in a first-order continuous
                :class:`FunctionSpace`. This is already correct on the interior
                of the domain. It will be returned with corrected values.
            method (:class:`BoundaryMethod`, optional): enumerator specifying
                the method to use. Defaults to `BoundaryMethod.extruded`.
            eff_coords (:class:`Function`, optional): the effective coordinates
                corresponding to the initial recovery process. Should be in the
                :class:`VectorFunctionSpace` corresponding to the space of the
                `v_DG1` variable. This must be provided for the Taylor expansion
                boundary method. Defaults to None.

        Raises:
            ValueError: if the `v_CG1` field is in a space that is not CG1 when
                using the Taylor boundary method.
            ValueError: if the `v_DG1` field is in a space that is not the DG1
                equispaced space when using the Taylor boundary method.
            ValueError: if the effective coordinates are not provided when using
                the Taylor expansion boundary method.
            ValueError: using the extruded or hcurl boundary methods with a
                non-extruded mesh.
        """

        self.x_inout = x_inout
        self.method = method
        self.eff_coords = eff_coords

        V_inout = x_inout.function_space()
        mesh = V_inout.mesh()

        # -------------------------------------------------------------------- #
        # Checks
        # -------------------------------------------------------------------- #
        if self.method == BoundaryMethod.taylor:
            CG1 = FunctionSpace(mesh, "CG", 1)
            if x_inout.function_space() != CG1:
                raise ValueError("This boundary recovery method requires v1 to be in CG1.")
            if eff_coords is None:
                raise ValueError('Need eff_coords field for Taylor expansion boundary method')

        elif self.method in [BoundaryMethod.extruded, BoundaryMethod.hcurl]:
            # check that mesh is valid -- must be an extruded mesh
            if not V_inout.extruded:
                raise ValueError('The extruded boundary method only works on extruded meshes')

        else:
            raise ValueError("Boundary method should be a Boundary Method Enum object.")


        # -------------------------------------------------------------------- #
        # Initalisation for different boundary methods
        # -------------------------------------------------------------------- #

        if self.method == BoundaryMethod.extruded:
            # create field to temporarily hold values
            self.x_tmp = Function(V_inout)
            self.kernel = BoundaryRecoveryExtruded(V_inout)

        elif self.method == BoundaryMethod.hcurl:
            # create field to temporarily hold values
            self.x_tmp = Function(V_inout)
            self.kernel = BoundaryRecoveryHCurl(V_inout)

        elif self.method == BoundaryMethod.taylor:
            # Create DG1 space ----------------------------------------------- #
            if V_inout.extruded:
                cell = mesh._base_mesh.ufl_cell().cellname()
                DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
                DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
                DG1_element = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
            else:
                cell = mesh.ufl_cell().cellname()
                DG1_element = FiniteElement("DG", cell, 1, variant="equispaced")

            vec_DG1 = VectorFunctionSpace(mesh, DG1_element)

            # Create coordinates --------------------------------------------- #
            coords = SpatialCoordinate(mesh)
            self.act_coords = Function(vec_DG1).project(coords)  # actual coordinates
            self.eff_coords = eff_coords  # effective coordinates
            self.on_exterior = find_domain_boundaries(mesh)
            self.num_ext = find_domain_boundaries(mesh)

            # Make operators used in process --------------------------------- #
            V_broken = FunctionSpace(mesh, BrokenElement(V_inout.ufl_element()))
            self.x_DG1_wrong = Function(V_broken)
            self.x_DG1_correct = Function(V_broken)
            self.interpolator = Interpolator(self.x_inout, self.x_DG1_wrong)
            self.averager = Averager(self.x_DG1_correct, self.x_inout)
            self.kernel = BoundaryGaussianElimination(V_broken)


    def apply(self):
        """Applies the boundary recovery process."""
        if self.method == BoundaryMethod.taylor:
            self.interpolator.interpolate()
            self.kernel.apply(self.x_DG1_wrong, self.x_DG1_correct,
                              self.act_coords, self.eff_coords, self.num_ext)
            self.averager.project()

        else:
            self.x_tmp.assign(self.x_inout)
            self.kernel.apply(self.x_inout, self.x_tmp)


class Recoverer(object):
    """
    Recovers a field from a low-order space to a higher-order space.

    An object that 'recovers' a field from a low-order space (e.g. DG0) into a
    higher-order space (e.g. CG1). This first interpolates or projects the field
    into the broken (fully-discontinuous) form of the target higher-order space,
    then uses the :class:`Averager` to restore continuity. This may not be
    accurate at domain boundaries, so if specified, the field will then be
    corrected using the :class:`BoundaryRecoverer`.
    """

    def __init__(self, x_in, x_out, method='interpolate', boundary_method=None):
        """
        Args:
            x_in (:class:`Function`): the field or :class:`ufl.Expr` to project.
                For instance this could be in the DG0 space.
            x_out (:class:`Function`): to field to put the result in. This could
                for instance lie in the CG1 space.
            method (str, optional): method for obtaining the field in the broken
                space. Must be 'interpolate' or 'project'. Defaults to
                'interpolate'.
            boundary_method (:class:`BoundaryMethod`, optional): enumerator
                specifying the boundary method to use. Defaults to None.

        Raises:
            ValueError: the `VDG` argument must be specified if the
                `boundary_method` is not None.
        """

        # check if v_in is valid
        if not isinstance(x_in, (ufl.core.expr.Expr, function.Function)):
            raise ValueError("Can only recover UFL expression or Functions not '%s'" % type(x_in))

        self.x_out = x_out
        V_out = x_out.function_space()
        mesh = V_out.mesh()
        rec_elt = V_out.ufl_element()

        # -------------------------------------------------------------------- #
        # Set up broken space
        # -------------------------------------------------------------------- #
        self.vector_function_space = isinstance(rec_elt, VectorElement)
        if self.vector_function_space:
            # VectorElement has to be on the outside
            # so first need to get underlying finite element
            brok_elt = VectorElement(BrokenElement(rec_elt.sub_elements()[0]))
        else:
            # Otherwise we can immediately get broken element
            brok_elt = BrokenElement(rec_elt)
        V_brok = FunctionSpace(mesh, brok_elt)

        # -------------------------------------------------------------------- #
        # Set up interpolation / projection
        # -------------------------------------------------------------------- #
        x_brok = Function(V_brok)

        self.method = method
        if method == 'interpolate':
            self.broken_op = Interpolator(x_in, x_brok)
        elif method == 'project':
            self.broken_op = Projector(x_in, x_brok)
        else:
            raise ValueError(f'Valid methods are "interpolate" or "project", not {method}')

        self.averager = Averager(x_brok, self.x_out)

        # -------------------------------------------------------------------- #
        # Set up boundary recovery
        # -------------------------------------------------------------------- #
        self.boundary_method = boundary_method

        # check boundary method options are valid
        if boundary_method is not None:
            if boundary_method not in [BoundaryMethod.extruded, BoundaryMethod.taylor, BoundaryMethod.hcurl]:
                raise TypeError("Boundary method must be a BoundaryMethod Enum object.")

            # now specify things that we'll need if we are doing boundary recovery
            if boundary_method == BoundaryMethod.extruded:
                # check dimensions
                if self.vector_function_space:
                    raise ValueError('This method only works for scalar functions.')
                self.boundary_recoverer = BoundaryRecoverer(self.x_out, method=BoundaryMethod.extruded)

            elif boundary_method == BoundaryMethod.hcurl:
                self.boundary_recoverer = BoundaryRecoverer(self.x_out, method=BoundaryMethod.hcurl)

            else:

                eff_coords = find_eff_coords(x_in.function_space())

                # For scalar functions, just set up boundary recoverer and return field into x_brok
                if not self.vector_function_space:
                    self.boundary_recoverer = BoundaryRecoverer(self.x_out,
                                                                method=BoundaryMethod.taylor,
                                                                eff_coords=eff_coords)

                else:
                    # Must set up scalar functions for each component
                    CG1 = FunctionSpace(mesh, "CG", 1)

                    # now, break the problem down into components
                    x_out_scalars = []
                    self.boundary_recoverers = []
                    self.interpolate_to_scalars = []
                    self.extra_averagers = []
                    # the boundary recoverer needs to be done on a scalar fields
                    # so need to extract component and restore it after the boundary recovery is done
                    for i in range(V_out.value_size):
                        x_out_scalars.append(Function(CG1))
                        self.interpolate_to_scalars.append(Interpolator(self.x_out[i], x_out_scalars[i]))
                        self.boundary_recoverers.append(BoundaryRecoverer(x_out_scalars[i],
                                                                          method=BoundaryMethod.taylor,
                                                                          eff_coords=eff_coords[i]))
                    self.interpolate_to_vector = Interpolator(as_vector(x_out_scalars), self.x_out)

    def project(self):
        """Perform the whole recovery step."""

        # Initial averaging step
        self.broken_op.project() if self.method == 'project' else self.broken_op.interpolate()
        self.averager.project()

        # Boundary recovery
        if self.boundary_method is not None:
            # For vector elements, treat each component separately
            if self.vector_function_space:
                for (interpolate_to_scalar, boundary_recoverer) \
                        in zip(self.interpolate_to_scalars, self.boundary_recoverers):
                    interpolate_to_scalar.interpolate()
                    # Correct at boundaries
                    boundary_recoverer.apply()
                # Combine the components to obtain the vector field
                self.interpolate_to_vector.interpolate()
            else:
                # Extrapolate at boundaries
                self.boundary_recoverer.apply()

        return self.x_out


def find_eff_coords(V0):
    """
    Find the effective coordinates corresponding to a recovery process.

    Takes a function in a space `V0` and returns the effective coordinates,
    in an equispaced vector DG1 space, of a recovery into a CG1 field. This is
    for use with the :class:`BoundaryRecoverer`, as it facilitates the Gaussian
    elimination used to get second-order recovery at boundaries. If `V0` is a
    vector function space, this returns an array of coordinates for each
    component. Geocentric Cartesian coordinates are returned.

    Args:
        V0 (:class:`FunctionSpace`): the function space of the original field
            before the recovery process.
    """

    mesh = V0.mesh()
    if V0.extruded:
        cell = mesh._base_mesh.ufl_cell().cellname()
        DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
        DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
        DG1_element = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
    else:
        cell = mesh.ufl_cell().cellname()
        DG1_element = FiniteElement("DG", cell, 1, variant="equispaced")

    vec_CG1 = VectorFunctionSpace(mesh, "CG", 1)
    vec_DG1 = VectorFunctionSpace(mesh, DG1_element)
    x = SpatialCoordinate(mesh)

    if V0.ufl_element().value_size() > 1:
        eff_coords_list = []
        V0_coords_list = []

        # treat this separately for each component
        for i in range(V0.ufl_element().value_size()):
            # fill an d-dimensional list with i-th coordinate
            x_list = [x[i] for j in range(V0.ufl_element().value_size())]

            # the i-th element in V0_coords_list is a vector with all components the i-th coord
            ith_V0_coords = Function(V0).project(as_vector(x_list))
            V0_coords_list.append(ith_V0_coords)

        for i in range(V0.ufl_element().value_size()):
            # slice through V0_coords_list to obtain the coords of the DOFs for that component
            x_list = [V0_coords[i] for V0_coords in V0_coords_list]

            # average these to find effective coords in CG1
            V0_coords_in_DG1 = Function(vec_DG1).interpolate(as_vector(x_list))
            eff_coords_in_CG1 = Function(vec_CG1)
            eff_coords_averager = Averager(V0_coords_in_DG1, eff_coords_in_CG1)
            eff_coords_averager.project()

            # obtain these in DG1
            eff_coords_in_DG1 = Function(vec_DG1).interpolate(eff_coords_in_CG1)
            eff_coords_list.append(correct_eff_coords(eff_coords_in_DG1))

        return eff_coords_list

    else:
        # find the coordinates at DOFs in V0
        vec_V0 = VectorFunctionSpace(mesh, V0.ufl_element())
        V0_coords = Function(vec_V0).project(x)

        # average these to find effective coords in CG1
        V0_coords_in_DG1 = Function(vec_DG1).interpolate(V0_coords)
        eff_coords_in_CG1 = Function(vec_CG1)
        eff_coords_averager = Averager(V0_coords_in_DG1, eff_coords_in_CG1)
        eff_coords_averager.project()

        # obtain these in DG1
        eff_coords_in_DG1 = Function(vec_DG1).interpolate(eff_coords_in_CG1)

        return correct_eff_coords(eff_coords_in_DG1)


def correct_eff_coords(eff_coords):
    """
    Corrects the effective coordinates.

    This corrects the effective coordinates that have been calculated by simply
    averaging, as they may which will not be correct for periodic meshes.

    Args:
        eff_coords (:class:`Function`): the effective coordinates field in
            the vector equispaced DG1 :class:`FunctionSpace`.
    """

    mesh = eff_coords.function_space().mesh()
    vec_CG1 = VectorFunctionSpace(mesh, "CG", 1)

    if vec_CG1.extruded:
        cell = mesh._base_mesh.ufl_cell().cellname()
        DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
        DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
        DG1_element = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
    else:
        cell = mesh.ufl_cell().cellname()
        DG1_element = FiniteElement("DG", cell, 1, variant="equispaced")

    vec_DG1 = VectorFunctionSpace(mesh, DG1_element)

    x = SpatialCoordinate(mesh)

    if eff_coords.function_space() != vec_DG1:
        raise ValueError('eff_coords needs to be in the vector DG1 space')

    # obtain different coords in DG1
    DG1_coords = Function(vec_DG1).interpolate(x)
    CG1_coords_from_DG1 = Function(vec_CG1)
    averager = Averager(DG1_coords, CG1_coords_from_DG1)
    averager.project()
    DG1_coords_from_averaged_CG1 = Function(vec_DG1).interpolate(CG1_coords_from_DG1)
    DG1_coords_diff = Function(vec_DG1).interpolate(DG1_coords - DG1_coords_from_averaged_CG1)

    # interpolate coordinates, adjusting those different coordinates
    adjusted_coords = Function(vec_DG1)
    adjusted_coords.interpolate(eff_coords + DG1_coords_diff)

    return adjusted_coords


def find_domain_boundaries(mesh):
    """
    Find the cells on the domain boundaries.

    Makes a field in the scalar DG0 :class:`FunctionSpace`, whose values are 0
    everywhere except for in cells on the boundary of the domain, where the
    values are 1.0. This allows boundary cells to be identified.

    Args:
        mesh (:class:`Mesh`): the mesh.
    """

    DG0 = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)

    on_exterior_DG0 = Function(DG0)
    on_exterior_CG1 = Function(CG1)

    # we get values in CG1 initially as DG0 will not work for triangular elements
    bc_codes = ['on_boundary', 'top', 'bottom']
    bcs = [DirichletBC(CG1, Constant(1.0), bc_code) for bc_code in bc_codes]

    for bc in bcs:
        try:
            bc.apply(on_exterior_CG1)
        except ValueError:
            pass

    on_exterior_DG0.interpolate(on_exterior_CG1)

    return on_exterior_DG0
