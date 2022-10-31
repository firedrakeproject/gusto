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
                       VectorFunctionSpace, as_vector, function, interval)
from firedrake.utils import cached_property

from gusto import kernels

__all__ = ["Averager", "Boundary_Method", "Boundary_Recoverer", "Recoverer"]


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

        if not isinstance(v, (ufl.core.expr.Expr, function.Function)):
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

        self.average_kernel = kernels.Average(self.V)

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


class Boundary_Method(Enum):
    """
    Method for correcting the recovery at the domain boundaries.

    An enumerator object encoding methods for correcting boundary recovery:
    dynamics: which corrects a field recovered into CG1.
    physics: corrects a field recovered into the lowest-order temperature space.
    """

    dynamics = 0
    physics = 1


class Boundary_Recoverer(object):
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

    def __init__(self, v_CG1, v_DG1, method=Boundary_Method.physics, eff_coords=None):
        """
        Args:
            v_CG1 (:class:`Function`): the continuous function after the first
                recovery is performed. Should be in a first-order continuous
                :class:`FunctionSpace`. This is already correct on the interior
                of the domain.
            v_DG1 (:class:`Function`): the function to be output. Should be in a
                discontinuous first-order :class:`FunctionSpace`.
            method (:class:`Boundary_Method`, optional): enumerator specifying
                the method to use. Defaults to `Boundary_Method.physics`.
            eff_coords (:class:`Function`, optional): the effective coordinates
                corresponding to the initial recovery process. Should be in the
                :class:`VectorFunctionSpace` corresponding to the space of the
                `v_DG1` variable. This must be provided for the dynamics
                boundary method. Defaults to None.

        Raises:
            ValueError: if the `v_CG1` field is in a space that is not CG1 when
                using the dynamics boundary method.
            ValueError: if the `v_DG1` field is in a space that is not the DG1
                equispaced space when using the dynamics boundary method.
            ValueError: if the effective coordinates are not provided when using
                the dynamics boundary method.
            ValueError: using the physics boundary method with a non-extruded
                mesh.
            ValueError: using the physics boundary method `v_CG1` in the
                DG0 x CG1 tensor product space.
            ValueError: using the physics boundary method `v_DG1` in the
                DG0 x DG1 tensor product space.
        """

        self.v_DG1 = v_DG1
        self.v_CG1 = v_CG1
        self.v_DG1_old = Function(v_DG1.function_space())
        self.eff_coords = eff_coords

        self.method = method
        mesh = v_CG1.function_space().mesh()
        DG0 = FunctionSpace(mesh, "DG", 0)
        CG1 = FunctionSpace(mesh, "CG", 1)

        if DG0.extruded:
            cell = mesh._base_mesh.ufl_cell().cellname()
            DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
            DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
            DG1_element = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
        else:
            cell = mesh.ufl_cell().cellname()
            DG1_element = FiniteElement("DG", cell, 1, variant="equispaced")
        DG1 = FunctionSpace(mesh, DG1_element)

        self.num_ext = find_domain_boundaries(mesh)

        # check function spaces of functions
        if self.method == Boundary_Method.dynamics:
            if v_CG1.function_space() != CG1:
                raise ValueError("This boundary recovery method requires v1 to be in CG1.")
            if v_DG1.function_space() != DG1:
                raise ValueError("This boundary recovery method requires v_out to be in DG1.")
            if eff_coords is None:
                raise ValueError('Need eff_coords field for dynamics boundary methods')

        elif self.method == Boundary_Method.physics:
            # check that mesh is valid -- must be an extruded mesh
            if not DG0.extruded:
                raise ValueError('The physics boundary method only works on extruded meshes')
            # check that function spaces are valid
            sub_elements = v_CG1.function_space().ufl_element().sub_elements()
            if (sub_elements[0].family() not in ['Discontinuous Lagrange', 'DQ']
                    or sub_elements[1].family() != 'Lagrange'
                    or v_CG1.function_space().ufl_element().degree() != (0, 1)):
                raise ValueError("This boundary recovery method requires v_CG1 to be in DG0xCG1 TensorProductSpace.")

            brok_elt = v_DG1.function_space().ufl_element()
            if (brok_elt.degree() != (0, 1)
                or (type(brok_elt) is not BrokenElement
                    and (brok_elt.sub_elements[0].family() not in ['Discontinuous Lagrange', 'DQ']
                         or brok_elt.sub_elements[1].family() != 'Discontinuous Lagrange'))):
                raise ValueError("This boundary recovery method requires v_DG1 to be in the broken DG0xCG1 TensorProductSpace.")
        else:
            raise ValueError("Boundary method should be a Boundary Method Enum object.")

        vec_DG1 = VectorFunctionSpace(DG0.mesh(), DG1_element)
        x = SpatialCoordinate(DG0.mesh())
        self.interpolator = Interpolator(self.v_CG1, self.v_DG1)

        if self.method == Boundary_Method.dynamics:

            # STRATEGY
            # obtain a coordinate field for all the nodes
            self.act_coords = Function(vec_DG1).project(x)  # actual coordinates
            self.eff_coords = eff_coords  # effective coordinates
            self.output = Function(DG1)
            self.on_exterior = find_domain_boundaries(mesh)

            self.gaussian_elimination_kernel = kernels.GaussianElimination(DG1)

        elif self.method == Boundary_Method.physics:

            self.bottom_kernel = kernels.PhysicsRecoveryBottom()
            self.top_kernel = kernels.PhysicsRecoveryTop()

    def apply(self):
        """Applies the boundary recovery process."""
        self.interpolator.interpolate()
        if self.method == Boundary_Method.physics:
            self.bottom_kernel.apply(self.v_DG1, self.v_CG1)
            self.top_kernel.apply(self.v_DG1, self.v_CG1)

        else:
            self.v_DG1_old.assign(self.v_DG1)
            self.gaussian_elimination_kernel.apply(self.v_DG1_old,
                                                   self.v_DG1,
                                                   self.act_coords,
                                                   self.eff_coords,
                                                   self.num_ext)


class Recoverer(object):
    """
    Recovers a field from a low-order space to a higher-order space.

    An object that 'recovers' a field from a low-order space (e.g. DG0) into a
    higher-order space (e.g. CG1). This encompasses the process of interpolating
    first to a the right space before using the :class:`Averager` object, and
    if specified this also coordinates the boundary recovery process.
    """

    def __init__(self, v_in, v_out, VDG=None, boundary_method=None):
        """
        Args:
            v_in (:class:`Function`): the field or :class:`ufl.Expr` to project.
                For instance this could be in the DG0 space.
            v_out (:class:`Function`): to field to put the result in. This could
                for instance lie in the CG1 space.
            VDG (:class:`FunctionSpace`, optional): if specified, `v_in` is
                interpolated to this space first before the recovery happens.
                Defaults to None.
            boundary_method (:class:`Boundary_Method`, optional): enumerator
                specifying the boundary method to use. Defaults to None.

        Raises:
            ValueError: the `VDG` argument must be specified if the
                `boundary_method` is not None.
        """

        # check if v_in is valid
        if not isinstance(v_in, (ufl.core.expr.Expr, function.Function)):
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
            if boundary_method != Boundary_Method.dynamics and boundary_method != Boundary_Method.physics:
                raise ValueError("Boundary method must be a Boundary_Method Enum object.")
            if VDG is None:
                raise ValueError("If boundary_method is specified, VDG also needs specifying.")

            # now specify things that we'll need if we are doing boundary recovery
            if boundary_method == Boundary_Method.physics:
                # check dimensions
                if self.V.value_size != 1:
                    raise ValueError('This method only works for scalar functions.')
                self.boundary_recoverer = Boundary_Recoverer(self.v_out, self.v, method=Boundary_Method.physics)
            else:

                mesh = self.V.mesh()
                # this ensures we get the pure function space, not an indexed function space
                V0 = FunctionSpace(mesh, self.v_in.function_space().ufl_element())
                CG1 = FunctionSpace(mesh, "CG", 1)
                eff_coords = find_eff_coords(V0)

                if V0.extruded:
                    cell = mesh._base_mesh.ufl_cell().cellname()
                    DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
                    DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
                    DG1_element = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
                else:
                    cell = mesh.ufl_cell().cellname()
                    DG1_element = FiniteElement("DG", cell, 1, variant="equispaced")
                DG1 = FunctionSpace(mesh, DG1_element)

                if self.V.value_size == 1:

                    self.boundary_recoverer = Boundary_Recoverer(self.v_out, self.v,
                                                                 method=Boundary_Method.dynamics,
                                                                 eff_coords=eff_coords)
                else:

                    # now, break the problem down into components
                    v_scalars = []
                    v_out_scalars = []
                    self.boundary_recoverers = []
                    self.project_to_scalars_CG = []
                    self.extra_averagers = []
                    for i in range(self.V.value_size):
                        v_scalars.append(Function(DG1))
                        v_out_scalars.append(Function(CG1))
                        self.project_to_scalars_CG.append(Projector(self.v_out[i], v_out_scalars[i]))
                        self.boundary_recoverers.append(Boundary_Recoverer(v_out_scalars[i], v_scalars[i],
                                                                           method=Boundary_Method.dynamics,
                                                                           eff_coords=eff_coords[i]))
                        # need an extra averager that works on the scalar fields rather than the vector one
                        self.extra_averagers.append(Averager(v_scalars[i], v_out_scalars[i]))

                    # the boundary recoverer needs to be done on a scalar fields
                    # so need to extract component and restore it after the boundary recovery is done
                    self.interpolate_to_vector = Interpolator(as_vector(v_out_scalars), self.v_out)

    def project(self):
        """Perform the fully specified recovery."""

        if self.interpolator is not None:
            self.interpolator.interpolate()
        self.averager.project()
        if self.boundary_method is not None:
            if self.V.value_size > 1:
                for i in range(self.V.value_size):
                    self.project_to_scalars_CG[i].project()
                    self.boundary_recoverers[i].apply()
                    self.extra_averagers[i].project()
                self.interpolate_to_vector.interpolate()
            else:
                self.boundary_recoverer.apply()
                self.averager.project()
        return self.v_out


def find_eff_coords(V0):
    """
    Find the effective coordinates corresponding to a recovery process.

    Takes a function in a space `V0` and returns the effective coordinates,
    in an equispaced vector DG1 space, of a recovery into a CG1 field. This is
    for use with the :class:`Boundary_Recoverer`, as it facilitates the Gaussian
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
