"""
The Domain object that is provided in this module contains the model's mesh and
the set of compatible function spaces defined upon it. It also contains the
model's time interval.
"""

from gusto.core.coordinates import Coordinates
from gusto.core.function_spaces import Spaces, check_degree_args
from firedrake import (Constant, SpatialCoordinate, sqrt, CellNormal, cross,
                       inner, grad, VectorFunctionSpace, Function, FunctionSpace,
                       perp)
import numpy as np


class Domain(object):
    """
    The Domain holds the model's mesh and its compatible function spaces.

    The compatible function spaces are given by the de Rham complex, and are
    specified here through the family of the HDiv velocity space and the degree
    of the DG space.

    For extruded meshes, it is possible to seperately specify the horizontal and
    vertical degrees of the elements. Alternatively, if these degrees should be
    the same then this can be specified through the "degree" argument.
    """
    def __init__(self, mesh, dt, family, degree=None,
                 horizontal_degree=None, vertical_degree=None,
                 rotated_pole=None, max_quad_degree=None):
        """
        Args:
            mesh (:class:`Mesh`): the model's mesh.
            dt (:class:`Constant`): the time taken to perform a single model
                step. If a float or int is passed, it will be cast to a
                :class:`Constant`.
            family (str): the finite element space family used for the velocity
                field. This determines the other finite element spaces used via
                the de Rham complex.
            degree (int, optional): the element degree used for the DG space
                Defaults to None, in which case the horizontal degree must be provided.
            horizontal_degree (int, optional): the element degree used for the
                horizontal part of the DG space. Defaults to None.
            vertical_degree (int, optional): the element degree used for the
                vertical part of the DG space. Defaults to None.
            rotated_pole (tuple, optional): a tuple of floats (lon, lat) of the
                location to use as the north pole in a spherical coordinate
                system. These are expressed in the original coordinate system.
                The longitude and latitude must be expressed in radians.
                Defaults to None. This is unused for non-spherical domains.
            max_quad_degree (int, optional): the maximum quadrature degree to
                use in certain non-linear terms (e.g. when using an expression
                for the Exner pressure). Defaults to None, in which case this
                will be set to the 2*p+3, where p is the maximum polynomial
                degree for the DG space.

        Raises:
            ValueError: if incompatible degrees are specified (e.g. specifying
                both "degree" and "horizontal_degree").
        """

        # -------------------------------------------------------------------- #
        # Time step
        # -------------------------------------------------------------------- #

        # Store central dt for use in the rest of the model
        if type(dt) is Constant:
            self.dt = dt
        elif type(dt) in (float, int):
            self.dt = Constant(dt)
        else:
            raise TypeError(f'dt must be a Constant, float or int, not {type(dt)}')

        # Make a placeholder for the time
        self.t = Constant(0.0)

        # -------------------------------------------------------------------- #
        # Build compatible function spaces
        # -------------------------------------------------------------------- #

        check_degree_args('Domain', mesh, degree, horizontal_degree, vertical_degree)

        # Get degrees
        self.horizontal_degree = degree if horizontal_degree is None else horizontal_degree
        self.vertical_degree = degree if vertical_degree is None else vertical_degree

        if max_quad_degree is None:
            max_degree = max(self.horizontal_degree, self.vertical_degree)
            self.max_quad_degree = 2*max_degree + 3
        else:
            self.max_quad_degree = max_quad_degree

        self.mesh = mesh
        self.family = family
        self.spaces = Spaces(mesh)
        self.spaces.build_compatible_spaces(self.family, self.horizontal_degree,
                                            self.vertical_degree)
        self.spaces.build_dg1_equispaced()

        # -------------------------------------------------------------------- #
        # Determine some useful aspects of domain
        # -------------------------------------------------------------------- #

        # Figure out if we're on a sphere
        # WARNING: if we ever wanted to run on other domains (e.g. circle, disk
        # or torus) then the identification of domains would no longer be unique
        if hasattr(mesh, "_base_mesh") and hasattr(mesh._base_mesh, 'geometric_dimension'):
            self.on_sphere = (mesh._base_mesh.geometric_dimension() == 3 and mesh._base_mesh.topological_dimension() == 2)
        else:
            self.on_sphere = (mesh.geometric_dimension() == 3 and mesh.topological_dimension() == 2)

        #  build the vertical normal and define perp for 2d geometries
        dim = mesh.topological_dimension()
        if self.on_sphere:
            x = SpatialCoordinate(mesh)
            R = sqrt(inner(x, x))
            self.k = grad(R)
            if dim == 2:
                if hasattr(mesh, "_base_mesh"):
                    sphere_degree = mesh._base_mesh.coordinates.function_space().ufl_element().degree()
                else:
                    if not hasattr(mesh, "_cell_orientations"):
                        mesh.init_cell_orientations(x)
                    sphere_degree = mesh.coordinates.function_space().ufl_element().degree()
                V = VectorFunctionSpace(mesh, "DG", sphere_degree)
                self.outward_normals = Function(V).interpolate(CellNormal(mesh))
                self.perp = lambda u: cross(self.outward_normals, u)
        else:
            kvec = [0.0]*dim
            kvec[dim-1] = 1.0
            self.k = Constant(kvec)
            if dim == 2:
                self.perp = perp

        # -------------------------------------------------------------------- #
        # Construct information relating to height/radius
        # -------------------------------------------------------------------- #

        if self.on_sphere:
            spherical_shell_mesh = mesh._base_mesh if hasattr(mesh, "_base_mesh") else mesh
            xyz_shell = SpatialCoordinate(spherical_shell_mesh)
            r_shell = sqrt(inner(xyz_shell, xyz_shell))
            CG1 = FunctionSpace(spherical_shell_mesh, "CG", 1)
            radius_field = Function(CG1)
            radius_field.interpolate(r_shell)
            # TODO: this should use global min kernel
            radius = Constant(np.min(radius_field.dat.data_ro))
        else:
            radius = None

        # -------------------------------------------------------------------- #
        # Set up coordinates
        # -------------------------------------------------------------------- #

        self.coords = Coordinates(mesh, on_sphere=self.on_sphere,
                                  rotated_pole=rotated_pole, radius=radius)
        # Set up DG1 equispaced space, used for making metadata
        _ = self.spaces('DG1_equispaced')
        self.coords.register_space(self, 'DG1_equispaced')

        # Set height above surface (requires coordinates)
        if hasattr(mesh, "_base_mesh"):
            self.set_height_above_surface()

        # -------------------------------------------------------------------- #
        # Construct metadata about domain
        # -------------------------------------------------------------------- #

        self.metadata = construct_domain_metadata(mesh, self.coords, self.on_sphere)

    def set_height_above_surface(self):
        """
        Sets a coordinate field which corresponds to height above the domain's
        surface.
        """

        from firedrake import dot

        x = SpatialCoordinate(self.mesh)

        # Make a height field in CG1
        CG1 = FunctionSpace(self.mesh, "CG", 1, name='CG1')
        self.spaces.add_space('CG1', CG1)
        self.coords.register_space(self, 'CG1')
        CG1_height = Function(CG1)
        CG1_height.interpolate(dot(self.k, x))
        height_above_surface = Function(CG1)

        # Turn height into columnwise data
        columnwise_height, index_data = self.coords.get_column_data(CG1_height, self)

        # Find minimum height in each column
        surface_height_1d = np.min(columnwise_height, axis=1)
        height_above_surface_data = columnwise_height - surface_height_1d[:, None]

        self.coords.set_field_from_column_data(height_above_surface,
                                               height_above_surface_data,
                                               index_data)

        self.height_above_surface = height_above_surface


def construct_domain_metadata(mesh, coords, on_sphere):
    """
    Builds a dictionary containing metadata about the domain.

    Args:
        mesh (:class:`Mesh`): the model's mesh.
        coords (:class:`Coordinates`): the model's coordinate object.
        on_sphere (bool): whether the domain is on the sphere or not.

    Returns:
        dict: a dictionary of metadata relating to the domain.
    """
    metadata = {}
    metadata['extruded'] = mesh.extruded

    if on_sphere and hasattr(mesh, "_base_mesh"):
        metadata['domain_type'] = 'extruded_spherical_shell'
    elif on_sphere:
        metadata['domain_type'] = 'spherical_shell'
    elif mesh.geometric_dimension() == 1 and mesh.topological_dimension() == 1:
        metadata['domain_type'] = 'interval'
    elif mesh.geometric_dimension() == 2 and mesh.topological_dimension() == 2 and mesh.extruded:
        metadata['domain_type'] = 'vertical_slice'
    elif mesh.geometric_dimension() == 2 and mesh.topological_dimension() == 2:
        metadata['domain_type'] = 'plane'
    elif mesh.geometric_dimension() == 3 and mesh.topological_dimension() == 3 and mesh.extruded:
        metadata['domain_type'] = 'extruded_plane'
    else:
        raise ValueError('Unable to determine domain type')

    comm = mesh.comm
    my_rank = comm.Get_rank()

    # Properties of domain will be determined from full coords, so need
    # doing on the first processor then broadcasting to others

    if my_rank == 0:
        chi = coords.global_chi_coords['DG1_equispaced']
        if not on_sphere:
            metadata['domain_extent_x'] = np.max(chi[0, :]) - np.min(chi[0, :])
            if metadata['domain_type'] in ['plane', 'extruded_plane']:
                metadata['domain_extent_y'] = np.max(chi[1, :]) - np.min(chi[1, :])
        if mesh.extruded:
            metadata['domain_extent_z'] = np.max(chi[-1, :]) - np.min(chi[-1, :])

    else:
        metadata = {}

    # Send information to other processors
    metadata = comm.bcast(metadata, root=0)

    return metadata
