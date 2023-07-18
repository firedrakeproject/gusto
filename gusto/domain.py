"""
The Domain object that is provided in this module contains the model's mesh and
the set of compatible function spaces defined upon it. It also contains the
model's time interval.
"""

from gusto.coordinates import Coordinates
from gusto.function_spaces import Spaces, check_degree_args
from firedrake import (Constant, SpatialCoordinate, sqrt, CellNormal, cross,
                       as_vector, inner, interpolate, VectorFunctionSpace,
                       Function)
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
                 move_mesh=False):
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

        # -------------------------------------------------------------------- #
        # Are we moving the mesh?
        self.move_mesh = move_mesh
        # -------------------------------------------------------------------- #

        # -------------------------------------------------------------------- #
        # Build compatible function spaces
        # -------------------------------------------------------------------- #

        check_degree_args('Domain', mesh, degree, horizontal_degree, vertical_degree)

        # Get degrees
        self.horizontal_degree = degree if horizontal_degree is None else horizontal_degree
        self.vertical_degree = degree if vertical_degree is None else vertical_degree

        self.mesh = mesh
        self.family = family
        self.spaces = Spaces(mesh)
        # Build and store compatible spaces
        self.compatible_spaces = [space for space in self.spaces.build_compatible_spaces(self.family, self.horizontal_degree, self.vertical_degree)]

        # -------------------------------------------------------------------- #
        # Determine some useful aspects of domain
        # -------------------------------------------------------------------- #

        # Figure out if we're on a sphere
        # TODO: could we run on other domains that could confuse this?
        # TODO: could this be combined with domain metadata below?
        if hasattr(mesh, "_base_mesh") and hasattr(mesh._base_mesh, 'geometric_dimension'):
            self.on_sphere = (mesh._base_mesh.geometric_dimension() == 3 and mesh._base_mesh.topological_dimension() == 2)
        else:
            self.on_sphere = (mesh.geometric_dimension() == 3 and mesh.topological_dimension() == 2)

        #  build the vertical normal and define perp for 2d geometries
        dim = mesh.topological_dimension()
        if self.on_sphere:
            x = SpatialCoordinate(mesh)
            R = sqrt(inner(x, x))
            self.k = interpolate(x/R, mesh.coordinates.function_space())
            if dim == 2:
                if hasattr(mesh, "_bash_mesh"):
                    sphere_degree = mesh._base_mesh.coordinates.function_space().ufl_element().degree()
                else:
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
                self.perp = lambda u: as_vector([-u[1], u[0]])

        # -------------------------------------------------------------------- #
        # Set up coordinates
        # -------------------------------------------------------------------- #

        self.coords = Coordinates(mesh)
        # Set up DG1 equispaced space, used for making metadata
        _ = self.spaces('DG1_equispaced')
        self.coords.register_space(self, 'DG1_equispaced')

        # -------------------------------------------------------------------- #
        # Construct metadata about domain
        # -------------------------------------------------------------------- #

        # TODO: would this be better as an object?
        self.metadata = {}
        self.metadata['extruded'] = mesh.extruded

        if self.on_sphere and hasattr(mesh, "_base_mesh"):
            self.metadata['domain_type'] = 'extruded_spherical_shell'
        elif self.on_sphere:
            self.metadata['domain_type'] = 'spherical_shell'
        elif mesh.geometric_dimension() == 1 and mesh.topological_dimension() == 1:
            self.metadata['domain_type'] = 'interval'
        elif mesh.geometric_dimension() == 2 and mesh.topological_dimension() == 2 and mesh.extruded:
            self.metadata['domain_type'] = 'vertical_slice'
        elif mesh.geometric_dimension() == 2 and mesh.topological_dimension() == 2:
            self.metadata['domain_type'] = 'plane'
        elif mesh.geometric_dimension() == 3 and mesh.topological_dimension() == 3 and mesh.extruded:
            self.metadata['domain_type'] = 'extruded_plane'
        else:
            raise ValueError('Unable to determine domain type')

        comm = self.mesh.comm
        comm_size = comm.Get_size()
        my_rank = comm.Get_rank()
        max_num_domain_infos = 3

        # Properties of domain will be determined from full coords, so need
        # doing on the first processor then broadcasting to others

        if my_rank == 0:
            chi = self.coords.global_chi_coords['DG1_equispaced']
            if not self.on_sphere:
                self.metadata['domain_extent_x'] = np.max(chi[0, :]) - np.min(chi[0, :])
                if self.metadata['domain_type'] in ['plane', 'extruded_plane']:
                    self.metadata['domain_extent_y'] = np.max(chi[1, :]) - np.min(chi[1, :])
            if mesh.extruded:
                self.metadata['domain_extent_z'] = np.max(chi[-1, :]) - np.min(chi[-1, :])

            # Send information to other processors
            for j, metadata_key in enumerate([f'domain_extent_{xyz}' for xyz in ['x', 'y', 'z']]):
                if metadata_key in self.metadata.keys():
                    metadata_value = self.metadata[metadata_key]
                else:
                    metadata_value = None
                for procid in range(1, comm_size):
                    my_tag = comm_size*j + procid
                    comm.send((metadata_key, metadata_value), dest=procid, tag=my_tag)
        else:
            # Need to receive information and store in metadata
            for j in range(max_num_domain_infos):
                my_tag = comm_size*j + my_rank
                metadata_key, metadata_value = comm.recv(source=0, tag=my_tag)
                if metadata_value is not None:
                    self.metadata[metadata_key] = metadata_value
