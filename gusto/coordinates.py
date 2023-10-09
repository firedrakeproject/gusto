"""
This file provides a coordinate object, dependent on the mesh.
Coordinate fields are stored in specified VectorFunctionSpaces.
"""

from gusto.coord_transforms import lonlatr_from_xyz, rotated_lonlatr_coords
from gusto.logging import logger
from firedrake import SpatialCoordinate, Function
import numpy as np


class Coordinates(object):
    """
    An object for holding and setting up coordinate fields.
    """
    def __init__(self, mesh, on_sphere=False, rotated_pole=None, radius=None):
        """
        Args:
            mesh (:class:`Mesh`): the model's domain object.
            on_sphere (bool, optional): whether the domain is on the surface of
                a sphere. If False, the domain is assumed to be Cartesian.
                Defaults to False.
            rotated_pole (tuple, optional): a tuple of floats (lon, lat) of the
                location to use as the north pole in a spherical coordinate
                system. These are expressed in the original coordinate system.
                The longitude and latitude must be expressed in radians.
                Defaults to None. This is unused for non-spherical domains.
            radius (float, optional): the radius of a spherical domain. Defaults
                to None. This is unused for non-spherical domains.
        """

        self.mesh = mesh

        # -------------------------------------------------------------------- #
        # Set up spatial coordinate
        # -------------------------------------------------------------------- #

        if on_sphere:
            xyz = SpatialCoordinate(mesh)
            if rotated_pole is not None:
                lon, lat, r = rotated_lonlatr_coords(xyz, rotated_pole)
            else:
                lon, lat, r = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])

            if mesh.extruded:
                self.coords = (lon, lat, r-radius)
                self.coords_name = ['lon', 'lat', 'h']
            else:
                self.coords = (lon, lat)
                self.coords_name = ['lon', 'lat']
        else:
            self.coords = SpatialCoordinate(mesh)
            if mesh.geometric_dimension() == 1:
                self.coords_name = ['x']
            elif mesh.geometric_dimension() == 2 and mesh.extruded:
                self.coords_name = ['x', 'z']
            elif mesh.geometric_dimension() == 2:
                self.coords_name = ['x', 'y']
            elif mesh.geometric_dimension() == 3:
                self.coords_name = ['x', 'y', 'z']
            else:
                raise ValueError('Cannot work out coordinates of domain')

        # -------------------------------------------------------------------- #
        # Store chi field
        # -------------------------------------------------------------------- #

        self.chi_coords = {}           # Dict of natural coords by space
        self.global_chi_coords = {}    # Dict of whole coords stored on first proc
        self.parallel_array_lims = {}  # Dict of array lengths for each proc

    def register_space(self, domain, space_name):
        """
        Computes the coordinate fields at the DoFs of a function space, which
        are subsequently used for outputting.

        As proper parallel outputting is not yet implemented, the array of
        coordinate data is entirely broadcast to the first processor.

        Args:
            space_name (str): the name of the function space to be registered
                with the :class:`Coordinates` object.

        Raises:
            NotImplementedError: only scalar-valued spaces are implemented.
        """

        comm = self.mesh.comm
        comm_size = comm.Get_size()
        my_rank = comm.Get_rank()
        topological_dimension = self.mesh.topological_dimension()

        if space_name in self.chi_coords.keys():
            logger.warning(f'Coords for {space_name} space have already been computed')
            return None

        # Loop through spaces
        space = domain.spaces(space_name)

        # Use the appropriate scalar function space if the space is vector
        if np.prod(space.ufl_element().value_shape()) > 1:
            # TODO: get scalar space, and only compute coordinates if necessary
            logger.warning(f'Space {space_name} has more than one dimension, '
                           + 'and coordinates used for netCDF output have not '
                           + 'yet been implemented for this.')
            return None

        self.chi_coords[space_name] = []

        # Now set up
        for i in range(topological_dimension):
            self.chi_coords[space_name].append(Function(space).interpolate(self.coords[i]))

        # -------------------------------------------------------------------- #
        # Code for settings up coordinates for parallel-serial IO
        # -------------------------------------------------------------------- #

        len_coords = space.dim()
        my_num_dofs = len(self.chi_coords[space_name][0].dat.data_ro[:])

        if my_rank != 0:
            # Do not make full coordinate array
            self.global_chi_coords[space_name] = None
            self.parallel_array_lims[space_name] = None
            # Find number of DoFs on this processor
            comm.send(my_num_dofs, dest=0)
        else:
            # First processor has one large array of the global chi data
            self.global_chi_coords[space_name] = np.zeros((topological_dimension, len_coords))
            # Store the limits inside this array telling us how data is partitioned
            self.parallel_array_lims[space_name] = np.zeros((comm_size, 2), dtype=int)
            # First processor has the first bit of data
            self.parallel_array_lims[space_name][my_rank][0] = 0
            self.parallel_array_lims[space_name][my_rank][1] = my_num_dofs - 1
            # Receive number of DoFs on other processors
            for procid in range(1, comm_size):
                other_num_dofs = comm.recv(source=procid)
                self.parallel_array_lims[space_name][procid][0] = self.parallel_array_lims[space_name][procid-1][1] + 1
                self.parallel_array_lims[space_name][procid][1] = self.parallel_array_lims[space_name][procid][0] + other_num_dofs - 1

        # Now move coordinates to first processor
        for i in range(topological_dimension):
            if my_rank != 0:
                # Send information to rank 0
                my_tag = comm_size*i + my_rank
                comm.send(self.chi_coords[space_name][i].dat.data_ro[:], dest=0, tag=my_tag)
            else:
                # Rank 0 -- the receiver
                (low_lim, up_lim) = self.parallel_array_lims[space_name][my_rank][:]
                self.global_chi_coords[space_name][i][low_lim:up_lim+1] = self.chi_coords[space_name][i].dat.data_ro[:]
                # Receive coords from each processor and put them into array
                for procid in range(1, comm_size):
                    my_tag = comm_size*i + procid
                    new_coords = comm.recv(source=procid, tag=my_tag)
                    (low_lim, up_lim) = self.parallel_array_lims[space_name][procid][:]
                    self.global_chi_coords[space_name][i, low_lim:up_lim+1] = new_coords
