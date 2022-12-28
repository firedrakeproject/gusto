"""
This file provides a coordinate object, dependent on the domain and mesh.
Coordinate fields are stored in specified VectorFunctionSpaces.
"""

from firedrake import (SpatialCoordinate, Constant, as_vector, atan_2, asin,
                       Function, VectorElement, TensorElement, sqrt)
import numpy as np

class Coordinates(object):
    """
    An object for holding and setting up coordinate fields.
    """
    def __init__(self, domain, space_names):
        """
        Args:
            domain (:class:`Domain`): the model's domain object.
            space_names (iter): an iterable containing strings of the names of
                function spaces to create coordinates for.
        """

        mesh = domain.mesh
        self.mesh = mesh
        self.geometric_dimension = mesh.geometric_dimension()
        self.topological_dimension = mesh.topological_dimension()

        #----------------------------------------------------------------------#
        # Set up spatial coordinate
        #----------------------------------------------------------------------#

        # Add cartesian coordinates for whatever the domain
        self.cart = SpatialCoordinate(mesh)

        if mesh.geometric_dimension() == 1:
            self.x = SpatialCoordinate(mesh)
            self.e_x = as_vector([Constant(1.0)])
            self.cart_name = ['x']
        elif mesh.geometric_dimension() == 2:
            # We might be an extruded interval -- then we want xz
            self.e_x = as_vector([Constant(1.0), Constant(0.0)])
            if domain == 'interval':
                self.xy = None
                self.xz = SpatialCoordinate(mesh)
                self.e_y = None
                self.e_z = as_vector([Constant(0.0), Constant(1.0)])
                self.cart_name = ['x', 'z']
                self.e_up = self.e_z
            else:
                self.xy = SpatialCoordinate(mesh)
                self.xz = None
                self.e_y = as_vector([Constant(0.0), Constant(1.0)])
                self.e_z = None
                self.cart_name = ['x', 'y']
                self.e_up = None
        elif mesh.geometric_dimension() == 3:
            self.xyz = SpatialCoordinate(mesh)
            self.e_x = as_vector([Constant(1.0), Constant(0.0), Constant(0.0)])
            self.e_y = as_vector([Constant(0.0), Constant(1.0), Constant(0.0)])
            self.e_z = as_vector([Constant(0.0), Constant(0.0), Constant(1.0)])
            self.cart_name = ['x', 'y', 'z']
            self.e_up = self.e_z

        #----------------------------------------------------------------------#
        # Set up specific coordinate systems for each domain type
        #----------------------------------------------------------------------#

        if domain.on_sphere:

            R = sqrt(self.xyz[0]**2 + self.xyz[1]**2)  # distance from z axis
            r = sqrt(self.xyz[0]**2 + self.xyz[1]**2 + self.xyz[2]**2)  # distance from origin

            lon = atan_2(self.xyz[1], self.xyz[0])
            lat = asin(self.xyz[2]/r)

            self.lonlatr = (lon, lat, r)

            # Basis vectors
            self.e_lon = (self.xyz[0] * self.e_y - self.xyz[1] * self.e_x) / R
            self.e_lat = (-self.xyz[0]*self.xyz[2]/R * self.e_x - self.xyz[1]*self.xyz[2]/R * self.e_y + R * self.e_z) / r
            self.e_r = (self.xyz[0] * self.e_x + self.xyz[1] * self.e_y + self.xyz[2] * self.e_z) / r

            self.coords = self.lonlatr
            self.coords_name = ['lon', 'lat', 'r']
            self.e_up = self.e_r
        else:
            self.coords = self.cart
            self.coords_name = self.cart_name

        #----------------------------------------------------------------------#
        # Store chi field
        #----------------------------------------------------------------------#

        self.chi_cart = {}        # Dictionary of Cartesian coords by space
        self.chi_coords = {}      # Dictionary of natural coords by space
        self.full_chi_coords = {}   # Dictionary of numpy arrays of coord data
        self.parallel_array_lims = {}   # Dictionary of parallel lengths

        comm = mesh.comm
        comm_size = comm.Get_size()
        my_rank = comm.Get_rank()

        # Loop through spaces
        for space_name in space_names:
            space = domain.spaces(space_name)

            # Use the appropriate scalar function space if the space is vector
            if (isinstance(space.ufl_element(), VectorElement) or
                isinstance(space.ufl_element(), TensorElement)):
                raise NotImplementedError('Coordinates for vector or tensor function spaces not implemented')
                # TODO: get scalar space, and only compute coordinates if necessary

            self.chi_cart[space_name] = []
            self.chi_coords[space_name] = []

            # Now set up
            for i in range(mesh.geometric_dimension()):
                self.chi_cart[space_name].append(Function(space).interpolate(self.cart[i]))
                self.chi_coords[space_name].append(Function(space).interpolate(self.coords[i]))

            # ---------------------------------------------------------------- #
            # Code for settings up coordinates for parallel-serial IO
            # ---------------------------------------------------------------- #

            len_coords = space.dim()
            my_num_dofs = len(self.chi_coords[space_name][0].dat.data_ro[:])

            if my_rank != 0:
                # Do not make full coordinate array
                self.full_chi_coords[space_name] = None
                self.parallel_array_lims[space_name] = None
                # Find number of DoFs on this processor
                comm.send(my_num_dofs, dest=0)
            else:
                # First processor has one large array of the global chi data
                self.full_chi_coords[space_name] = np.zeros((self.topological_dimension, len_coords))
                # Store the limits inside this array telling us how data is partitioned
                self.parallel_array_lims[space_name] = np.zeros((comm_size,2), dtype=int)
                # First processor has the first bit of data
                self.parallel_array_lims[space_name][my_rank][0] = 0
                self.parallel_array_lims[space_name][my_rank][1] = my_num_dofs - 1
                # Receive number of DoFs on other processors
                for procid in range(1,comm_size):
                    other_num_dofs = comm.recv(source=procid)
                    self.parallel_array_lims[space_name][procid][0] = self.parallel_array_lims[space_name][procid-1][1] + 1
                    self.parallel_array_lims[space_name][procid][1] = self.parallel_array_lims[space_name][procid][0] + other_num_dofs - 1

            # Now move coordinates to first processor
            for i in range(self.topological_dimension):
                if my_rank != 0:
                    # Send information to rank 0
                    my_tag = comm_size*i + my_rank
                    comm.send(self.chi_coords[space_name][i].dat.data_ro[:], dest=0, tag=my_tag)
                else:
                    # Rank 0 -- the receiver
                    (low_lim, up_lim) = self.parallel_array_lims[space_name][my_rank][:]
                    self.full_chi_coords[space_name][i][low_lim:up_lim+1] = self.chi_coords[space_name][i].dat.data_ro[:]
                    # Receive coords from each processor and put them into array
                    for procid in range(1, comm_size):
                        my_tag = comm_size*i + procid
                        new_coords = comm.recv(source=procid, tag=my_tag)
                        (low_lim, up_lim) = self.parallel_array_lims[space_name][procid][:]
                        self.full_chi_coords[space_name][i,low_lim:up_lim+1] = new_coords
