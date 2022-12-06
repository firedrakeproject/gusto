"""
The Domain object that is provided in this module contains the model's mesh and
the set of compatible function spaces defined upon it.
"""

from gusto.spaces import Spaces
from firedrake import (Constant, SpatialCoordinate, sqrt, CellNormal, cross,
                       as_vector)

class Domain(object):
    """The Domain holds the model's mesh and its compatible function spaces."""
    def __init__(self, mesh, family, degree):
        """
        Args:
            mesh (:class:`Mesh`): the model's mesh.
            family (str): the finite element space family used for the velocity
                field. This determines the other finite element spaces used via
                the de Rham complex.
            degree (int): the element degree used for the velocity space.
        """

        self.mesh = mesh
        self.spaces = [space for space in self._build_spaces(state, family, degree)]

        # figure out if we're on a sphere
        try:
            self.on_sphere = (mesh._base_mesh.geometric_dimension() == 3 and mesh._base_mesh.topological_dimension() == 2)
        except AttributeError:
            self.on_sphere = (mesh.geometric_dimension() == 3 and mesh.topological_dimension() == 2)

        #  build the vertical normal and define perp for 2d geometries
        dim = mesh.topological_dimension()
        if self.on_sphere:
            x = SpatialCoordinate(mesh)
            R = sqrt(inner(x, x))
            self.k = interpolate(x/R, mesh.coordinates.function_space())
            if dim == 2:
                outward_normals = CellNormal(mesh)
                self.perp = lambda u: cross(outward_normals, u)
        else:
            kvec = [0.0]*dim
            kvec[dim-1] = 1.0
            self.k = Constant(kvec)
            if dim == 2:
                self.perp = lambda u: as_vector([-u[1], u[0]])

    # TODO: why have this as a separate routine?
    def _build_spaces(self, family, degree):
        spaces = Spaces(self.mesh)
        return spaces.build_compatible_spaces(family, degree)
