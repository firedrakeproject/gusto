"""
The Domain object that is provided in this module contains the model's mesh and
the set of compatible function spaces defined upon it.
"""

from gusto.function_spaces import Spaces
from firedrake import (Constant, SpatialCoordinate, sqrt, CellNormal, cross,
                       as_vector, inner, interpolate)

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
    def __init__(self, mesh, family, degree=None,
                 horizontal_degree=None, vertical_degree=None):
        """
        Args:
            mesh (:class:`Mesh`): the model's mesh.
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

        # Checks on degree arguments
        if degree is None and horizontal_degree is None:
            raise ValueError('Either "degree" or "horizontal_degree" must be passed to Domain')
        if mesh.extruded and degree is None and vertical_degree is None:
            raise ValueError('For extruded meshes, either degree or "vertical_degree" must be passed to Domain')
        if degree is not None and horizontal_degree is not None:
            raise ValueError('Cannot pass both "degree" and "horizontal_degree" to Domain')
        if mesh.extruded and degree is not None and vertical_degree is not None:
            raise ValueError('Cannot pass both "degree" and "vertical_degree" to Domain')
        if not mesh.extruded and vertical_degree is not None:
            raise ValueError('Cannot pass "vertical_degree" to Domain if mesh is not extruded')

        # Get degrees
        self.horizontal_degree = degree if horizontal_degree is None else horizontal_degree
        self.vertical_degree = degree if vertical_degree is None else vertical_degree

        self.mesh = mesh
        self.family = family
        self.spaces = Spaces(mesh)
        # Build and store compatible spaces
        self.compatible_spaces = [space for space in self.spaces.build_compatible_spaces(self.family, self.horizontal_degree, self.vertical_degree)]

        # Figure out if we're on a sphere
        # TODO: could we run on other domains that could confuse this?
        if hasattr(mesh, "_base_mesh"):
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
                outward_normals = CellNormal(mesh)
                self.perp = lambda u: cross(outward_normals, u)
        else:
            kvec = [0.0]*dim
            kvec[dim-1] = 1.0
            self.k = Constant(kvec)
            if dim == 2:
                self.perp = lambda u: as_vector([-u[1], u[0]])
