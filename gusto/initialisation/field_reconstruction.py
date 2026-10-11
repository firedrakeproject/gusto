"""Tools for reconstructing fields between different element degrees."""

import numpy as np
from firedrake import (
    PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate, Function,
    VectorFunctionSpace, Constant, dx, assemble
)
from gusto.core import Domain


__all__ = ["initial_buoyancy_field_degree1_from_degree0"]


def initial_buoyancy_field_degree1_from_degree0(domain, buoyancy_expr):
    """
    Initialises a theta-space field (such as buoyancy) on the degree 1
    theta space of the given domain by first interpolating an analytic
    expression to the nodes of the corresponding degree 0 theta space
    (which has twice the number of columns), and then, within each degree
    1 cell, linearly interpolating the two corresponding degree 0 nodal
    values to obtain the degree 1 nodal values.

    This construction is the inverse of the point-evaluation process used
    to extract slice data from a field at the centres of the degree 0
    cells, so the resulting degree 1 field reproduces the same spectrum as
    the degree 0 field when it is in turn sampled at those points.

    Args:
        domain (:class:`Domain`): the degree 1 domain to initialise the
            field on. Must use an extruded, two-dimensional mesh with
            horizontal and vertical degree 1.
        buoyancy_expr (callable): a function taking the (x, z) components
            of a mesh's :class:`SpatialCoordinate` and returning the UFL
            expression for the initial field on that mesh.

    Returns:
        :class:`Function`: the initialised field on the degree 1 theta
            space.
    """

    mesh = domain.mesh

    if not mesh.extruded:
        raise ValueError(
            'initial_buoyancy_field_degree1_from_degree0 requires an '
            'extruded mesh'
        )
    if mesh.topological_dimension != 2:
        raise ValueError(
            'initial_buoyancy_field_degree1_from_degree0 only supports '
            f'two-dimensional (vertical slice) meshes, not dimension '
            f'{mesh.topological_dimension}'
        )
    if domain.horizontal_degree != 1 or domain.vertical_degree != 1:
        raise ValueError(
            'initial_buoyancy_field_degree1_from_degree0 requires a '
            'domain with horizontal and vertical degree 1, but got '
            f'horizontal degree {domain.horizontal_degree} and vertical '
            f'degree {domain.vertical_degree}'
        )

    Vb = domain.spaces('theta')

    base_mesh = mesh._base_mesh
    ncolumns = base_mesh.num_cells()
    nlayers = mesh.layers - 1

    domain_width = assemble(Constant(1.0)*dx(domain=base_mesh))
    domain_height = mesh.coordinates.dat.data_ro[:, 1].max()

    # Degree 0 field, with twice the number of columns
    base_mesh_fine = PeriodicIntervalMesh(2*ncolumns, domain_width)
    mesh_fine = ExtrudedMesh(
        base_mesh_fine, nlayers, layer_height=domain_height/nlayers
    )
    domain_fine = Domain(mesh_fine, 1.0, 'CG', 0)
    Vb_fine = domain_fine.spaces('theta')
    x_fine, z_fine = SpatialCoordinate(mesh_fine)
    b0_fine = Function(Vb_fine).interpolate(buoyancy_expr(x_fine, z_fine))

    # Coordinates of every degree 1 nodal point
    W = VectorFunctionSpace(mesh, Vb.ufl_element())
    xz = Function(W).interpolate(SpatialCoordinate(mesh)).dat.data_ro

    coarse_dx = domain_width / ncolumns
    cell_index = np.floor(xz[:, 0] / coarse_dx).astype(int) % ncolumns
    xi_local = xz[:, 0]/coarse_dx - cell_index
    left_edge = cell_index * coarse_dx

    # The two degree 0 cells within each degree 1 cell sit at the
    # quarter-points of the degree 1 cell
    points_v1 = list(zip(left_edge + 0.25*coarse_dx, xz[:, 1]))
    points_v2 = list(zip(left_edge + 0.75*coarse_dx, xz[:, 1]))
    v1 = np.array(b0_fine.at(points_v1))
    v2 = np.array(b0_fine.at(points_v2))

    # Linear interpolation through the two degree 0 values, which sit at
    # relative positions 0.25 and 0.75 within the degree 1 cell
    b0_coarse = Function(Vb)
    b0_coarse.dat.data[:] = v1 + (v2 - v1)*2.0*(xi_local - 0.25)

    return b0_coarse
