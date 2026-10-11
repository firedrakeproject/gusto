"""
Tests the `initial_buoyancy_field_degree1_from_degree0` routine, which
reconstructs a degree 1 theta-space field via cellwise linear interpolation
of nodal values taken from the corresponding (double horizontal resolution)
degree 0 theta space.
"""

from firedrake import (
    PeriodicIntervalMesh, ExtrudedMesh, UnitSquareMesh, SpatialCoordinate,
    Function, VectorFunctionSpace
)
from gusto import Domain, initial_buoyancy_field_degree1_from_degree0
import numpy as np
import pytest


def build_vertical_slice_domain(ncolumns, nlayers, domain_width,
                                 domain_height, degree):
    base_mesh = PeriodicIntervalMesh(ncolumns, domain_width)
    mesh = ExtrudedMesh(base_mesh, nlayers, layer_height=domain_height/nlayers)
    return Domain(mesh, 1.0, 'CG', degree)


def test_initial_buoyancy_field_degree1_from_degree0_exact_for_affine():

    ncolumns = 4
    nlayers = 3
    domain_width = 8.0
    domain_height = 6.0

    domain = build_vertical_slice_domain(
        ncolumns, nlayers, domain_width, domain_height, 1
    )

    def buoyancy_expr(x, z):
        return 2.0*x + 3.0*z + 1.0

    b0 = initial_buoyancy_field_degree1_from_degree0(domain, buoyancy_expr)

    # An affine function should be reconstructed exactly, since two points
    # always determine a unique linear function that matches it everywhere
    Vb = domain.spaces('theta')
    W = VectorFunctionSpace(domain.mesh, Vb.ufl_element())
    xz = Function(W).interpolate(SpatialCoordinate(domain.mesh)).dat.data_ro
    expected = 2.0*xz[:, 0] + 3.0*xz[:, 1] + 1.0

    err_tol = 1e-10
    assert np.allclose(b0.dat.data_ro, expected, atol=err_tol), \
        'initial_buoyancy_field_degree1_from_degree0 should exactly ' \
        + 'reproduce an affine buoyancy expression'


@pytest.mark.parametrize(
    "invalid_case", ["wrong_degree", "not_extruded", "wrong_dimension"]
)
def test_initial_buoyancy_field_degree1_from_degree0_invalid_domain(invalid_case):

    def buoyancy_expr(x, z):
        return x + z

    if invalid_case == "wrong_degree":
        domain = build_vertical_slice_domain(4, 3, 8.0, 6.0, 0)
    elif invalid_case == "not_extruded":
        mesh = PeriodicIntervalMesh(4, 8.0)
        domain = Domain(mesh, 1.0, 'CG', 1)
    elif invalid_case == "wrong_dimension":
        base_mesh = UnitSquareMesh(2, 2)
        mesh = ExtrudedMesh(base_mesh, 3, 3)
        domain = Domain(mesh, 1.0, 'RT', 1)
    else:
        raise ValueError(f'{invalid_case} not recognised')

    with pytest.raises(ValueError):
        initial_buoyancy_field_degree1_from_degree0(domain, buoyancy_expr)
