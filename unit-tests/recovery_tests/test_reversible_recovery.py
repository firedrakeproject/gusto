"""
Test whether the reversible recovery process is working appropriately.

This is tested for:
- the lowest-order density space recovered to DG1 on a 1D mesh
- the lowest-order density space recovered to DG1 on 2D spherical mesh
- the lowest-order HDiv spaces recovered to the next-to-lowest order space on
  2D spherical meshes
"""

from firedrake import (IntervalMesh, CubedSphereMesh, IcosahedralSphereMesh,
                       SpatialCoordinate, FunctionSpace,
                       Function, norm, errornorm, as_vector)
from gusto import *
import numpy as np
import pytest

np.random.seed(0)


@pytest.fixture
def mesh(geometry):

    if geometry == "interval":
        Lx = 100.
        deltax = Lx / 5.
        ncolumnsx = int(Lx/deltax)
        m = IntervalMesh(ncolumnsx, Lx)
    else:
        ref_level = 2
        radius = 10.0
        if geometry == "spherical_quads":
            m = CubedSphereMesh(radius, refinement_level=ref_level, degree=2)
        elif geometry == "spherical_triangles":
            m = IcosahedralSphereMesh(radius, refinement_level=ref_level, degree=2)

    return m


def expr(geometry, mesh):

    if geometry == "interval":
        x, = SpatialCoordinate(mesh)
        scalar_expr = np.random.randn() + np.random.randn() * x
        vector_expr = None

    else:
        xyz = SpatialCoordinate(mesh)
        mesh.init_cell_orientations(xyz)
        scalar_expr = np.random.randn() + np.random.randn() * xyz[2]
        u0 = np.random.randn()
        vector_expr = as_vector([-u0*xyz[1], u0*xyz[0], 0.0])

    return scalar_expr, vector_expr


def low_projector(method, field_in, field_out):

    if method == 'interpolate':
        operator = Interpolator(field_in, field_out)
    elif method == 'project':
        operator = Projector(field_in, field_out)
    elif method == 'broken':
        operator = Recoverer(field_in, field_out)

    return operator


@pytest.mark.parametrize("geometry", ["interval", "spherical_quads", "spherical_triangles"])
@pytest.mark.parametrize("method", ["interpolate", "project"])
def test_reversible_recovery(geometry, mesh, method):

    # spaces
    DG0 = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)
    DG1 = FunctionSpace(mesh, "DG", 1)

    if geometry == 'spherical_quads':
        Vu_low = FunctionSpace(mesh, "RTCF", 1)
        Vu_high = FunctionSpace(mesh, "RTCF", 2)
        Vu_hcurl = FunctionSpace(mesh, "RTCE", 1)
    elif geometry == 'spherical_triangles':
        Vu_low = FunctionSpace(mesh, "RTF", 1)
        Vu_high = FunctionSpace(mesh, "RTF", 2)
        Vu_hcurl = FunctionSpace(mesh, "RTE", 1)

    scalar_expr, vector_expr = expr(geometry, mesh)

    # scalar fields
    rho_low = Function(DG0).interpolate(scalar_expr)
    rho_high_true = Function(DG1).interpolate(scalar_expr)
    rho_low_back = Function(DG0)
    rho_high = Function(DG1)

    # vector fields
    if geometry != 'interval' and method == 'project':
        u_low = Function(Vu_low).project(vector_expr)
        u_high_true = Function(Vu_high).project(vector_expr)
        u_low_back = Function(Vu_low)
        u_high = Function(Vu_high)

    # set recovery options
    scalar_rec_opts = RecoveryOptions(embedding_space=DG1,
                                      recovered_space=CG1,
                                      injection_method=method,
                                      project_high_method=method,
                                      project_low_method=method,
                                      boundary_method=BoundaryMethod.taylor)

    scalar_recoverer = ReversibleRecoverer(rho_low, rho_high, scalar_rec_opts)
    scalar_back_operator = low_projector(method, rho_high, rho_low_back)

    if geometry != 'interval' and method == 'project':
        vector_rec_opts = RecoveryOptions(embedding_space=Vu_high,
                                          recovered_space=Vu_hcurl,
                                          injection_method='recover',
                                          project_high_method='project',
                                          project_low_method='project',
                                          broken_method='project')

        vector_recoverer = ReversibleRecoverer(u_low, u_high, vector_rec_opts)
        vector_back_operator = low_projector('project', u_high, u_low_back)

    # items for testing
    # no exact answer for recovery on sphere
    rec_tol = 1e-11 if geometry == 'interval' else 0.03
    rev_tol = 1e-11
    vec_rev_tol = 2e-9
    error_message = ("""
                     Unacceptable error in {test} test for {variable} on
                     {geometry} mesh with {method} method
                     """)

    # perform scalar recovery and check if answers are correct
    scalar_recoverer.project()
    scalar_back_operator.interpolate() if method == 'interpolate' else scalar_back_operator.project()

    rho_diff = errornorm(rho_high, rho_high_true) / norm(rho_high_true)
    assert rho_diff < rec_tol, error_message.format(test='recovery',
                                                    variable='scalar',
                                                    geometry=geometry,
                                                    method=method)

    rho_diff = errornorm(rho_low, rho_low_back) / norm(rho_low)
    assert rho_diff < rev_tol, error_message.format(test='reversibility',
                                                    variable='scalar',
                                                    geometry=geometry,
                                                    method=method)

    if geometry != 'interval' and method == 'project':
        vector_recoverer.project()
        vector_back_operator.project()

        u_diff = errornorm(u_high, u_high_true) / norm(u_high_true)
        assert u_diff < rec_tol, error_message.format(test='recovery',
                                                      variable='vector',
                                                      geometry=geometry,
                                                      method=method)

        u_diff = errornorm(u_low, u_low_back) / norm(u_low)
        assert u_diff < vec_rev_tol, error_message.format(test='reversibility',
                                                          variable='vector',
                                                          geometry=geometry,
                                                          method=method)
