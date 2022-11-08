"""
Test whether the boundary recovery is working in 2D vertical slices.
To be working, a linearly varying field should be exactly recovered.

This is tested for:
- the lowest-order density space recovered to DG1
- the lowest-order velocity space recovered to vector DG1
- the lowest-order temperature space recovered to DG1
- the lowest-order density space recovered to lowest-order temperature space.
"""

from firedrake import (PeriodicIntervalMesh, IntervalMesh, ExtrudedMesh,
                       SpatialCoordinate, FiniteElement, HDiv, FunctionSpace,
                       TensorProductElement, Function, interval, norm, errornorm,
                       VectorFunctionSpace, as_vector)
from gusto import *
import numpy as np
import pytest

np.random.seed(0)


@pytest.fixture
def mesh(geometry):

    L = 100.
    H = 100.

    deltax = L / 5.
    deltaz = H / 5.
    nlayers = int(H/deltaz)
    ncolumns = int(L/deltax)

    if geometry == "periodic":
        m = PeriodicIntervalMesh(ncolumns, L)
    elif geometry == "non-periodic":
        m = IntervalMesh(ncolumns, L)

    extruded_mesh = ExtrudedMesh(m, layers=nlayers, layer_height=deltaz)

    return extruded_mesh


@pytest.fixture
def expr(geometry, mesh):

    x, z = SpatialCoordinate(mesh)

    if geometry == "periodic":
        analytic_expr = np.random.randn() + np.random.randn() * z
    elif geometry == "non-periodic":
        analytic_expr = np.random.randn() + np.random.randn() * x + np.random.randn() * z

    return analytic_expr


@pytest.mark.parametrize("geometry", ["periodic", "non-periodic"])
def test_vertical_slice_recovery(geometry, mesh, expr):

    # horizontal base spaces
    cell = mesh._base_mesh.ufl_cell().cellname()
    u_hori = FiniteElement("CG", cell, 1)
    w_hori = FiniteElement("DG", cell, 0)

    # vertical base spaces
    u_vert = FiniteElement("DG", interval, 0)
    w_vert = FiniteElement("CG", interval, 1)

    # build elements
    u_element = HDiv(TensorProductElement(u_hori, u_vert))
    w_element = HDiv(TensorProductElement(w_hori, w_vert))
    theta_element = TensorProductElement(w_hori, w_vert)
    v_element = u_element + w_element

    # spaces
    DG0 = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)
    Vt = FunctionSpace(mesh, theta_element)
    Vu = FunctionSpace(mesh, v_element)
    vec_CG1 = VectorFunctionSpace(mesh, "CG", 1)

    # our actual theta and rho and v
    rho_CG1_true = Function(CG1).interpolate(expr)
    theta_CG1_true = Function(CG1).interpolate(expr)
    v_CG1_true = Function(vec_CG1).interpolate(as_vector([expr, expr]))
    rho_Vt_true = Function(Vt).interpolate(expr)

    # make the initial fields by projecting expressions into the lowest order spaces
    rho_DG0 = Function(DG0).interpolate(expr)
    rho_CG1 = Function(CG1)
    theta_Vt = Function(Vt).interpolate(expr)
    theta_CG1 = Function(CG1)
    v_Vu = Function(Vu).project(as_vector([expr, expr]))
    v_CG1 = Function(vec_CG1)
    rho_Vt = Function(Vt)

    # make the recoverers and do the recovery
    rho_recoverer = Recoverer(rho_DG0, rho_CG1, boundary_method=BoundaryMethod.taylor)
    theta_recoverer = Recoverer(theta_Vt, theta_CG1, boundary_method=BoundaryMethod.taylor)
    v_recoverer = Recoverer(v_Vu, v_CG1, boundary_method=BoundaryMethod.taylor)
    rho_Vt_recoverer = Recoverer(rho_DG0, rho_Vt, boundary_method=BoundaryMethod.extruded)

    rho_recoverer.project()
    theta_recoverer.project()
    v_recoverer.project()
    rho_Vt_recoverer.project()

    rho_diff = errornorm(rho_CG1, rho_CG1_true) / norm(rho_CG1_true)
    theta_diff = errornorm(theta_CG1, theta_CG1_true) / norm(theta_CG1_true)
    v_diff = errornorm(v_CG1, v_CG1_true) / norm(v_CG1_true)
    rho_Vt_diff = errornorm(rho_Vt, rho_Vt_true) / norm(rho_Vt_true)

    tolerance = 1e-7
    error_message = 'Incorrect recovery for {variable} with {boundary} boundary method on {geometry} vertical slice'
    assert rho_diff < tolerance, error_message.format(variable='rho', boundary='taylor', geometry=geometry)
    assert theta_diff < tolerance, error_message.format(variable='theta', boundary='taylor', geometry=geometry)
    assert v_diff < tolerance, error_message.format(variable='v', boundary='taylor', geometry=geometry)
    assert rho_Vt_diff < tolerance, error_message.format(variable='rho', boundary='physics', geometry=geometry)
