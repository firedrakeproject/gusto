"""
Test whether the boundary recovery is working in on 2D horizontal Cartesian meshes.
To be working, a linearly varying field should be exactly recovered.

This is tested for:
- the lowest-order density space recovered to DG1
- the lowest-order velocity space recovered to vector DG1
"""

from firedrake import (PeriodicRectangleMesh, RectangleMesh,
                       SpatialCoordinate, FiniteElement, FunctionSpace,
                       Function, norm, errornorm,
                       VectorFunctionSpace, as_vector)
from gusto import *
import numpy as np
import pytest

np.random.seed(0)


@pytest.fixture
def mesh(geometry, element):

    Lx = 100.
    Ly = 100.

    deltax = Lx / 5.
    deltay = Ly / 5.
    ncolumnsy = int(Ly/deltay)
    ncolumnsx = int(Lx/deltax)

    quadrilateral = True if element == "quadrilateral" else False

    if geometry == "periodic-in-both":
        m = PeriodicRectangleMesh(ncolumnsx, ncolumnsy, Lx, Ly,
                                  direction='both', quadrilateral=quadrilateral)
    elif geometry == "periodic-in-x":
        m = PeriodicRectangleMesh(ncolumnsx, ncolumnsy, Lx, Ly,
                                  direction='x', quadrilateral=quadrilateral)
    elif geometry == "periodic-in-y":
        m = PeriodicRectangleMesh(ncolumnsx, ncolumnsy, Lx, Ly,
                                  direction='y', quadrilateral=quadrilateral)
    elif geometry == "non-periodic":
        m = RectangleMesh(ncolumnsx, ncolumnsy, Lx, Ly, quadrilateral=quadrilateral)

    return m


@pytest.fixture
def expr(geometry, mesh):

    x, y = SpatialCoordinate(mesh)

    if geometry == "periodic-in-both":
        # N.B. this is a very trivial test -- no boundary recovery should happen
        analytic_expr = np.random.randn() + 0.0 * x
    elif geometry == "periodic-in-x":
        analytic_expr = np.random.randn() + np.random.randn() * y
    elif geometry == "periodic-in-y":
        analytic_expr = np.random.randn() + np.random.randn() * x
    elif geometry == "non-periodic":
        analytic_expr = np.random.randn() + np.random.randn() * x + np.random.randn() * y
    return analytic_expr


@pytest.mark.parametrize("geometry", ["periodic-in-both", "periodic-in-x",
                                      "periodic-in-y", "non-periodic"])
@pytest.mark.parametrize("element", ["quadrilateral", "triangular"])
def test_2D_cartesian_recovery(geometry, element, mesh, expr):

    family = "RTCF" if element == "quadrilateral" else "BDM"

    # horizontal base spaces
    cell = mesh.ufl_cell().cellname()

    # DG1
    DG1_elt = FiniteElement("DG", cell, 1, variant="equispaced")
    DG1 = FunctionSpace(mesh, DG1_elt)
    vec_DG1 = VectorFunctionSpace(mesh, DG1_elt)

    # spaces
    DG0 = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)
    Vu = FunctionSpace(mesh, family, 1)
    vec_CG1 = VectorFunctionSpace(mesh, "CG", 1)

    # our actual theta and rho and v
    rho_CG1_true = Function(CG1).interpolate(expr)
    v_CG1_true = Function(vec_CG1).interpolate(as_vector([expr, expr]))

    # make the initial fields by projecting expressions into the lowest order spaces
    rho_DG0 = Function(DG0).interpolate(expr)
    rho_CG1 = Function(CG1)
    v_Vu = Function(Vu).project(as_vector([expr, expr]))
    v_CG1 = Function(vec_CG1)

    # make the recoverers and do the recovery
    rho_recoverer = Recoverer(rho_DG0, rho_CG1, VDG=DG1, boundary_method=Boundary_Method.dynamics)
    v_recoverer = Recoverer(v_Vu, v_CG1, VDG=vec_DG1, boundary_method=Boundary_Method.dynamics)

    rho_recoverer.project()
    v_recoverer.project()

    rho_diff = errornorm(rho_CG1, rho_CG1_true) / norm(rho_CG1_true)
    v_diff = errornorm(v_CG1, v_CG1_true) / norm(v_CG1_true)
    tolerance = 1e-7
    error_message = ("""
                     Incorrect recovery for {variable} with {boundary} boundary method
                     on {geometry} 2D Cartesian plane with {element} elements
                     """)
    assert rho_diff < tolerance, error_message.format(variable='rho', boundary='dynamics', geometry=geometry, element=element)
    assert v_diff < tolerance, error_message.format(variable='v', boundary='dynamics', geometry=geometry, element=element)
