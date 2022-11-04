"""
Test whether the boundary recovery is working in on a 1D mesh.
To be working, a linearly varying field should be exactly recovered.

This is tested for:
- the lowest-order density space recovered to DG1
"""

from firedrake import (PeriodicIntervalMesh, IntervalMesh,
                       SpatialCoordinate, FunctionSpace,
                       Function, norm, errornorm)
from gusto import *
import numpy as np
import pytest

np.random.seed(0)


@pytest.fixture
def mesh(geometry):

    Lx = 100.
    deltax = Lx / 5.
    ncolumnsx = int(Lx/deltax)

    if geometry == "periodic":
        m = PeriodicIntervalMesh(ncolumnsx, Lx)
    elif geometry == "non-periodic":
        m = IntervalMesh(ncolumnsx, Lx)

    return m


@pytest.fixture
def expr(geometry, mesh):

    x, = SpatialCoordinate(mesh)

    if geometry == "periodic":
        # N.B. this is a very trivial test -- no boundary recovery should happen
        analytic_expr = np.random.randn() + 0.0 * x
    elif geometry == "non-periodic":
        analytic_expr = np.random.randn() + np.random.randn() * x

    return analytic_expr


@pytest.mark.parametrize("geometry", ["periodic", "non-periodic"])
def test_1D_recovery(geometry, mesh, expr):

    # spaces
    DG0 = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)

    # make the initial fields by projecting expressions into the lowest order spaces
    rho_DG0 = Function(DG0).interpolate(expr)

    # make the recoverers and do the recovery
    rho_CG1 = Function(CG1)
    rho_CG1_true = Function(CG1).interpolate(expr)
    rho_recoverer = Recoverer(rho_DG0, rho_CG1, boundary_method=BoundaryMethod.taylor)
    rho_recoverer.project()

    rho_diff = errornorm(rho_CG1, rho_CG1_true) / norm(rho_CG1_true)

    tolerance = 1e-7
    error_message = ("""
                     Incorrect recovery for {variable} with {boundary} boundary method
                     on {geometry} 1D domain
                     """)
    assert rho_diff < tolerance, error_message.format(variable='rho', boundary='taylor', geometry=geometry)
