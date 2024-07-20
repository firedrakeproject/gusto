"""
This tests the ConservativeProjector object, by projecting a mixing ratio from
DG1 to DG0, relative to different density fields, and checking that the tracer
mass is conserved.
"""

from firedrake import (UnitSquareMesh, FunctionSpace, Constant,
                       Function, assemble, dx, sin, SpatialCoordinate)
from gusto import ConservativeProjector, ContinuousConservativeProjector
import pytest

@pytest.mark.parametrize("projection", ["discontinuous", "continuous"])
def test_conservative_projection(projection):

    # Set up mesh on plane
    mesh = UnitSquareMesh(3, 3)

    # Function spaces and functions
    DG0 = FunctionSpace(mesh, "DG", 0)
    DG1 = FunctionSpace(mesh, "DG", 1)

    rho_DG0 = Function(DG0)
    rho_DG1 = Function(DG1)
    m_DG1 = Function(DG1)

    if projection == "continuous":
        CG1 = FunctionSpace(mesh, "CG", 1)
        m_CG1 = Function(CG1)
    else:
        m_DG0 = Function(DG0)

    # Projector object
    if projection == "continuous":
        projector = ContinuousConservativeProjector(rho_DG1, rho_DG0, m_DG1, m_CG1)
    else:
        projector = ConservativeProjector(rho_DG1, rho_DG0, m_DG1, m_DG0)

    # Initial conditions
    x, y = SpatialCoordinate(mesh)

    rho_expr = Constant(1.0) + 0.5*x*y**2
    m_expr = Constant(2.0) + 0.6*sin(x)

    rho_DG1.interpolate(rho_expr)
    m_DG1.interpolate(m_expr)
    rho_DG0.project(rho_DG1)

    # Test projection
    projector.project()

    tol = 1e-14
    mass_DG1 = assemble(rho_DG1*m_DG1*dx)

    if projection == "continuous":
        mass_CG1 = assemble(rho_DG0*m_CG1*dx)

        assert abs(mass_CG1 - mass_DG1) < tol, "continuous projection is not conservative"

    else:
        mass_DG0 = assemble(rho_DG0*m_DG0*dx)

        assert abs(mass_DG0 - mass_DG1) < tol, "discontinuous projection is not conservative"

