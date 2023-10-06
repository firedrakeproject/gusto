"""
Check that the H1 spaces are correct by:
- creating HDiv spaces from a stream function in H1, and checking that their
  divergence in L2 is zero
- checking that this velocity gives a sensible answer
- projecting the velocity back to obtain the vorticity field, and checking
  that this is correct
"""

from firedrake import (PeriodicRectangleMesh, SpatialCoordinate, norm,
                       Function, sin, cos, pi, as_vector, grad, div,
                       TestFunction, TrialFunction, dx, inner, errornorm,
                       LinearVariationalProblem, LinearVariationalSolver)
from gusto import Domain
import pytest

hdiv_families = ["RTF", "BDMF", "BDFM", "RTCF", "BDMCF"]
degrees = [1, 2]


def combos(families, degs):
    # Form all combinations of families/degrees
    # This is done because for BDFM family, only degree 1 is possible
    all_combos = []
    for family in families:
        for deg in degs:
            if not (family == 'BDFM' and deg != 1):
                all_combos.append((family, deg))
    return all_combos


@pytest.mark.parametrize("hdiv_family, degree", combos(hdiv_families, degrees))
def test_h1_spaces(hdiv_family, degree):
    # Set up mesh and domain
    dt = 2.0
    Lx = 10
    Ly = 10
    nx = int(20 / degree)
    ny = int(20 / degree)
    quadrilateral = True if hdiv_family in ['RTCF', 'BDMCF'] else False
    mesh = PeriodicRectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)
    domain = Domain(mesh, dt, hdiv_family, degree)
    x, y = SpatialCoordinate(mesh)

    # Declare spaces and functions
    H1_space = domain.spaces('H1')
    HDiv_space = domain.spaces('HDiv')
    L2_space = domain.spaces('L2')
    streamfunc = Function(H1_space)
    velocity = Function(HDiv_space)
    velocity_true = Function(HDiv_space)
    vorticity = Function(H1_space)
    vorticity_true = Function(H1_space)
    divergence = Function(L2_space)
    test = TestFunction(H1_space)
    trial = TrialFunction(H1_space)

    # Expressions
    streamfunc_expr = sin(2*pi*x/Lx)*cos(4*pi*y/Ly)
    velocity_expr = as_vector([4*pi/Ly*sin(4*pi*y/Ly)*sin(2*pi*x/Lx),
                               2*pi/Lx*cos(2*pi*x/Lx)*cos(4*pi*y/Ly)])
    vorticity_expr = - ((2*pi/Lx)**2 + (4*pi/Ly)**2)*streamfunc_expr

    # Evaluate velocity from stream function
    gradperp = lambda q: domain.perp(grad(q))
    streamfunc.project(streamfunc_expr)
    velocity.project(gradperp(streamfunc))
    velocity_true.project(velocity_expr)
    divergence.project(div(velocity))
    velocity_norm = errornorm(velocity, velocity_true) / norm(velocity_true)
    divergence_norm = norm(divergence) / (Lx*Ly)

    # Evaluate vorticity from velocity
    velocity.project(velocity_expr)
    vorticity_true.project(vorticity_expr)
    eqn_lhs = trial * test * dx
    eqn_rhs = -inner(gradperp(test), velocity) * dx
    problem = LinearVariationalProblem(eqn_lhs, eqn_rhs, vorticity)
    solver = LinearVariationalSolver(problem)
    solver.solve()
    vorticity_norm = errornorm(vorticity, vorticity_true) / norm(vorticity_true)

    # Check values
    assert divergence_norm < 1e-8, \
        f'Divergence for family {hdiv_family} degree {degree} is too large'
    assert velocity_norm < 0.015, \
        f'Error in velocity for family {hdiv_family} degree {degree} is too large'
    assert vorticity_norm < 2e-6, \
        f'Error in vorticity for family {hdiv_family} degree {degree} is too large'
