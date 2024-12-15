"""
This tests the transport of a vector-valued field using vorticity augmentation.
The computed solution is compared with a true one to check that the transport
is working correctly.
"""

from gusto import *
from firedrake import (
    as_vector, norm, exp, PeriodicRectangleMesh, SpatialCoordinate, min_value
)
import pytest


def run(timestepper, tmax, f_end):
    timestepper.run(0, tmax)

    return norm(timestepper.fields("f") - f_end) / norm(f_end)


@pytest.mark.parametrize("supg", [False, True])
def test_vorticity_transport_setup(tmpdir, supg):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    Lx = 2000.       # length of domain in x direction, in m
    Ly = 2000.       # width of domain in y direction, in m
    ncells_1d = 20   # number of points in x and y directions
    tmax = 500.
    degree = 1

    if supg:
        # Smaller time steps for RK scheme
        dt = 5.0
        dumpfreq = 50
    else:
        dt = 25.0
        dumpfreq = 10

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    mesh = PeriodicRectangleMesh(ncells_1d, ncells_1d, Lx, Ly, quadrilateral=True)
    domain = Domain(mesh, dt, "RTCF", degree)
    x, y = SpatialCoordinate(mesh)

    Vu = domain.spaces("HDiv")
    CG = domain.spaces("H1")

    augmentation = VorticityTransport(domain, Vu, CG, supg=True)

    # Equation
    eqn = AdvectionEquation(domain, Vu, "f")

    # I/O
    dirname = f'{tmpdir}/vorticity_plane'
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_nc=False, dump_vtus=True
    )
    io = IO(domain, output)

    augmentation = VorticityTransport(domain, Vu, CG, supg=supg)

    # Make equation
    eqn = AdvectionEquation(domain, Vu, "f")
    if supg:
        transport_scheme = SSPRK3(
            domain, augmentation=augmentation,
            rk_formulation=RungeKuttaFormulation.predictor
        )
    else:
        transport_scheme = TrapeziumRule(domain, augmentation=augmentation)
    transport_method = DGUpwind(eqn, "f")

    time_varying_velocity = False
    timestepper = PrescribedTransport(
        eqn, transport_scheme, io, time_varying_velocity, transport_method
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    # Specify locations of the two Gaussians
    xc = Lx/2.
    xend = 3.*Lx/4.
    yc = Ly/2.

    def l2_dist(xc, yc):
        return min_value(abs(x - xc), Lx - abs(x - xc))**2 + (y - yc)**2

    f0 = 1.
    lc = 4.*Lx/25.

    init_scalar_expr = f0*exp(-l2_dist(xc, yc)/lc**2)
    uexpr = as_vector([1.0, 0.0])

    # Set fields
    f = timestepper.fields("f")
    f.project(as_vector([init_scalar_expr, init_scalar_expr]))

    u0 = timestepper.fields("u")
    u0.project(uexpr)

    final_scalar_expr = f0*exp(-l2_dist(xend, yc)/lc**2)
    final_vector_expr = as_vector([final_scalar_expr, final_scalar_expr])

    # Run and check error
    error = run(timestepper, tmax, final_vector_expr)

    tol = 1e-1
    assert error < tol, \
        'The transport error is greater than the permitted tolerance'
