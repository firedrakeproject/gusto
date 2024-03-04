"""
This tests the ZeroLimiter, which enforces non-negativity.
A sharp bubble of warm air is generated in a vertical slice and then transported
by a prescribed transport scheme. If the limiter is working, the transport
should have produced no negative values.
"""

from gusto import *
from firedrake import (as_vector, PeriodicIntervalMesh, pi, SpatialCoordinate,
                       ExtrudedMesh, FunctionSpace, Function, norm,
                       conditional, sqrt)
import numpy as np
import pytest


def setup_zero_limiter(dirname, clipping_space):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    Ld = 1.
    tmax = 0.2
    dt = tmax / 40
    rotations = 0.25

    # ------------------------------------------------------------------------ #
    # Build model objects
    # ------------------------------------------------------------------------ #

    # Domain
    m = PeriodicIntervalMesh(20, Ld)
    mesh = ExtrudedMesh(m, layers=20, layer_height=(Ld/20))
    degree = 1

    domain = Domain(mesh, dt, family="CG", degree=degree)

    DG1 = FunctionSpace(mesh, 'DG', 1)
    DG1_equispaced = domain.spaces('DG1_equispaced')

    Vpsi = domain.spaces('H1')

    eqn = AdvectionEquation(domain, DG1, 'tracer')
    output = OutputParameters(dirname=dirname+'/limiters',
                              dumpfreq=1, dumplist=['u', 'tracer', 'true_tracer'])

    io = IO(domain, output)

    # ------------------------------------------------------------------------ #
    # Set up transport scheme
    # ------------------------------------------------------------------------ #

    if clipping_space is None:
        limiter = ZeroLimiter(DG1)
    elif clipping_space == 'equispaced':
        limiter = ZeroLimiter(DG1, clipping_space=DG1_equispaced)

    transport_schemes = SSPRK3(domain, limiter=limiter)
    transport_method = DGUpwind(eqn, "tracer")

    # Build time stepper
    stepper = PrescribedTransport(eqn, transport_schemes, io, transport_method)

    # ------------------------------------------------------------------------ #
    # Initial condition
    # ------------------------------------------------------------------------ #

    tracer0 = stepper.fields('tracer', DG1)
    true_field = stepper.fields('true_tracer', space=DG1)

    x, z = SpatialCoordinate(mesh)

    tracer_min = 12.6
    dtracer = 3.2

    # First time do initial conditions, second time do final conditions
    for i in range(2):

        if i == 0:
            x1_lower = 2 * Ld / 5
            x1_upper = 3 * Ld / 5
            z1_lower = 6 * Ld / 10
            z1_upper = 8 * Ld / 10
            x2_lower = 6 * Ld / 10
            x2_upper = 8 * Ld / 10
            z2_lower = 2 * Ld / 5
            z2_upper = 3 * Ld / 5
        elif i == 1:
            # Rotated anti-clockwise by 90 degrees (x -> z, z -> -x)
            x1_lower = 2 * Ld / 10
            x1_upper = 4 * Ld / 10
            z1_lower = 2 * Ld / 5
            z1_upper = 3 * Ld / 5
            x2_lower = 2 * Ld / 5
            x2_upper = 3 * Ld / 5
            z2_lower = 6 * Ld / 10
            z2_upper = 8 * Ld / 10
        else:
            raise ValueError

        expr_1 = conditional(x > x1_lower,
                             conditional(x < x1_upper,
                                         conditional(z > z1_lower,
                                                     conditional(z < z1_upper, dtracer, 0.0),
                                                     0.0),
                                         0.0),
                             0.0)

        expr_2 = conditional(x > x2_lower,
                             conditional(x < x2_upper,
                                         conditional(z > z2_lower,
                                                     conditional(z < z2_upper, dtracer, 0.0),
                                                     0.0),
                                         0.0),
                             0.0)

        if i == 0:
            tracer0.interpolate(Constant(tracer_min) + expr_1 + expr_2)
        elif i == 1:
            true_field.interpolate(Constant(tracer_min) + expr_1 + expr_2)
        else:
            raise ValueError

    # ------------------------------------------------------------------------ #
    # Velocity profile
    # ------------------------------------------------------------------------ #

    psi = Function(Vpsi)
    u = stepper.fields('u')

    # set up solid body rotation for transport
    # we do this slightly complicated stream function to make the velocity 0 at edges
    # thus we avoid odd effects at boundaries
    xc = Ld / 2
    zc = Ld / 2
    r = sqrt((x - xc) ** 2 + (z - zc) ** 2)
    omega = rotations * 2 * pi / tmax
    r_out = 9 * Ld / 20
    r_in = 2 * Ld / 5
    A = omega * r_in / (2 * (r_in - r_out))
    B = - omega * r_in * r_out / (r_in - r_out)
    C = omega * r_in ** 2 * r_out / (r_in - r_out) / 2
    psi_expr = conditional(r < r_in,
                           omega * r ** 2 / 2,
                           conditional(r < r_out,
                                       A * r ** 2 + B * r + C,
                                       A * r_out ** 2 + B * r_out + C))
    psi.interpolate(psi_expr)

    gradperp = lambda v: as_vector([-v.dx(1), v.dx(0)])
    u.project(gradperp(psi))

    return stepper, tmax, true_field


@pytest.mark.parametrize('space', [None, 'equispaced'])
def test_zero_limiter(tmpdir, space):

    # Setup and run
    dirname = str(tmpdir)

    stepper, tmax, true_field = setup_zero_limiter(dirname, space)

    stepper.run(t=0, tmax=tmax)

    final_field = stepper.fields('tracer')

    # Check tracer is roughly in the correct place
    assert norm(true_field - final_field) / norm(true_field) < 0.05, \
        'Something appears to have gone wrong with transport of tracer using a limiter'

    # Check for no new overshoots
    assert np.min(final_field.dat.data) >= 0.0, \
        'Application of limiter has not prevented negative values'
