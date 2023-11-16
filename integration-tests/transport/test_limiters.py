"""
This tests limiter options for different transport schemes.
A sharp bubble of warm air is generated in a vertical slice and then transported
by a prescribed transport scheme. If the limiter is working, the transport
should have produced no new maxima or minima.
"""

from gusto import *
from firedrake import (as_vector, PeriodicIntervalMesh, pi, SpatialCoordinate,
                       ExtrudedMesh, FunctionSpace, Function, norm,
                       conditional, sqrt)
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
import numpy as np
import pytest


def setup_limiters(dirname, space):

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
    degree = 0 if space in ['DG0', 'Vtheta_degree_0'] else 1

    domain = Domain(mesh, dt, family="CG", degree=degree)

    if space == 'DG0':
        V = domain.spaces('DG')
        VCG1 = FunctionSpace(mesh, 'CG', 1)
        VDG1 = domain.spaces('DG1_equispaced')
    elif space == 'DG1':
        V = domain.spaces('DG')
    elif space == 'DG1_equispaced':
        V = domain.spaces('DG1_equispaced')
    elif space == 'Vtheta_degree_0':
        V = domain.spaces('theta')
        VCG1 = FunctionSpace(mesh, 'CG', 1)
        VDG1 = domain.spaces('DG1_equispaced')
    elif space == 'Vtheta_degree_1':
        V = domain.spaces('theta')
    elif space == 'mixed_FS':
        V = domain.spaces('HDiv')
        VA = domain.spaces('DG')
        VB = domain.spaces('DG1_equispaced')
    else:
        raise NotImplementedError

    Vpsi = domain.spaces('H1')

    # Equation
    if space == 'mixed_FS':
        tracerA = ActiveTracer(name='tracerA', space='DG',
                               variable_type=TracerVariableType.mixing_ratio,
                               transport_eqn=TransportEquationType.advective)
        tracerB = ActiveTracer(name='tracerB', space='DG1_equispaced',
                               variable_type=TracerVariableType.mixing_ratio,
                               transport_eqn=TransportEquationType.advective)
        tracers = [tracerA, tracerB]
        eqn = CoupledTransportEquation(domain, active_tracers=tracers, Vu=V)
        output = OutputParameters(dirname=dirname+'/limiters', dumpfreq=1,
                                  dumplist=['u', 'tracerA', 'tracerB', 'true_tracerA', 'true_tracerB'])
    else:
        eqn = AdvectionEquation(domain, V, 'tracer')
        output = OutputParameters(dirname=dirname+'/limiters',
                                  dumpfreq=1, dumplist=['u', 'tracer', 'true_tracer'])

    io = IO(domain, output)

    # ------------------------------------------------------------------------ #
    # Set up transport scheme
    # ------------------------------------------------------------------------ #

    if space in ['DG0', 'Vtheta_degree_0']:
        opts = RecoveryOptions(embedding_space=VDG1,
                               recovered_space=VCG1,
                               project_low_method='recover',
                               boundary_method=BoundaryMethod.taylor)
        transport_schemes = SSPRK3(domain, options=opts,
                                   limiter=VertexBasedLimiter(VDG1))

    elif space == 'DG1':
        transport_schemes = SSPRK3(domain, limiter=DG1Limiter(V))

    elif space == 'DG1_equispaced':
        transport_schemes = SSPRK3(domain, limiter=VertexBasedLimiter(V))

    elif space == 'Vtheta_degree_1':
        opts = EmbeddedDGOptions()
        transport_schemes = SSPRK3(domain, options=opts, limiter=ThetaLimiter(V))

    elif space == 'mixed_FS':
        sublimiters = {'tracerA': DG1Limiter(domain.spaces('DG')),
                       'tracerB': VertexBasedLimiter(domain.spaces('DG1_equispaced'))}
        MixedLimiter = MixedFSLimiter(eqn, sublimiters)
        transport_schemes = SSPRK3(domain, limiter=MixedLimiter)
    else:
        raise NotImplementedError

    if space == 'mixed_FS':
        transport_method = [DGUpwind(eqn, 'tracerA'), DGUpwind(eqn, 'tracerB')]
    else:
        transport_method = DGUpwind(eqn, "tracer")

    # Build time stepper
    stepper = PrescribedTransport(eqn, transport_schemes, io, transport_method)

    # ------------------------------------------------------------------------ #
    # Initial condition
    # ------------------------------------------------------------------------ #

    if space == 'mixed_FS':
        tracerA_0 = stepper.fields('tracerA', space=VA)
        tracerB_0 = stepper.fields('tracerB', space=VB)
        true_fieldA = stepper.fields('true_tracerA', space=VA)
        true_fieldB = stepper.fields('true_tracerB', space=VB)
    else:
        tracer0 = stepper.fields('tracer', V)
        true_field = stepper.fields('true_tracer', space=V)

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
            if space == 'mixed_FS':
                tracerA_0.interpolate(Constant(tracer_min) + expr_1 + expr_2)
                tracerB_0.interpolate(Constant(tracer_min) + expr_1 + expr_2)
            else:
                tracer0.interpolate(Constant(tracer_min) + expr_1 + expr_2)
        elif i == 1:
            if space == 'mixed_FS':
                true_fieldA.interpolate(Constant(tracer_min) + expr_1 + expr_2)
                true_fieldB.interpolate(Constant(tracer_min) + expr_1 + expr_2)
            else:
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

    if space == 'mixed_FS':
        return stepper, tmax, true_fieldA, true_fieldB
    else:
        return stepper, tmax, true_field

@pytest.mark.parametrize('space', ['Vtheta_degree_0', 'Vtheta_degree_1', 'DG0',
                                   'DG1', 'DG1_equispaced', 'mixed_FS'])
                                   

def test_limiters(tmpdir, space):

    # Setup and run
    dirname = str(tmpdir)

    if space == 'mixed_FS':
        stepper, tmax, true_fieldA, true_fieldB = setup_limiters(dirname, space)
    else:
        stepper, tmax, true_field = setup_limiters(dirname, space)

    stepper.run(t=0, tmax=tmax)

    tol = 1e-9

    if space == 'mixed_FS':
        final_fieldA = stepper.fields('tracerA')
        final_fieldB = stepper.fields('tracerB')

        # Check tracer is roughly in the correct place
        assert norm(true_fieldA - final_fieldA) / norm(true_fieldA) < 0.05, \
            'Something is wrong with the DG space tracer using a mixed limiter'

        # Check tracer is roughly in the correct place
        assert norm(true_fieldB - final_fieldB) / norm(true_fieldB) < 0.05, \
            'Something is wrong with the DG1 equispaced tracer using a mixed limiter'

        # Check for no new overshoots in A
        assert np.max(final_fieldA.dat.data) <= np.max(true_fieldA.dat.data) + tol, \
            'Application of the DG space limiter in the mixed limiter has not prevented overshoots'

        # Check for no new undershoots in A
        assert np.min(final_fieldA.dat.data) >= np.min(true_fieldA.dat.data) - tol, \
            'Application of the DG space limiter in the mixed limiter has not prevented undershoots'

        # Check for no new overshoots in B
        assert np.max(final_fieldB.dat.data) <= np.max(true_fieldB.dat.data) + tol, \
            'Application of the DG1 equispaced limiter in the mixed limiter has not prevented overshoots'

        # Check for no new undershoots in B
        assert np.min(final_fieldB.dat.data) >= np.min(true_fieldB.dat.data) - tol, \
            'Application of the DG1 equispaced limiter in the mixed limiter has not prevented undershoots'

    else:
        final_field = stepper.fields('tracer')

        # Check tracer is roughly in the correct place
        assert norm(true_field - final_field) / norm(true_field) < 0.05, \
            'Something appears to have gone wrong with transport of tracer using a limiter'

        # Check for no new overshoots
        assert np.max(final_field.dat.data) <= np.max(true_field.dat.data) + tol, \
            'Application of limiter has not prevented overshoots'

        # Check for no new undershoots
        assert np.min(final_field.dat.data) >= np.min(true_field.dat.data) - tol, \
            'Application of limiter has not prevented undershoots'
