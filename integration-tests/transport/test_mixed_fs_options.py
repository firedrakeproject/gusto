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


def setup_limiters(dirname, space_A, space_B):

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
    degree = 0 if space_A in ['DG0', 'Vtheta_degree_0'] or space_B in ['DG0', 'Vtheta_degree_0'] else 1

    domain = Domain(mesh, dt, family="CG", degree=degree)
    
    # Transporting velocity space
    V = domain.spaces('HDiv')
    
    # Tracer A spaces
    if space_A == 'DG0':
        VA = domain.spaces('DG')
        VCG1_A = FunctionSpace(mesh, 'CG', 1)
        VDG1_A = domain.spaces('DG1_equispaced')
        space_A_string = 'DG'
    elif space_A == 'DG1':
        VA = domain.spaces('DG')
        space_A_string = 'DG'
    elif space_A == 'DG1_equispaced':
        VA = domain.spaces('DG1_equispaced')
        space_A_string = 'DG1_equispaced'
    elif space_A == 'Vtheta_degree_0':
        VA = domain.spaces('theta')
        VCG1_A = FunctionSpace(mesh, 'CG', 1)
        VDG1_A = domain.spaces('DG1_equispaced')
        space_A_string = 'theta'
    elif space_A == 'Vtheta_degree_1':
        VA = domain.spaces('theta')
        space_A_string = 'theta'
    else:
        raise NotImplementedError
    
    # Tracer B spaces
    if space_B == 'DG0':
        VB = domain.spaces('DG')
        VCG1_B = FunctionSpace(mesh, 'CG', 1)
        VDG1_B = domain.spaces('DG1_equispaced')
        space_B_string = 'DG'
    elif space_B == 'DG1':
        VB = domain.spaces('DG')
        space_B_string = 'DG'
    elif space_B == 'DG1_equispaced':
        VB = domain.spaces('DG1_equispaced')
        space_B_string = 'DG1_equispaced'
    elif space_B == 'Vtheta_degree_0':
        VB = domain.spaces('theta')
        VCG1_B = FunctionSpace(mesh, 'CG', 1)
        VDG1_B = domain.spaces('DG1_equispaced')
        space_B_string = 'theta'
    elif space_B == 'Vtheta_degree_1':
        VB = domain.spaces('theta')
        space_B_string = 'theta'
    else:
        raise NotImplementedError
        
    Vpsi = domain.spaces('H1')     
        
    tracerA = ActiveTracer(name='tracerA', space=space_A_string,
                               variable_type=TracerVariableType.mixing_ratio,
                               transport_eqn=TransportEquationType.advective)
    
    tracerB = ActiveTracer(name='tracerB', space=space_B_string,
                               variable_type=TracerVariableType.mixing_ratio,
                               transport_eqn=TransportEquationType.advective)
        
        
    tracers = [tracerA, tracerB]

    eqn = CoupledTransportEquation(domain, active_tracers=tracers, Vu=V)
    output = OutputParameters(dirname=dirname+'/limiters', dumpfreq=1,
                                  dumplist=['u', 'tracerA', 'tracerB', 'true_tracerA', 'true_tracerB'])
    
    io = IO(domain, output)

    # ------------------------------------------------------------------------ #
    # Set up transport scheme with options
    # ------------------------------------------------------------------------ #

    suboptions = {}
    sublimiters = {}

    # Options and limiters for tracer_A

    if space_A in ['DG0', 'Vtheta_degree_0']:
        suboptions.update({'tracerA': RecoveryOptions(embedding_space=VDG1_A,
                                              recovered_space=VCG1_A,
                                              project_low_method='recover',
                                              boundary_method=BoundaryMethod.taylor)})
                                   
        sublimiters.update({'tracerA': VertexBasedLimiter(VDG1_A)})
        
    elif space_A == 'DG1':
        sublimiters.update({'tracerA': DG1Limiter(VA)})

    elif space_A == 'DG1_equispaced':
        sublimiters.update({'tracerA': VertexBasedLimiter(VA)})

    elif space_A == 'Vtheta_degree_1':
        opts = EmbeddedDGOptions()
        sublimiters.update({'tracerA': ThetaLimiter(VA)})

    else:
        raise NotImplementedError
        
    # Options and limiters for tracer_B

    if space_B in ['DG0', 'Vtheta_degree_0']:
        recover_opts = RecoveryOptions(embedding_space=VDG1_B,
                               recovered_space=VCG1_B,
                               project_low_method='recover',
                               boundary_method=BoundaryMethod.taylor)
                                   
        sublimiters.update({'tracerB': VertexBasedLimiter(VDG1_B)})
        
    elif space_B == 'DG1':
        sublimiters.update({'tracerB': DG1Limiter(VB)})

    elif space_B == 'DG1_equispaced':
        sublimiters.update({'tracerB': VertexBasedLimiter(VB)})

    elif space_B == 'Vtheta_degree_1':
        opts = EmbeddedDGOptions()
        sublimiters.update({'tracerB': ThetaLimiter(VB)})

    else:
        raise NotImplementedError

    opts = MixedOptions(eqn, suboptions)
    MixedLimiter = MixedFSLimiter(eqn, sublimiters)

    # Give the scheme for the coupled transport
    transport_schemes = SSPRK3(domain, options=opts, limiter=MixedLimiter)
    
    # DG Upwind transport for both tracers:
    transport_method = [DGUpwind(eqn, 'tracerA'), DGUpwind(eqn, 'tracerB')]
    
    # Build time stepper
    stepper = PrescribedTransport(eqn, transport_schemes, io, transport_method)

    # ------------------------------------------------------------------------ #
    # Initial condition
    # ------------------------------------------------------------------------ #

    tracerA_0 = stepper.fields('tracerA', space=VA)
    tracerB_0 = stepper.fields('tracerB', space=VB)
    true_fieldA = stepper.fields('true_tracerA', space=VA)
    true_fieldB = stepper.fields('true_tracerB', space=VB)
    
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
            tracerA_0.interpolate(Constant(tracer_min) + expr_1 + expr_2)
            tracerB_0.interpolate(Constant(tracer_min) + expr_1 + expr_2)
        elif i == 1:
            true_fieldA.interpolate(Constant(tracer_min) + expr_1 + expr_2)
            true_fieldB.interpolate(Constant(tracer_min) + expr_1 + expr_2)
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

    return stepper, tmax, true_fieldA, true_fieldB


@pytest.mark.parametrize('space_A', ['Vtheta_degree_0', 'Vtheta_degree_1', 'DG0',
                                   'DG1', 'DG1_equispaced'])
#Remove Dg1-dg1 and other easy ones after debugging
@pytest.mark.parametrize('space_B', ['Vtheta_degree_0'])#, 'Vtheta_degree_1', 'DG0',
                                    #'DG1', 'DG1_equispaced'])


def test_limiters(tmpdir, space_A, space_B):

    # Setup and run
    dirname = str(tmpdir)

    stepper, tmax, true_fieldA, true_fieldB = setup_limiters(dirname, space_A, space_B)

    stepper.run(t=0, tmax=tmax)

    tol = 1e-9

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

    