"""
This tests the source/sink
"""

from gusto import *
from firedrake import (as_vector, PeriodicSquareMesh, SpatialCoordinate,
                       assemble, Constant)


def run_conservative_transport_with_physics(dirname):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # set up mesh and domain
    L = 10
    nx = 10
    mesh = PeriodicSquareMesh(nx, nx, L, quadrilateral=True)
    dt = 0.1
    tmax = 5*dt
    domain = Domain(mesh, dt, "RTCF", 1)
    x, y = SpatialCoordinate(mesh)

    rho_d_space = 'DG'
    ash_space = 'DG'

    ash = ActiveTracer(name='ash', space=ash_space,
                       variable_type=TracerVariableType.mixing_ratio,
                       transport_eqn=TransportEquationType.tracer_conservative,
                       density_name='rho_d')

    rho_d = ActiveTracer(name='rho_d', space=rho_d_space,
                         variable_type=TracerVariableType.density,
                         transport_eqn=TransportEquationType.conservative)

    tracers = [ash, rho_d]

    eqn = CoupledTransportEquation(domain, active_tracers=tracers)

    # I/O
    output = OutputParameters(dirname=dirname+"/conservative_transport_with_physics",
                              dumpfreq=1)
    diagnostic_fields = [CourantNumber()]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)
    transport_method = [DGUpwind(eqn, "rho_d"), DGUpwind(eqn, "ash")]

    # Physics scheme --------------------------------------------------------- #
    # Source is a constant, so that the total ash value
    # should be L**2 * T = 50.
    basic_expression = -Constant(1.0)

    physics_schemes = [(SourceSink(eqn, 'ash', basic_expression), SSPRK3(domain))]

    # Time stepper
    stepper = SplitPrescribedTransport(eqn, SSPRK3(domain), io, transport_method,
                                  physics_schemes=physics_schemes)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    rho0 = stepper.fields("rho_d")
    ash0 = stepper.fields("ash")
    
    # Set an initial constant density field
    rho0.interpolate(Constant(1.0))
    ash0.interpolate(Constant(0.0))

    # Constant wind
    u = stepper.fields("u")
    u.project(as_vector([1.0, 0.0]))

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=tmax)
    return stepper


def test_conservative_transport_with_physics(tmpdir):
    dirname = str(tmpdir)
    stepper = run_conservative_transport_with_physics(dirname)
    final_ash = stepper.fields("ash")

    final_total_ash = assemble(final_ash*dx)
    print(final_total_ash)

    tol = 1e-5
    assert np.abs(final_total_ash - 50.0) < tol, \
            "Conservative transport did not correctly implement the Source physics"
