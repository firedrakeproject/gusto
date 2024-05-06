"""
This tests that the conservative transport is correctly working with a physics
scheme. The transport equations require the tracer to be multiplied by the density 
(through the 'mass_weighted' label) whilst the physics equation does not. 
This checks that we correctly solve a problem where some tracer terms have 
a mass_weighted label and some do not.
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
    # Source is a constant, but constrained to a box in the bottom left corner
    # of size 1-by-1, such that the total ash value
    # should be equal to tmax = 0.5.
    basic_expression = conditional(x<1.0, conditional(y<1.0, -Constant(1.0), Constant(0.0)), Constant(0.0))

    physics_schemes = [(SourceSink(eqn, 'ash', basic_expression), SSPRK3(domain))]

    # Time stepper
    stepper = SplitPrescribedTransport(eqn, SSPRK3(domain, increment_form=False),
                                       io, transport_method,
                                       physics_schemes=physics_schemes)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    rho0 = stepper.fields("rho_d")
    ash0 = stepper.fields("ash")

    # Set a spatially varying density field and no ash
    rho0.interpolate(1000.0*sin(pi*x/L)*sin(pi*y/L)+1000.0)
    ash0.interpolate(Constant(0.0))

    # Constant wind
    u = stepper.fields("u")
    u.project(as_vector([0.5, 0.5]))

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

    tol = 1e-3
    assert np.abs(final_total_ash - 0.5) < tol, \
        "Conservative transport did not correctly implement the Source physics"
