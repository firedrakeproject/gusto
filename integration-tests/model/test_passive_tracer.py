"""
This tests the 'auxiliary_equations_and_schemes' option of the timeloop,
by transporting a passive tracer alongside the dynamical core.

The test uses the linear shallow-water equations, using initial conditions from
the Williamson 2 test case, so that the wind corresponds to a solid body
rotation. The tracer initial condition is a Gaussian bell, the same as is
used with the transport tests.
"""

from gusto import *
from firedrake import SpatialCoordinate, Function, norm
import pytest


def run_tracer(setup):

    # Get initial conditions from shared config
    domain = setup.domain
    mesh = domain.mesh
    output = setup.output

    x = SpatialCoordinate(mesh)
    H = 0.1
    parameters = ShallowWaterParameters(H=H)
    Omega = parameters.Omega
    g = parameters.g
    umax = setup.umax
    R = setup.radius
    fexpr = 2*Omega*x[2]/R

    # Need to create a new state containing parameters

    # Equations
    eqns = LinearShallowWaterEquations(domain, parameters, fexpr=fexpr)
    tracer_eqn = AdvectionEquation(domain, domain.spaces("DG"), "tracer")
    io = IO(domain, eqns, output=output)

    # Specify initial prognostic fields
    u0 = eqns.fields("u")
    D0 = eqns.fields("D")
    tracer0 = tracer_eqn.fields("tracer", D0.function_space())
    tracer_end = Function(D0.function_space())

    # Expressions for initial fields corresponding to Williamson 2 test case
    Dexpr = H - ((R * Omega * umax)*(x[2]*x[2]/(R*R))) / g
    u0.project(setup.uexpr)
    D0.interpolate(Dexpr)
    Dbar = Function(D0.function_space()).assign(H)
    tracer0.interpolate(setup.f_init)
    tracer_end.interpolate(setup.f_end)

    eqns.set_reference_profiles([('D', Dbar)])

    # set up transport schemes
    transport_schemes = [ForwardEuler(domain, "D")]

    # Set up tracer transport
    tracer_transport = [(tracer_eqn, SSPRK3(domain))]

    # build time stepper
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transport_schemes,
        auxiliary_equations_and_schemes=tracer_transport)

    stepper.run(t=0, tmax=setup.tmax)

    error = norm(tracer_eqn.fields("tracer") - tracer_end) / norm(tracer_end)

    return error


@pytest.mark.parametrize("geometry", ["sphere"])
def test_tracer_setup(tmpdir, geometry, tracer_setup):

    setup = tracer_setup(tmpdir, geometry)
    error = run_tracer(setup)

    assert error < setup.tol, 'The error in transporting ' + \
        'the tracer is greater than the permitted tolerance'
