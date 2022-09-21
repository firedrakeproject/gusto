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
    state = setup.state
    mesh = state.mesh
    dt = state.dt
    output = state.output

    x = SpatialCoordinate(state.mesh)
    H = 0.1
    parameters = ShallowWaterParameters(H=H)
    Omega = parameters.Omega
    g = parameters.g
    umax = setup.umax
    R = setup.radius
    fexpr = 2*Omega*x[2]/R

    # Need to create a new state containing parameters
    state = State(mesh, dt=dt, output=output, parameters=parameters)

    # Equations
    eqns = LinearShallowWaterEquations(state, setup.family,
                                       setup.degree, fexpr=fexpr)
    tracer_eqn = AdvectionEquation(state, state.spaces("DG"), "tracer")

    # Specify initial prognostic fields
    u0 = state.fields("u")
    D0 = state.fields("D")
    tracer0 = state.fields("tracer", D0.function_space())
    tracer_end = Function(D0.function_space())

    # Expressions for initial fields corresponding to Williamson 2 test case
    Dexpr = H - ((R * Omega * umax)*(x[2]*x[2]/(R*R))) / g
    u0.project(setup.uexpr)
    D0.interpolate(Dexpr)
    tracer0.interpolate(setup.f_init)
    tracer_end.interpolate(setup.f_end)

    # set up transport schemes
    transport_schemes = [ForwardEuler(state, "D")]

    # Set up tracer transport
    tracer_transport = [(tracer_eqn, SSPRK3(state))]

    # build time stepper
    stepper = SemiImplicitQuasiNewton(
        state, eqns, transport_schemes,
        auxiliary_equations_and_schemes=tracer_transport)

    stepper.run(t=0, tmax=setup.tmax)

    error = norm(state.fields("tracer") - tracer_end) / norm(tracer_end)

    return error


@pytest.mark.parametrize("geometry", ["sphere"])
def test_tracer_setup(tmpdir, geometry, tracer_setup):

    setup = tracer_setup(tmpdir, geometry)
    error = run_tracer(setup)

    assert error < setup.tol, 'The error in transporting ' + \
        'the tracer is greater than the permitted tolerance'
