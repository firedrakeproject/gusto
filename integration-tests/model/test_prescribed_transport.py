"""
This tests the prescribed wind feature of the PrescribedTransport timestepper.
A tracer is transported with a time-varying wind and the computed solution is
compared with a true one to check that the transport is working correctly.
"""

from gusto import *
from firedrake import sin, cos, norm, pi, as_vector


def run(eqn, transport_scheme, state, tmax, f_end, prescribed_u):
    timestepper = PrescribedTransport(eqn, transport_scheme, state,
                                      prescribed_transporting_velocity=prescribed_u)
    timestepper.run(0, tmax)
    return norm(state.fields("f") - f_end) / norm(f_end)


def test_prescribed_transport_setup(tmpdir, tracer_setup):

    # Make mesh and state using routine from conftest
    geometry = "slice"
    setup = tracer_setup(tmpdir, geometry, degree=1)
    state = setup.state
    _, z = SpatialCoordinate(state.mesh)

    V = state.spaces("DG", "DG", 1)
    # Make equation
    eqn = AdvectionEquation(state, V, "f",
                            ufamily=setup.family, udegree=1)

    # Initialise fields
    def u_evaluation(t):
        return as_vector([2.0*cos(2*pi*t/setup.tmax),
                          sin(2*pi*t/setup.tmax)*sin(pi*z)])

    state.fields("f").interpolate(setup.f_init)
    state.fields("u").project(u_evaluation(Constant(0.0)))

    transport_scheme = SSPRK3(state)

    # Run and check error
    error = run(eqn, transport_scheme, state, setup.tmax,
                setup.f_init, u_evaluation)
    assert error < setup.tol, \
        'The transport error is greater than the permitted tolerance'
