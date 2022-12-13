"""
This tests the prescribed wind feature of the PrescribedTransport timestepper.
A tracer is transported with a time-varying wind and the computed solution is
compared with a true one to check that the transport is working correctly.
"""

from gusto import *
from firedrake import sin, cos, norm, pi, as_vector


def run(eqn, transport_scheme, io, tmax, f_end, prescribed_u):
    timestepper = PrescribedTransport(eqn, transport_scheme, io,
                                      prescribed_transporting_velocity=prescribed_u)
    timestepper.run(0, tmax)
    return norm(eqn.fields("f") - f_end) / norm(f_end)


def test_prescribed_transport_setup(tmpdir, tracer_setup):

    # Make domain using routine from conftest
    geometry = "slice"
    setup = tracer_setup(tmpdir, geometry, degree=1)
    domain = setup.domain
    _, z = SpatialCoordinate(domain.mesh)

    V = domain.spaces("DG")
    # Make equation
    eqn = AdvectionEquation(domain, V, "f")
    io = IO(domain, eqn, output=setup.output)

    # Initialise fields
    def u_evaluation(t):
        return as_vector([2.0*cos(2*pi*t/setup.tmax),
                          sin(2*pi*t/setup.tmax)*sin(pi*z)])

    eqn.fields("f").interpolate(setup.f_init)
    eqn.fields("u").project(u_evaluation(Constant(0.0)))

    transport_scheme = SSPRK3(domain, io)

    # Run and check error
    error = run(eqn, transport_scheme, io, setup.tmax,
                setup.f_init, u_evaluation)
    assert error < setup.tol, \
        'The transport error is greater than the permitted tolerance'
