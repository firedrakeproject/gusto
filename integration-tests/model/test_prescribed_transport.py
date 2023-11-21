"""
This tests the prescribed wind feature of the PrescribedTransport and 
SplitPresribedTransport (with no physics schemes) timesteppers.
A tracer is transported with a time-varying wind and the computed solution is
compared with a true one to check that the transport is working correctly.
"""

from gusto import *
from firedrake import sin, cos, norm, pi, as_vector
import pytest


def run(timestepper, tmax, f_end):
    timestepper.run(0, tmax)
    return norm(timestepper.fields("f") - f_end) / norm(f_end)

@pytest.mark.parametrize('timestep_method', ['prescribed', 'split_prescribed'])
def test_prescribed_transport_setup(tmpdir, tracer_setup, timestep_method):

    # Make domain using routine from conftest
    geometry = "slice"
    setup = tracer_setup(tmpdir, geometry, degree=1)
    domain = setup.domain
    _, z = SpatialCoordinate(domain.mesh)

    V = domain.spaces("DG")
    # Make equation
    eqn = AdvectionEquation(domain, V, "f")

    # Initialise fields
    def u_evaluation(t):
        return as_vector([2.0*cos(2*pi*t/setup.tmax),
                          sin(2*pi*t/setup.tmax)*sin(pi*z)])

    transport_scheme = SSPRK3(domain)

    if timestep_method == 'prescribed':
        transport_method = DGUpwind(eqn, 'f')
        timestepper = PrescribedTransport(eqn, transport_scheme, setup.io,
                                          transport_method,
                                          prescribed_transporting_velocity=u_evaluation)
    elif timestep_method == 'split_prescribed':
        transport_method = [DGUpwind(eqn, 'f')]
        timestepper = SplitPrescribedTransport(eqn, transport_scheme, setup.io,
                                               transport_method,
                                               prescribed_transporting_velocity=u_evaluation)
    else:
        raise NotImplementedError


    # Initial conditions
    timestepper.fields("f").interpolate(setup.f_init)
    timestepper.fields("u").project(u_evaluation(Constant(0.0)))

    # Run and check error
    error = run(timestepper, setup.tmax, setup.f_init)
    assert error < setup.tol, \
        'The transport error is greater than the permitted tolerance'
