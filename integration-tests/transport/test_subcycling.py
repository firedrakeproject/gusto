"""
This tests transport using the subcycling option. The computed solution is
compared with a true one to check that the transport is working correctly.
"""

from gusto import *
from firedrake import norm
import pytest


def run(eqn, transport_scheme, io, tmax, f_end):
    timestepper = PrescribedTransport(eqn, transport_scheme, io)
    timestepper.run(0, tmax)
    return norm(eqn.fields("f") - f_end) / norm(f_end)


@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
def test_subcyling(tmpdir, equation_form, tracer_setup):
    geometry = "slice"
    setup = tracer_setup(tmpdir, geometry)
    domain = setup.domain
    V = domain.spaces("DG")
    if equation_form == "advective":
        eqn = AdvectionEquation(domain, V, "f")
    else:
        eqn = ContinuityEquation(domain, V, "f")

    io = IO(domain, eqn, dt=setup.dt, output=setup.output)

    eqn.fields("f").interpolate(setup.f_init)
    eqn.fields("u").project(setup.uexpr)

    transport_scheme = SSPRK3(domain, io, subcycles=2)
    error = run(eqn, transport_scheme, io, setup.tmax, setup.f_end)
    assert error < setup.tol, \
        'The transport error is greater than the permitted tolerance'
