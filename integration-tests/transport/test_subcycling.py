"""
This tests transport using the subcycling option. The computed solution is
compared with a true one to check that the transport is working correctly.
"""

from gusto import *
from firedrake import norm
import pytest


def run(timestepper, tmax, f_end):
    timestepper.run(0, tmax)
    return norm(timestepper.fields("f") - f_end) / norm(f_end)


@pytest.mark.parametrize("subcycling", ["fixed", "adaptive"])
def test_subcyling(tmpdir, subcycling, tracer_setup):
    geometry = "slice"
    setup = tracer_setup(tmpdir, geometry)
    domain = setup.domain
    V = domain.spaces("DG")
    eqn = AdvectionEquation(domain, V, "f")

    if subcycling == "fixed":
        transport_scheme = SSPRK3(domain, subcycles=2)
    elif subcycling == "adaptive":
        transport_scheme = SSPRK3(domain, subcycle_by=0.25)
    transport_method = DGUpwind(eqn, "f")

    timestepper = PrescribedTransport(eqn, transport_scheme, setup.io, transport_method)

    # Initial conditions
    timestepper.fields("f").interpolate(setup.f_init)
    timestepper.fields("u").project(setup.uexpr)

    error = run(timestepper, setup.tmax, setup.f_end)
    assert error < setup.tol, \
        'The transport error is greater than the permitted tolerance'
