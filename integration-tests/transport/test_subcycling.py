"""
This tests transport using the subcycling option. The computed solution is
compared with a true one to check that the transport is working correctly.
"""

from gusto import *
from firedrake import norm
import pytest


def run(state, transport_scheme, tmax, f_end):
    timestepper = PrescribedTransport(state, transport_scheme)
    timestepper.run(0, tmax)
    return norm(state.fields("f") - f_end) / norm(f_end)


@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
def test_subcyling(tmpdir, equation_form, tracer_setup):
    geometry = "slice"
    setup = tracer_setup(tmpdir, geometry)
    state = setup.state
    V = state.spaces("DG", "DG", 1)
    if equation_form == "advective":
        eqn = AdvectionEquation(state, V, "f", ufamily=setup.family,
                                udegree=setup.degree)
    else:
        eqn = ContinuityEquation(state, V, "f", ufamily=setup.family,
                                 udegree=setup.degree)
    state.fields("f").interpolate(setup.f_init)
    state.fields("u").project(setup.uexpr)

    transport_scheme = [(eqn, SSPRK3(state, subcycles=2))]
    error = run(state, transport_scheme, setup.tmax, setup.f_end)
    assert error < setup.tol, \
        'The transport error is greater than the permitted tolerance'
