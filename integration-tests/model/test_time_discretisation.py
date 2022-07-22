from firedrake import norm
from gusto import *
import pytest


def run(state, transport_scheme, tmax, f_end):
    timestepper = PrescribedTransport(state, transport_scheme)
    timestepper.run(0, tmax)
    return norm(state.fields("f") - f_end)


@pytest.mark.parametrize("scheme", ["ssprk", "implicit_midpoint",
                                    "RK4", "Heun"])
def test_time_discretisation(tmpdir, scheme, tracer_setup):
    geometry = "sphere"
    setup = tracer_setup(tmpdir, geometry)
    state = setup.state
    V = state.spaces("DG", "DG", 1)

    eqn = AdvectionEquation(state, V, "f", ufamily=setup.family,
                            udegree=setup.degree)

    state.fields("f").interpolate(setup.f_init)
    state.fields("u").project(setup.uexpr)

    if scheme == "ssprk":
        transport_scheme = [(eqn, SSPRK3(state))]
    elif scheme == "implicit_midpoint":
        transport_scheme = [(eqn, ImplicitMidpoint(state))]
    elif scheme == "RK4":
        transport_scheme = [(eqn, RK4(state))]
    elif scheme == "Heun":
        transport_scheme = [(eqn, Heun(state))]
    assert run(state, transport_scheme, setup.tmax, setup.f_end) < setup.tol
