from firedrake import norm
from gusto import *
import pytest


def run(eqn, transport_scheme, state, tmax, f_end):
    timestepper = PrescribedTransport(eqn, transport_scheme, state)
    timestepper.run(0, tmax)
    return norm(state.fields("f") - f_end) / norm(f_end)


@pytest.mark.parametrize("scheme", ["ssprk", "implicit_midpoint",
                                    "RK4", "Heun", "BDF2"])
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
        transport_scheme = SSPRK3(state)
    elif scheme == "implicit_midpoint":
        transport_scheme = ImplicitMidpoint(state)
    elif scheme == "RK4":
        transport_scheme = RK4(state)
    elif scheme == "Heun":
        transport_scheme = Heun(state)
    elif scheme == "BDF2":
        transport_scheme = BDF2(state)
    assert run(eqn, transport_scheme, state, setup.tmax, setup.f_end) < setup.tol
