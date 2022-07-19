"""
Tests the embedded DG transport scheme, checking that the field is transported
to the right place.
"""

from firedrake import norm
from gusto import *
import pytest


def run(state, transport_schemes, tmax, f_end):
    timestepper = PrescribedTransport(state, transport_schemes)
    timestepper.run(0, tmax)
    return norm(state.fields("f") - f_end)


@pytest.mark.parametrize("ibp", [IntegrateByParts.ONCE, IntegrateByParts.TWICE])
@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
@pytest.mark.parametrize("space", ["broken", "dg"])
def test_embedded_dg_advection_scalar(tmpdir, ibp, equation_form, space,
                                      tracer_setup):
    setup = tracer_setup(tmpdir, geometry="slice")
    state = setup.state
    V = state.spaces("theta", degree=1)

    if space == "broken":
        opts = EmbeddedDGOptions()
    elif space == "dg":
        opts = EmbeddedDGOptions(embedding_space=state.spaces("DG1", "DG", 1))

    if equation_form == "advective":
        eqn = AdvectionEquation(state, V, "f", ufamily=setup.family,
                                udegree=setup.degree, ibp=ibp)
    else:
        eqn = ContinuityEquation(state, V, "f", ufamily=setup.family,
                                 udegree=setup.degree, ibp=ibp)
    state.fields("f").interpolate(setup.f_init)
    state.fields("u").project(setup.uexpr)

    transport_schemes = [(eqn, SSPRK3(state, options=opts))]

    assert run(state, transport_schemes, setup.tmax, setup.f_end) < setup.tol
