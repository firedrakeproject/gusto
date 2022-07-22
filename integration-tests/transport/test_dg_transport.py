"""
Tests the DG upwind transport scheme, with various options. This tests that the
field is transported to the correct place.
"""

from firedrake import norm, VectorFunctionSpace, as_vector
from gusto import *
import pytest


def run(state, transport_scheme, tmax, f_end):
    timestepper = PrescribedTransport(state, transport_scheme)
    timestepper.run(0, tmax)
    return norm(state.fields("f") - f_end)


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
def test_dg_transport_scalar(tmpdir, geometry, equation_form, tracer_setup):
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

    transport_scheme = [(eqn, SSPRK3(state))]
    assert run(state, transport_scheme, setup.tmax, setup.f_end) < setup.tol


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
def test_dg_transport_vector(tmpdir, geometry, equation_form, tracer_setup):
    setup = tracer_setup(tmpdir, geometry)
    state = setup.state
    gdim = state.mesh.geometric_dimension()
    f_init = as_vector((setup.f_init, *[0.]*(gdim-1)))
    V = VectorFunctionSpace(state.mesh, "DG", 1)
    if equation_form == "advective":
        eqn = AdvectionEquation(state, V, "f", ufamily=setup.family,
                                udegree=setup.degree)
    else:
        eqn = ContinuityEquation(state, V, "f", ufamily=setup.family,
                                 udegree=setup.degree)
    state.fields("f").interpolate(f_init)
    state.fields("u").project(setup.uexpr)
    transport_schemes = [(eqn, SSPRK3(state))]
    f_end = as_vector((setup.f_end, *[0.]*(gdim-1)))
    assert run(state, transport_schemes, setup.tmax, f_end) < setup.tol
