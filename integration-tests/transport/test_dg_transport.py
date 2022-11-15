"""
Tests the DG upwind transport scheme, with various options. This tests that the
field is transported to the correct place.
"""

from firedrake import norm, VectorFunctionSpace, as_vector
from gusto import *
import pytest


def run(eqn, transport_scheme, state, tmax, f_end):
    timestepper = PrescribedTransport(eqn, transport_scheme, state)
    timestepper.run(0, tmax)
    return norm(state.fields("f") - f_end) / norm(f_end)


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

    transport_scheme = SSPRK3(state)
    error = run(eqn, transport_scheme, state, setup.tmax, setup.f_end)
    assert error < setup.tol, \
        'The transport error is greater than the permitted tolerance'


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
def test_dg_transport_vector(tmpdir, geometry, equation_form, tracer_setup):
    setup = tracer_setup(tmpdir, geometry)
    state = setup.state
    gdim = state.mesh.geometric_dimension()
    f_init = as_vector([setup.f_init]*gdim)
    V = VectorFunctionSpace(state.mesh, "DG", 1)
    if equation_form == "advective":
        eqn = AdvectionEquation(state, V, "f", ufamily=setup.family,
                                udegree=setup.degree)
    else:
        eqn = ContinuityEquation(state, V, "f", ufamily=setup.family,
                                 udegree=setup.degree)
    state.fields("f").interpolate(f_init)
    state.fields("u").project(setup.uexpr)
    transport_schemes = SSPRK3(state)
    f_end = as_vector([setup.f_end]*gdim)
    error = run(eqn, transport_schemes, state, setup.tmax, f_end)
    assert error < setup.tol, \
        'The transport error is greater than the permitted tolerance'
