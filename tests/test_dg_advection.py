from firedrake import norm, VectorFunctionSpace, as_vector
from gusto import *
import pytest


def run(state, advection_scheme, tmax, f_end):
    timestepper = PrescribedAdvection(state, advection_scheme)
    timestepper.run(0, tmax)
    return norm(state.fields("f") - f_end)


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
@pytest.mark.parametrize("scheme", ["ssprk", "implicit_midpoint", "RK4", "Heun"])
def test_dg_advection_scalar(tmpdir, geometry, equation_form, scheme,
                             tracer_setup):
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

    if scheme == "ssprk":
        advection_scheme = [(eqn, SSPRK3(state))]
    elif scheme == "implicit_midpoint":
        advection_scheme = [(eqn, ImplicitMidpoint(state))]
    elif scheme == "RK4":
        advection_scheme = [(eqn, RK4(state))]
    elif scheme == "Heun":
        advection_scheme = [(eqn, Heun(state))]
    assert run(state, advection_scheme, setup.tmax, setup.f_end) < setup.tol


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
@pytest.mark.parametrize("scheme", ["ssprk", "implicit_midpoint", "RK4", "Heun"])
def test_dg_advection_vector(tmpdir, geometry, equation_form, scheme,
                             tracer_setup):
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
    if scheme == "ssprk":
        advection_schemes = [(eqn, SSPRK3(state))]
    elif scheme == "implicit_midpoint":
        advection_schemes = [(eqn, ImplicitMidpoint(state))]
    elif scheme == "RK4":
         advection_schemes = [(eqn, RK4(state))]
    elif scheme == "Heun":
         advection_schemes = [(eqn, Heun(state))]
    f_end = as_vector((setup.f_end, *[0.]*(gdim-1)))
    assert run(state, advection_schemes, setup.tmax, f_end) < setup.tol
