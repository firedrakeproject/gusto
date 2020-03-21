from firedrake import norm, FunctionSpace, VectorFunctionSpace, as_vector
from gusto import *
import pytest


def run(state, advection_scheme, tmax, f_end):
    timestepper = PrescribedAdvection(state, advection_scheme)
    timestepper.run(0, tmax)
    return norm(state.fields("f") - f_end)


@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
@pytest.mark.parametrize("scheme", ["ssprk", "implicit_midpoint"])
@pytest.mark.parametrize("space", ["CG", "HDiv_v"])
def test_supg_advection_scalar(tmpdir, equation_form, scheme, space,
                               tracer_setup):

    setup = tracer_setup(tmpdir, geometry="slice")
    state = setup.state

    if space == "CG":
        V = FunctionSpace(state.mesh, "CG", 1)
        ibp = IntegrateByParts.NEVER
    else:
        V = state.spaces(space)
        ibp = IntegrateByParts.TWICE
    f = state.fields("f", V)
    f.interpolate(setup.f_init)

    opts = SUPGOptions()

    if equation_form == "advective":
        eqn = AdvectionEquation(state, V, "f", ibp=ibp)
    else:
        eqn = ContinuityEquation(state, V, "f", ibp=ibp)
    if scheme == "ssprk":
        advection_scheme = [(eqn, SSPRK3(state, options=opts))]
    elif scheme == "implicit_midpoint":
        advection_scheme = [(eqn, ImplicitMidpoint(state, options=opts))]

    assert run(state, advection_scheme, setup.tmax, setup.f_end) < setup.tol


@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
@pytest.mark.parametrize("scheme", ["ssprk", "implicit_midpoint"])
@pytest.mark.parametrize("space", ["CG", "HDiv"])
def test_supg_advection_vector(tmpdir, equation_form, scheme, space,
                               tracer_setup):

    setup = tracer_setup(tmpdir, geometry="slice")
    state = setup.state
    tmax = setup.tmax
    tol = setup.tol

    gdim = state.mesh.geometric_dimension()
    f_init = as_vector((setup.f_init, *[0.]*(gdim-1)))
    if space == "CG":
        V = VectorFunctionSpace(state.mesh, "CG", 1)
        f = state.fields("f", V)
        f.interpolate(f_init)
        ibp = IntegrateByParts.NEVER
    else:
        V = state.spaces(space)
        f = state.fields("f", V)
        f.project(f_init)
        ibp = IntegrateByParts.TWICE

    opts = SUPGOptions()

    if equation_form == "advective":
        eqn = AdvectionEquation(state, V, "f", ibp=ibp)
    else:
        eqn = ContinuityEquation(state, V, "f", ibp=ibp)
    if scheme == "ssprk":
        advection_scheme = [(eqn, SSPRK3(state, options=opts))]
    elif scheme == "implicit_midpoint":
        advection_scheme = [(eqn, ImplicitMidpoint(state, options=opts))]

    f_end = as_vector((setup.f_end, *[0.]*(gdim-1)))
    assert run(state, advection_scheme, tmax, f_end) < tol
