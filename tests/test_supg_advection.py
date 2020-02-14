from firedrake import norm, FunctionSpace, VectorFunctionSpace, as_vector
from gusto import *
import pytest

def run(state, advection_schemes, tmax, f_end):
    timestepper = Advection(state, advection_schemes)
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

    eqn = SUPGAdvection(state, V, ibp=ibp, equation_form=equation_form)
    if scheme == "ssprk":
        advection_schemes = [("f", SSPRK3(state, f, eqn))]
    elif scheme == "implicit_midpoint":
        advection_schemes = [("f", ImplicitMidpoint(state, f, eqn))]

    assert run(state, advection_schemes, setup.tmax, setup.f_end) < setup.tol


@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
@pytest.mark.parametrize("scheme", ["ssprk", "implicit_midpoint"])
@pytest.mark.parametrize("space", ["CG", "HDiv"])
def test_supg_advection_vector(tmpdir, equation_form, scheme, space,
                               tracer_setup):

    setup = tracer_setup(tmpdir, geometry="slice")
    state = setup.state

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

    eqn = SUPGAdvection(state, V, ibp=ibp, equation_form=equation_form)
    if scheme == "ssprk":
        advection_schemes = [("f", SSPRK3(state, f, eqn))]
    elif scheme == "implicit_midpoint":
        advection_schemes = [("f", ImplicitMidpoint(state, f, eqn))]

    f_end = as_vector((setup.f_end, *[0.]*(gdim-1)))
    assert run(state, advection_schemes, setup.tmax, f_end) < setup.tol
