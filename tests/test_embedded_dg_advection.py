from firedrake import Function, errornorm
from gusto import *
import pytest


def run(state, equations, schemes, dt, tmax):

    timestepper = PrescribedAdvectionTimestepper(
        state, equations=equations, schemes=schemes)
    timestepper.run(0, dt=dt, tmax=tmax)
    return timestepper.state.fields("f")


@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
@pytest.mark.parametrize("ibp", [IntegrateByParts.ONCE, IntegrateByParts.TWICE])
@pytest.mark.parametrize("space", ["broken", "dg"])
def test_advection_embedded_dg(equation_form, ibp, space, advection_setup):
    """
    This tests the embedded DG advection scheme for scalar fields
    in slice geometry.
    """
    setup = advection_setup("slice")
    state = setup.state
    dt = setup.dt
    tmax = setup.tmax
    f_init = setup.f_init
    f_end = setup.f_end
    err = setup.err

    fspace = state.spaces("HDiv_v")
    f_end = Function(fspace).interpolate(f_end)

    eqns = []
    schemes = []
    opts = {"broken": EmbeddedDGOptions(),
            "dg": EmbeddedDGOptions(embedding_space=state.spaces("DG"))}

    if equation_form == "advective":
        eqns.append(("f", AdvectionEquation(state, "f", fspace, ibp=ibp)))
    else:
        eqns.append(("f", ContinuityEquation(state, "f", fspace, ibp=ibp)))

    f = state.fields("f")
    f.interpolate(f_init)
    schemes.append(("f", SSPRK3(options=opts[space])))

    end_field = run(state, eqns, schemes, dt, tmax)
    assert(errornorm(end_field, f_end) < err)
