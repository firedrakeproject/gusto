from firedrake import Function, errornorm
from gusto import *
import pytest


def run(state, equations_schemes, dt, tmax):

    timestepper = PrescribedAdvectionTimestepper(
        state, equations_schemes)
    timestepper.run(0, dt=dt, tmax=tmax)
    return timestepper.state.fields("f")


@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
@pytest.mark.parametrize("ibp", [IntegrateByParts.ONCE, IntegrateByParts.TWICE])
@pytest.mark.parametrize("space", ["broken", "dg"])
def test_advection_embedded_dg(tmpdir, equation_form, ibp, space,
                               tracer_setup):
    """
    This tests the embedded DG advection scheme for scalar fields
    in slice geometry.
    """
    setup = tracer_setup(tmpdir, "slice")
    state = setup.state
    dt = setup.dt
    tmax = setup.tmax
    f_init = setup.f_init
    f_end = setup.f_end
    err = setup.err

    fspace = state.spaces("HDiv_v")
    f_end = Function(fspace).interpolate(f_end)

    opts = {"broken": EmbeddedDGOptions(),
            "dg": EmbeddedDGOptions(embedding_space=state.spaces("DG"))}

    if equation_form == "advective":
        equation = AdvectionEquation(state, fspace, "f", ibp=ibp)
    else:
        equation = ContinuityEquation(state, fspace, "f", ibp=ibp)

    f = state.fields("f")
    f.interpolate(f_init)
    equations_schemes = [(equation, SSPRK3(options=opts[space]))]

    f = run(state, equations_schemes, dt, tmax)
    assert(errornorm(f, f_end) < err)
