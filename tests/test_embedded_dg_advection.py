from firedrake import norm
from gusto import *
import pytest

def run(state, advection_schemes, tmax, f_end):
    timestepper = Advection(state, advection_schemes)
    timestepper.run(0, tmax)
    return norm(state.fields("f") - f_end)


@pytest.mark.parametrize("ibp", [IntegrateByParts.ONCE, IntegrateByParts.TWICE])
@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
@pytest.mark.parametrize("space", ["broken", "dg"])
def test_embedded_dg_advection_scalar(tmpdir, ibp, equation_form, space,
                                      tracer_setup):
    setup = tracer_setup(tmpdir, geometry="slice")
    state = setup.state
    V = state.spaces("HDiv_v")
    f = state.fields("f", V)
    f.interpolate(setup.f_init)

    if space == "broken":
        opts = EmbeddedDGOptions()
    elif space == "dg":
        opts = EmbeddedDGOptions(embedding_space=state.spaces("DG"))    

    eqn = EmbeddedDGAdvection(state, V, ibp=ibp,
                              equation_form=equation_form, options=opts)
    advection_schemes = [("f", SSPRK3(state, f, eqn))]

    assert run(state, advection_schemes, setup.tmax, setup.f_end) < setup.tol
