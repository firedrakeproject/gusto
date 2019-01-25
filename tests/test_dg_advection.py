from firedrake import VectorFunctionSpace, as_vector, Function, errornorm
from gusto import *
import pytest


def run(state, equations, schemes, dt, tmax):

    timestepper = PrescribedAdvectionTimestepper(
        state, equations=equations, schemes=schemes)
    timestepper.run(0, dt=dt, tmax=tmax)
    return timestepper.state.fields("f")


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
@pytest.mark.parametrize("ibp", [IntegrateByParts.ONCE, IntegrateByParts.TWICE])
@pytest.mark.parametrize("scheme", ["ssprk", "im"])
def test_scalar_advection_dg(geometry, ibp, scheme, advection_setup):
    """
    This tests the DG advection discretisation for both scalar
    fields in 2D slice and spherical geometry.
    """

    setup = advection_setup(geometry)
    state = setup.state
    dt = setup.dt
    tmax = setup.tmax
    f_init = setup.f_init
    f_end = setup.f_end
    err = setup.err

    fspace = state.spaces("DG")
    f_end = Function(fspace).interpolate(f_end)

    eqns = []
    schemes = []

    eqns.append(("f", AdvectionEquation(state, "f", fspace, ibp=ibp)))
    f = state.fields("f")
    f.interpolate(f_init)
    if scheme == "ssprk":
        schemes.append(("f", SSPRK3()))
    elif scheme == "im":
        schemes.append(("f", ThetaMethod()))

    end_field = run(state, eqns, schemes, dt, tmax)
    assert errornorm(end_field, f_end) < err


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
@pytest.mark.parametrize("ibp", [IntegrateByParts.ONCE, IntegrateByParts.TWICE])
@pytest.mark.parametrize("scheme", ["ssprk", "im"])
def test_vector_advection_dg(geometry, ibp, scheme, advection_setup):
    """
    This tests the DG advection discretisation for vector
    fields in 2D slice and spherical geometry.
    """

    setup = advection_setup(geometry)
    state = setup.state
    dt = setup.dt
    tmax = setup.tmax
    f_init = setup.f_init
    f_end = setup.f_end
    err = setup.err

    fspace = VectorFunctionSpace(state.mesh, "DG", 1)
    # expression for vector initial and final conditions
    vec_expr = [0.]*state.mesh.geometric_dimension()
    vec_expr[0] = f_init
    f_init = as_vector(vec_expr)
    vec_end_expr = [0.]*state.mesh.geometric_dimension()
    vec_end_expr[0] = f_end
    vec_end_expr = as_vector(vec_end_expr)
    f_end = Function(fspace).interpolate(vec_end_expr)

    eqns = []
    schemes = []

    eqns.append(("f", AdvectionEquation(state, "f", fspace, ibp=ibp)))
    f = state.fields("f")
    f.interpolate(f_init)
    if scheme == "ssprk":
        schemes.append(("f", SSPRK3()))
    elif scheme == "im":
        schemes.append(("f", ThetaMethod()))

    end_field = run(state, eqns, schemes, dt, tmax)
    assert errornorm(end_field, f_end) < err

