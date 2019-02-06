from firedrake import VectorFunctionSpace, as_vector, Function, errornorm
from gusto import *
import pytest


def run(setup, ibp, scheme, vector):

    state = setup.state
    dt = setup.dt
    tmax = setup.tmax
    f_init = setup.f_init
    f_end = setup.f_end

    if vector:
        fspace = VectorFunctionSpace(state.mesh, "DG", 1)
        # expression for vector initial and final conditions
        vec_expr = [0.]*state.mesh.geometric_dimension()
        vec_expr[0] = f_init
        f_init = as_vector(vec_expr)
        vec_end_expr = [0.]*state.mesh.geometric_dimension()
        vec_end_expr[0] = f_end
        vec_end_expr = as_vector(vec_end_expr)
        f_end = Function(fspace).interpolate(vec_end_expr)
    else:
        fspace = state.spaces("DG")
        f_end = Function(fspace).interpolate(f_end)

    equations = [("f", AdvectionEquation(state, fspace, "f", ibp=ibp))]
    f = state.fields("f", space=fspace)
    f.interpolate(f_init)

    if scheme == "ssprk":
        schemes = [("f", SSPRK3())]
    elif scheme == "im":
        schemes = [("f", ImplicitMidpoint())]

    timestepper = PrescribedAdvectionTimestepper(
        state, equations=equations, schemes=schemes)
    timestepper.run(0, dt=dt, tmax=tmax)

    return timestepper.state.fields("f"), f_end


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
@pytest.mark.parametrize("ibp", [IntegrateByParts.ONCE, IntegrateByParts.TWICE])
@pytest.mark.parametrize("scheme", ["ssprk"])
def test_scalar_advection_dg(geometry, ibp, scheme, tracer_setup):
    """
    This tests the DG advection discretisation for scalar
    fields in 2D slice and spherical geometry.
    """

    setup = tracer_setup(geometry)
    f, f_end = run(setup, ibp, scheme, vector=False)
    assert errornorm(f, f_end) < setup.err


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
@pytest.mark.parametrize("ibp", [IntegrateByParts.ONCE, IntegrateByParts.TWICE])
@pytest.mark.parametrize("scheme", ["ssprk"])
def test_vector_advection_dg(geometry, ibp, scheme, tracer_setup):
    """
    This tests the DG advection discretisation for vector
    fields in 2D slice and spherical geometry.
    """

    setup = tracer_setup(geometry)
    f, f_end = run(setup, ibp, scheme, vector=True)
    assert errornorm(f, f_end) < setup.err
