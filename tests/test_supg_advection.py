from firedrake import VectorFunctionSpace, Function, as_vector, errornorm
from gusto import *
import pytest


def run(state, equations, schemes, dt, tmax):

    timestepper = PrescribedAdvectionTimestepper(
        state, equations=equations, schemes=schemes)
    timestepper.run(0, dt=dt, tmax=tmax)
    return timestepper.state.fields("f")


@pytest.mark.parametrize("scheme", ["ssprk", "im"])
@pytest.mark.parametrize("space", ["cg", "theta", "vector", "HDiv"])
def test_advection_supg(scheme, space, tracer_setup):
    """
    This tests the embedded DG advection scheme for scalar and vector fields
    in slice geometry.
    """
    setup = tracer_setup("slice")
    state = setup.state
    dt = setup.dt
    tmax = setup.tmax
    f_init = setup.f_init
    f_end_expr = setup.f_end
    err = setup.err

    if space == "cg":
        fspace = FunctionSpace(state.mesh, "CG", 1)
        ibp = IntegrateByParts.NEVER
    elif space == "theta":
        fspace = state.spaces("HDiv_v")
        ibp = IntegrateByParts.TWICE
    elif space == "vector":
        fspace = VectorFunctionSpace(state.mesh, "CG", 1)
        ibp = IntegrateByParts.NEVER
    elif space == "HDiv":
        fspace = state.spaces("HDiv")
        ibp = IntegrateByParts.TWICE

    if space in ["vector", "HDiv"]:
        # expression for vector initial and final conditions
        vec_expr = [0.]*state.mesh.geometric_dimension()
        vec_expr[0] = f_init
        f_init = as_vector(vec_expr)
        vec_end_expr = [0.]*state.mesh.geometric_dimension()
        vec_end_expr[0] = f_end_expr
        f_end_expr = as_vector(vec_end_expr)

    try:
        f_end = Function(fspace).interpolate(f_end_expr)
    except NotImplementedError:
        f_end = Function(fspace).project(f_end_expr)

    supg_opts = SUPGOptions()
    equations = [("f", AdvectionEquation(state, fspace, "f", ibp=ibp))]
    f = state.fields("f")
    try:
        f.interpolate(f_init)
    except NotImplementedError:
        f.project(f_init)

    if scheme == "ssprk":
        schemes = [("f", SSPRK3(options=supg_opts))]
    elif scheme == "im":
        schemes = [("f", ThetaMethod(options=supg_opts))]

    f = run(state, equations, schemes, dt, tmax)
    assert errornorm(f, f_end) < err
