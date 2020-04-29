from gusto import *
from firedrake import (VectorFunctionSpace, Constant, as_vector, norm)
import pytest


def run(state, diffusion_scheme, tmax, f_end):

    timestepper = Timestepper(state, diffusion_scheme)
    timestepper.run(0., tmax)
    return norm(timestepper.state.fields("f") - f_end)


@pytest.mark.parametrize("vector", [True, False])
@pytest.mark.parametrize("DG", [True, False])
def test_diffusion(tmpdir, vector, DG, tracer_setup):

    setup = tracer_setup(tmpdir, geometry="slice", blob=True)
    state = setup.state
    f_init = setup.f_init
    tmax = setup.tmax
    tol = 3.e-2
    kappa = 1.

    f_end_expr = (1/(1+4*tmax))*f_init**(1/(1+4*tmax))

    if vector:
        kappa = Constant([[kappa, 0.], [0., kappa]])
        if DG:
            V = VectorFunctionSpace(state.mesh, "DG", 1)
        else:
            V = state.spaces("HDiv", "CG", 1)
        f_init = as_vector([f_init, 0.])
        f_end_expr = as_vector([f_end_expr, 0.])
    else:
        if DG:
            V = state.spaces("DG", "DG", 1)
        else:
            V = state.spaces("theta", degree=1)

    mu = 5.

    diffusion_params = DiffusionParameters(kappa=kappa, mu=mu)
    eqn = DiffusionEquation(state, V, "f",
                            diffusion_parameters=diffusion_params)
    f = state.fields("f")
    f_end = state.fields("f_end", V)
    try:
        f.interpolate(f_init)
        f_end.interpolate(f_end_expr)
    except NotImplementedError:
        f.project(f_init)
        f_end.project(f_end_expr)

    diffusion_scheme = [(eqn, BackwardEuler(state))]

    assert run(state, diffusion_scheme, tmax, f_end) < tol
