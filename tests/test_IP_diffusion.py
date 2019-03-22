from gusto import *
from firedrake import (VectorFunctionSpace, Constant, as_vector, norm)
import pytest


def setup_IPdiffusion(setup, vector, DG):

    state = setup.state
    dt = state.dt
    tmax = setup.tmax
    f_init = setup.f_init

    kappa_ = 1.
    mu = 5.

    if vector:
        kappa = Constant([[kappa_, 0.], [0., kappa_]])
        if DG:
            Space = VectorFunctionSpace(state.mesh, "DG", 1)
        else:
            Space = state.spaces("HDiv")
        fexpr = as_vector([f_init, 0.])
        f_exact_expr = as_vector([(1/(1+4*tmax))*f_init**(1/(1+4*tmax)), 0.])

    else:
        kappa = kappa_
        if DG:
            Space = state.spaces("DG")
        else:
            Space = state.spaces("HDiv_v")
        fexpr = f_init

        f_exact_expr = (1/(1+4*tmax))*f_init**(1/(1+4*tmax))

    eqn = DiffusionEquation(state, Space, "f", kappa=kappa, mu=mu)
    schemes = [BackwardEuler(state, eqn)]
    f = state.fields("f", space=Space)
    f_exact = Function(Space)
    try:
        f.interpolate(fexpr)
        f_exact.interpolate(f_exact_expr)
    except NotImplementedError:
        f.project(fexpr)
        f_exact.project(f_exact_expr)

    stepper = Timestepper(state, schemes=schemes)
    return stepper, dt, tmax, f_exact


def run(setup, vector, DG):

    stepper, dt, tmax, f_exact = setup_IPdiffusion(setup, vector, DG)
    stepper.run(t=0., tmax=tmax)
    f = stepper.state.fields("f")
    ferr = Function(f.function_space()).assign(f-f_exact)
    return ferr


@pytest.mark.parametrize("vector", [True, False])
@pytest.mark.parametrize("DG", [True, False])
def test_ipdiffusion(tmpdir, vector, DG, tracer_setup):

    setup = tracer_setup(tmpdir, geometry="slice", blob=True)
    ferr = run(setup, vector, DG)
    err = norm(ferr)
    assert err < 0.03
