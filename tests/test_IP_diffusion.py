from gusto import *
from firedrake import (VectorFunctionSpace, Constant, as_vector, norm)
import pytest


def setup_IPdiffusion(setup, vector, DG):

    state = setup.state
    dt = setup.dt
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

        def f_exact(t):
            return as_vector([(1/(1+4*t))*f_init**(1/(1+4*t)), 0.])

    else:
        kappa = kappa_
        if DG:
            Space = state.spaces("DG")
        else:
            Space = state.spaces("HDiv_v")
        fexpr = f_init

        def f_exact(t):
            return (1/(1+4*t))*f_init**(1/(1+4*t))

    equations_schemes = [
        (DiffusionEquation(state, Space, "f", kappa=kappa, mu=mu),
         BackwardEuler())]
    f = state.fields("f", space=Space)
    try:
        f.interpolate(fexpr)
    except NotImplementedError:
        f.project(fexpr)

    state.fields("f_exact", space=Space)
    prescribed_fields = [("f_exact", f_exact)]

    stepper = Timestepper(state, equations_schemes=equations_schemes,
                          prescribed_fields=prescribed_fields)
    return stepper, dt, tmax


def run(setup, vector, DG):

    stepper, dt, tmax = setup_IPdiffusion(setup, vector, DG)
    stepper.run(t=0., dt=dt, tmax=tmax)
    return stepper.state.fields("f_minus_f_exact")


@pytest.mark.parametrize("vector", [True, False])
@pytest.mark.parametrize("DG", [True, False])
def test_ipdiffusion(tmpdir, vector, DG, tracer_setup):

    setup = tracer_setup(tmpdir, geometry="slice", blob=True)
    ferr = run(setup, vector, DG)
    err = norm(ferr)
    assert err < 0.03
