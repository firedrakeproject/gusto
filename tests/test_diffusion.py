from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate,
                       VectorFunctionSpace, Constant, exp, as_vector)
import pytest


def run(state, diffusion_scheme, tmax):

    timestepper = Timestepper(state, diffusion_scheme)
    timestepper.run(0., tmax)
    return timestepper.state.fields("f")


@pytest.mark.parametrize("vector", [True, False])
@pytest.mark.parametrize("DG", [True, False])
def test_diffusion(tmpdir, vector, DG, tracer_setup):

    setup = tracer_setup(tmpdir, geometry="slice", blob=True)
    state = setup.state
    f_init = setup.f_init
    tmax = setup.tmax
    
    if vector:
        kappa = Constant([[0.05, 0.], [0., 0.05]])
        if DG:
            V = VectorFunctionSpace(state.mesh, "DG", 1)
        else:
            V = state.spaces("HDiv")
        f_init = as_vector([f_init, 0.])
    else:
        kappa = 0.05
        if DG:
            V = state.spaces("DG")
        else:
            V = state.spaces("HDiv_v")

    f = state.fields("f", V)
    try:
        f.interpolate(f_init)
    except NotImplementedError:
        f.project(f_init)

    mu = 5.

    eqn = DiffusionEquation(state, V, "f", kappa=kappa, mu=mu)
    diffusion_scheme = [(eqn, BackwardEuler(state))]
    f_end = run(state, diffusion_scheme, tmax)

    assert f_end.dat.data.max() < 0.7
