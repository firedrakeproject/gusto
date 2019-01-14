from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate,
                       VectorFunctionSpace, Constant, exp, as_vector)
import pytest


def setup_IPdiffusion(dirname, vector, DG):

    dt = 0.01
    L = 10.
    m = PeriodicIntervalMesh(50, L)
    mesh = ExtrudedMesh(m, layers=50, layer_height=0.2)

    output = OutputParameters(dirname=dirname)

    state = State(mesh, dt,
                  output=output)

    x = SpatialCoordinate(mesh)
    build_spaces(state, "CG", 1, 1)
    if vector:
        kappa = Constant([[0.05, 0.], [0., 0.05]])
        if DG:
            Space = VectorFunctionSpace(mesh, "DG", 1)
        else:
            Space = state.spaces("HDiv")
        fexpr = as_vector([exp(-(L/2.-x[0])**2 - (L/2.-x[1])**2), 0.])
    else:
        kappa = 0.05
        if DG:
            Space = state.spaces("DG")
        else:
            Space = state.spaces("HDiv_v")
        fexpr = exp(-(L/2.-x[0])**2 - (L/2.-x[1])**2)

    f = state.fields("f", Space)
    try:
        f.interpolate(fexpr)
    except NotImplementedError:
        f.project(fexpr)

    mu = 5.
    eqn = DiffusionEquation(state, f.name(), f.function_space(), kappa=kappa, mu=mu)
    schemes = [(f.name(), BackwardEuler(state, f, eqn))]
    stepper = Timestepper(state, schemes=schemes)
    return stepper


def run(dirname, vector, DG):

    stepper = setup_IPdiffusion(dirname, vector, DG)
    stepper.run(t=0., tmax=2.5)
    return stepper.state.fields("f")


@pytest.mark.parametrize("vector", [True, False])
@pytest.mark.parametrize("DG", [True, False])
def test_ipdiffusion(tmpdir, vector, DG):

    dirname = str(tmpdir)
    f = run(dirname, vector, DG)
    assert f.dat.data.max() < 0.7
