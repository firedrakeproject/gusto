from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate,
                       BrokenElement, VectorFunctionSpace, as_vector, exp, FunctionSpace)
import pytest


def setup_IPdiffusion(dirname, vector, DG):

    dt = 0.1
    L = 10.
    m = PeriodicIntervalMesh(20, L)
    mesh = ExtrudedMesh(m, layers=20, layer_height=0.5)

    fieldlist = ['u', 'D']
    timestepping = TimesteppingParameters(dt=dt)
    parameters = CompressibleParameters()
    output = OutputParameters(dirname=dirname)

    state = State(mesh, vertical_degree=0, horizontal_degree=0,
                  family="CG",
                  timestepping=timestepping,
                  parameters=parameters,
                  output=output,
                  fieldlist=fieldlist)

    DG1 = FunctionSpace(mesh, "DG", 1)
    VectorDG1 = VectorFunctionSpace(mesh, "DG", 1)
    CG1 = FunctionSpace(mesh, "CG", 1)
    VectorCG1 = VectorFunctionSpace(mesh, "CG", 1)

    x = SpatialCoordinate(mesh)
    kappa = 0.05
    if vector:
        if DG:
            Space = VectorFunctionSpace(mesh, "DG", 0)
            f_opts = RecoveredOptions(embedding_space=VectorDG1,
                                      recovered_space=VectorCG1,
                                      broken_space=Space,
                                      boundary_method=Boundary_Method.dynamics)

        else:
            Space = state.spaces("HDiv")
            f_opts = RecoveredOptions(embedding_space=VectorDG1,
                                      recovered_space=VectorCG1,
                                      broken_space=FunctionSpace(mesh, BrokenElement(Space.ufl_element())),
                                      boundary_method=Boundary_Method.dynamics)
        fexpr = as_vector([exp(-(L/2.-x[0])**2 - (L/2.-x[1])**2), 0.])
    else:
        if DG:
            Space = state.spaces("DG")
            f_opts = RecoveredOptions(embedding_space=DG1,
                                      recovered_space=CG1,
                                      broken_space=Space,
                                      boundary_method=Boundary_Method.dynamics)

        else:
            Space = state.spaces("HDiv_v")
            f_opts = RecoveredOptions(embedding_space=DG1,
                                      recovered_space=CG1,
                                      broken_space=FunctionSpace(mesh, BrokenElement(Space.ufl_element())),
                                      boundary_method=Boundary_Method.dynamics)

        fexpr = exp(-(L/2.-x[0])**2 - (L/2.-x[1])**2)

    f = state.fields("f", Space)
    try:
        f.interpolate(fexpr)
    except NotImplementedError:
        f.project(fexpr)

    mu = 5.

    if vector:
        f_diffusion = RecoveredDiffusion(state, [InteriorPenalty(state, DG1, kappa=kappa, mu=mu),
                                                 InteriorPenalty(state, DG1, kappa=kappa, mu=mu)],
                                         f.function_space(), f_opts)
    else:
        f_diffusion = RecoveredDiffusion(state, InteriorPenalty(state, DG1, kappa=kappa, mu=mu),
                                         f.function_space(), f_opts)
    diffused_fields = [("f", f_diffusion)]
    stepper = AdvectionDiffusion(state, diffused_fields=diffused_fields)
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
