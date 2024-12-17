import pytest
import numpy as np

from firedrake import *
from firedrake.adjoint import *
from pyadjoint import get_working_tape
from gusto import *


@pytest.fixture(autouse=True)
def handle_taping():
    yield
    tape = get_working_tape()
    tape.clear_tape()


@pytest.fixture(autouse=True, scope="module")
def handle_annotation():
    from firedrake.adjoint import annotate_tape, continue_annotation
    if not annotate_tape():
        continue_annotation()
    yield
    # Ensure annotation is paused when we finish.
    annotate = annotate_tape()
    if annotate:
        pause_annotation()


@pytest.mark.parametrize("nu_is_control", [True, False])
def test_diffusion(nu_is_control, tmpdir):
    assert get_working_tape()._blocks == []
    n = 30
    mesh = PeriodicUnitSquareMesh(n, n)
    output = OutputParameters(dirname=str(tmpdir))
    dt = 0.01
    domain = Domain(mesh, 10*dt, family="BDM", degree=1)
    io = IO(domain, output)

    V = VectorFunctionSpace(mesh, "CG", 2)
    domain.spaces.add_space("vecCG", V)

    R = FunctionSpace(mesh, "R", 0)
    # We need to define nu as a function in order to have a control variable.
    nu = Function(R, val=0.0001)
    diffusion_params = DiffusionParameters(kappa=nu)
    eqn = DiffusionEquation(domain, V, "f", diffusion_parameters=diffusion_params)

    diffusion_scheme = BackwardEuler(domain)
    diffusion_methods = [CGDiffusion(eqn, "f", diffusion_params)]
    timestepper = Timestepper(eqn, diffusion_scheme, io, spatial_methods=diffusion_methods)

    x = SpatialCoordinate(mesh)
    fexpr = as_vector((sin(2*pi*x[0]), cos(2*pi*x[1])))
    timestepper.fields("f").interpolate(fexpr)

    end = 0.1
    timestepper.run(0., end)

    u = timestepper.fields("f")
    J = assemble(inner(u, u)*dx)

    if nu_is_control:
        control = Control(nu)
        h = Function(R, val=0.0001)  # the direction of the perturbation
    else:
        control = Control(u)
        h = Function(V).interpolate(fexpr)  # the direction of the perturbation

    Jhat = ReducedFunctional(J, control)  # the functional as a pure function of nu

    if nu_is_control:
        assert np.allclose(J, Jhat(nu))
        assert taylor_test(Jhat, nu, h) > 1.95
    else:
        assert np.allclose(J, Jhat(u))
        assert taylor_test(Jhat, u, h) > 1.95
