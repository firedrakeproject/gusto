import pytest
import numpy as np

from firedrake import *
from firedrake.adjoint import *
from pyadjoint import get_working_tape
from gusto import *


@pytest.fixture(autouse=True)
def autouse_set_test_tape(set_test_tape):
    pass


@pytest.mark.parametrize("nu_is_control", [True, False])
def test_diffusion_sensitivity(nu_is_control, tmpdir):
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
    nu = Function(R, val=0.2)
    diffusion_params = DiffusionParameters(mesh, kappa=nu)
    eqn = DiffusionEquation(domain, V, "f", diffusion_parameters=diffusion_params)

    # Don't let an inexact solve get in the way of a good Taylor test
    solver_parameters = {
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
    }
    diffusion_scheme = BackwardEuler(domain, solver_parameters=solver_parameters)
    diffusion_methods = [CGDiffusion(eqn, "f", diffusion_params)]
    timestepper = Timestepper(eqn, diffusion_scheme, io, spatial_methods=diffusion_methods)

    x = SpatialCoordinate(mesh)
    fexpr = as_vector((sin(2*pi*x[0]), cos(2*pi*x[1])))
    ic = Function(V).interpolate(fexpr)

    # These are the only operations we are interested in rerunning with the tape.
    with set_working_tape() as tape:
        # initialise the solution
        timestepper.fields("f").assign(ic)

        # propagate forward
        end = 0.1
        timestepper.run(0., end)

        # Make sure we have more than a quadratic nonlinearity so Jhat.hessian isn't exact.
        u = timestepper.fields("f")
        J = assemble(inner(u, u)*dx)**2

        if nu_is_control:
            m = nu
        else:
            m = ic

        # the functional as a pure function of the control
        Jhat = ReducedFunctional(J, Control(m), tape=tape)

    assert np.allclose(J, Jhat(m)), "Functional re-evaluation does not match original evaluation."

    # perturbation direction for taylor test
    if nu_is_control:
        h = Function(R, val=0.1)
    else:
        h = Function(V).interpolate(as_vector([cos(4*pi*x[1]), sin(4*pi*x[0])]))

    # Check the TLM explicitly before checking the Hessian (which relies on the tlm)
    assert taylor_test(Jhat, m, h, dJdm=Jhat.tlm(h)) > 1.95, "TLM is not second order accurate."

    # Check the re-evaluation, derivative, and Hessian all converge at the expected rates.
    taylor = taylor_to_dict(Jhat, m, h)
    assert min(taylor['R0']['Rate']) > 0.95, taylor['R0']
    assert min(taylor['R1']['Rate']) > 1.95, taylor['R1']
    assert min(taylor['R2']['Rate']) > 2.95, taylor['R2']
