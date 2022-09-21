"""
Tests discretisations of the advection-diffusion equation. This checks the
errornorm for the resulting field to ensure that the result is reasonable.
"""

from firedrake import (SpatialCoordinate, PeriodicIntervalMesh, exp, as_vector,
                       norm, Constant, conditional, sqrt, VectorFunctionSpace)
from gusto import *


def run_advection_diffusion(tmpdir):

    # Mesh, state and equation
    L = 10
    mesh = PeriodicIntervalMesh(20, L)
    dt = 0.02
    tmax = 1.0

    diffusion_params = DiffusionParameters(kappa=0.75, mu=5)
    output = OutputParameters(dirname=str(tmpdir), dumpfreq=25)
    state = State(mesh, dt=dt, output=output)
    V = state.spaces("DG", "DG", 1)
    Vu = VectorFunctionSpace(mesh, "CG", 1)

    equation = AdvectionDiffusionEquation(state, V, "f", Vu=Vu,
                                          diffusion_parameters=diffusion_params)

    problem = [(equation, ((SSPRK3(state), False, transport),
                           (BackwardEuler(state), False, diffusion)))]

    # Initial conditions
    x = SpatialCoordinate(mesh)
    xc_init = 0.25*L
    xc_end = 0.75*L
    umax = 0.5*L/tmax

    # Get minimum distance on periodic interval to xc
    x_init = conditional(sqrt((x[0] - xc_init)**2) < 0.5*L,
                         x[0] - xc_init, L + x[0] - xc_init)

    x_end = conditional(sqrt((x[0] - xc_end)**2) < 0.5*L,
                        x[0] - xc_end, L + x[0] - xc_end)

    f_init = 5.0
    f_end = f_init / 2.0
    f_width_init = L / 10.0
    f_width_end = f_width_init * 2.0
    f_init_expr = f_init*exp(-(x_init / f_width_init)**2)
    f_end_expr = f_end*exp(-(x_end / f_width_end)**2)

    state.fields('f').interpolate(f_init_expr)
    state.fields('u').interpolate(as_vector([Constant(umax)]))
    f_end = state.fields('f_end', V).interpolate(f_end_expr)

    # Time stepper
    timestepper = PrescribedTransport(state, problem)
    timestepper.run(0, tmax=tmax)

    error = norm(state.fields('f') - f_end) / norm(f_end)

    return error


def test_advection_diffusion(tmpdir):

    tol = 0.01
    error = run_advection_diffusion(tmpdir)
    assert error < tol, 'The error in the advection-diffusion ' + \
        'equation is greater than the permitted tolerance'
