"""
Tests discretisations of the advection-diffusion equation. This checks the
errornorm for the resulting field to ensure that the result is reasonable.
"""

from firedrake import (SpatialCoordinate, PeriodicIntervalMesh, exp, as_vector,
                       norm, Constant, conditional, sqrt, VectorFunctionSpace)
from gusto import *


def run_advection_diffusion(tmpdir):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    dt = 0.02
    tmax = 1.0
    L = 10
    mesh = PeriodicIntervalMesh(20, L)
    domain = Domain(mesh, dt, "CG", 1)

    # Equation
    diffusion_params = DiffusionParameters(kappa=0.75, mu=5)
    V = domain.spaces("DG")
    Vu = VectorFunctionSpace(mesh, "CG", 1)

    equation = AdvectionDiffusionEquation(domain, V, "f", Vu=Vu,
                                          diffusion_parameters=diffusion_params)
    spatial_methods = [DGUpwind(equation, "f"),
                       InteriorPenaltyDiffusion(equation, "f", diffusion_params)]

    # I/O
    output = OutputParameters(dirname=str(tmpdir), dumpfreq=25)
    io = IO(domain, output)

    # Time stepper
    stepper = PrescribedTransport(equation, SSPRK3(domain), io, spatial_methods)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

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

    stepper.fields('f').interpolate(f_init_expr)
    stepper.fields('u').interpolate(as_vector([Constant(umax)]))
    f_end = stepper.fields('f_end', space=V)
    f_end.interpolate(f_end_expr)

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(0, tmax=tmax)

    error = norm(stepper.fields('f') - f_end) / norm(f_end)

    return error


def test_advection_diffusion(tmpdir):

    tol = 0.015
    error = run_advection_diffusion(tmpdir)
    assert error < tol, 'The error in the advection-diffusion ' + \
        'equation is greater than the permitted tolerance'
