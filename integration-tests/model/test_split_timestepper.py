"""
This script tests the split_timestepper, using an
advection-diffusion equation with a physics
parametrisation. Three different splittings are
tested, including splitting the dynamics and
physics into two substeps with different timesteps.
"""

from firedrake import (SpatialCoordinate, PeriodicIntervalMesh, exp, as_vector,
                       norm, Constant, conditional, sqrt, VectorFunctionSpace)
from gusto import *
import pytest


def run_split_timestepper_adv_diff_physics(tmpdir, timestepper):

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

    x = SpatialCoordinate(mesh)

    # Add a source term to inject mass into the domain.
    # Without the diffusion, this would simply add 0.1
    # units of mass equally across the domain.
    source_expression = -Constant(0.1)

    physics_schemes = [(SourceSink(equation, "f", source_expression), SSPRK3(domain))]

    # I/O
    output = OutputParameters(dirname=str(tmpdir), dumpfreq=25)
    io = IO(domain, output)

    # Time stepper
    if timestepper == 'split1':
        # Split with no defined weights
        dynamics_schemes = {'transport': ImplicitMidpoint(domain),
                            'diffusion': ForwardEuler(domain)}
        term_splitting = ['transport', 'diffusion', 'physics']
        stepper = SplitTimestepper(equation, term_splitting, dynamics_schemes, io, spatial_methods=spatial_methods, physics_schemes=physics_schemes)
    elif timestepper == 'split2':
        # Transport split into two substeps
        dynamics_schemes = {'transport': SSPRK3(domain),
                            'diffusion': ForwardEuler(domain)}
        term_splitting = ['diffusion', 'transport', 'physics', 'transport']
        weights = [1., 0.6, 1., 0.4]
        stepper = SplitTimestepper(equation, term_splitting, dynamics_schemes, io, weights=weights, spatial_methods=spatial_methods, physics_schemes=physics_schemes)
    else:
        # Physics split into two substeps
        dynamics_schemes = {'transport': SSPRK3(domain),
                            'diffusion': SSPRK3(domain)}
        term_splitting = ['physics', 'transport', 'diffusion', 'physics']
        weights = [1./3., 1, 1, 2./3.]
        stepper = SplitTimestepper(equation, term_splitting, dynamics_schemes, io, weights=weights, spatial_methods=spatial_methods, physics_schemes=physics_schemes)
    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

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

    # The end Gaussian should be advected by half the domain
    # length, be more spread out due to the dissipation,
    # and includes more mass due to the source term.
    f_end_expr = 0.1 + f_end*exp(-(x_end / f_width_end)**2)

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


@pytest.mark.parametrize("timestepper", ["split1", "split2", "split3"])
def test_split_timestepper_adv_diff_physics(tmpdir, timestepper):

    tol = 0.015
    error = run_split_timestepper_adv_diff_physics(tmpdir, timestepper)
    print(error)
    assert error < tol, 'The split timestepper in the advection-diffusion' + \
        'equation with source physics has an error greater than ' + \
        'the permitted tolerance'
