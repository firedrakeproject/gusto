"""
This script tests the non-split timestepper against the split timestepper
using an advection equation with a physics parametrisation.
One split method is tested, whilst different nonsplit IMEX and explicit time
discretisations are used for the dynamics and physics.
"""

from firedrake import (SpatialCoordinate, PeriodicIntervalMesh, exp, as_vector,
                       norm, Constant, conditional, sqrt, VectorFunctionSpace)
from gusto import *
import pytest


def run_nonsplit_adv_physics(tmpdir, timestepper):
    """
    Runs the advection equation with a physics parametrisation using different timesteppers.
    """

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    dt = 0.01
    tmax = 0.75
    L = 10
    mesh = PeriodicIntervalMesh(20, L)
    domain = Domain(mesh, dt, "CG", 1)

    # Equation
    V = domain.spaces("DG")
    Vu = VectorFunctionSpace(mesh, "CG", 1)
    equation = ContinuityEquation(domain, V, "f", Vu=Vu)
    spatial_methods = [DGUpwind(equation, "f")]

    x = SpatialCoordinate(mesh)

    # Add a source term to inject mass into the domain.
    source_expression = -Constant(0.5)
    physics_schemes = [(SourceSink(equation, "f", source_expression), SSPRK3(domain))]

    # I/O
    output = OutputParameters(dirname=str(tmpdir), dumpfreq=25)
    io = IO(domain, output)

    time_varying_velocity = False

    # Time stepper
    if timestepper == 'split':
        # Split with no defined weights
        dynamics_schemes = {'transport': ImplicitMidpoint(domain)}
        term_splitting = ['transport', 'physics']
        stepper = SplitTimestepper(equation, term_splitting, dynamics_schemes,
                                   io, spatial_methods=spatial_methods,
                                   physics_schemes=physics_schemes)
    elif timestepper == 'nonsplit_imex_rk':
        # Split continuity term
        equation = split_continuity_form(equation)
        # Label terms as implicit and explicit
        equation.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
        equation.label_terms(lambda t: t.has_label(transport), explicit)
        dynamics_schemes = IMEX_SSP3(domain)
        stepper = PrescribedTransport(equation, dynamics_schemes,
                                      io, time_varying_velocity,
                                      transport_method=spatial_methods)
    elif timestepper == 'nonsplit_exp_rk_predictor':
        dynamics_schemes = SSPRK3(domain, rk_formulation=RungeKuttaFormulation.predictor)
        stepper = PrescribedTransport(equation, dynamics_schemes,
                                      io, time_varying_velocity,
                                      transport_method=spatial_methods)
    elif timestepper == 'nonsplit_exp_rk_increment':
        dynamics_schemes = SSPRK3(domain, rk_formulation=RungeKuttaFormulation.increment)
        stepper = PrescribedTransport(equation, dynamics_schemes,
                                      io, time_varying_velocity,
                                      transport_method=spatial_methods)
    elif timestepper == 'nonsplit_imex_sdc':
        # Split continuity term
        equation = split_continuity_form(equation)
        # Label terms as implicit and explicit
        equation.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
        equation.label_terms(lambda t: t.has_label(transport), explicit)

        node_type = "LEGENDRE"
        qdelta_imp = "LU"
        qdelta_exp = "FE"
        quad_type = "RADAU-RIGHT"
        M = 2
        k = 2
        base_scheme = IMEX_Euler(domain)
        dynamics_schemes = SDC(base_scheme, domain, M, k, quad_type, node_type, qdelta_imp,
                               qdelta_exp, formulation="Z2N", final_update=True, initial_guess="base")
        stepper = PrescribedTransport(equation, dynamics_schemes,
                                      io, time_varying_velocity,
                                      transport_method=spatial_methods)
    elif timestepper == 'nonsplit_exp_multistep':
        dynamics_schemes = AdamsBashforth(domain, order=2)
        stepper = PrescribedTransport(equation, dynamics_schemes,
                                      io, time_varying_velocity,
                                      transport_method=spatial_methods)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    xc_init = 0.25 * L
    xc_end = 0.75 * L
    umax = 0.5 * L / tmax

    # Get minimum distance on periodic interval to xc
    x_init = conditional(sqrt((x[0] - xc_init) ** 2) < 0.5 * L,
                         x[0] - xc_init, L + x[0] - xc_init)

    x_end = conditional(sqrt((x[0] - xc_end) ** 2) < 0.5 * L,
                        x[0] - xc_end, L + x[0] - xc_end)

    f_init = 5.0
    f_end = f_init
    f_width_init = L / 10.0
    f_width_end = f_width_init
    f_init_expr = f_init * exp(-(x_init / f_width_init) ** 2)

    # The end Gaussian should be advected by half the domain
    # length and include more mass due to the source term.
    f_end_expr = 0.5 + f_end * exp(-(x_end / f_width_end) ** 2)

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


@pytest.mark.parametrize("timestepper", ["split", "nonsplit_imex_rk", "nonsplit_imex_sdc",
                                         "nonsplit_exp_rk_predictor", "nonsplit_exp_rk_increment", "nonsplit_exp_multistep"])
def test_nonsplit_adv_physics(tmpdir, timestepper):
    """
    Test the nonsplit timestepper in the advection equation with source physics.
    """
    tol = 0.12
    error = run_nonsplit_adv_physics(tmpdir, timestepper)
    assert error < tol, 'The nonsplit timestepper in the advection' + \
                        'equation with source physics has an error greater than ' + \
                        'the permitted tolerance'
