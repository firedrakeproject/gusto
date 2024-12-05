"""
This script tests the non-split timestepper against the split timestepper
using an forced advection equation with a physics parametrisation.
One split method is tested, whilst different nonsplit IMEX and explicit time
discretisations are used for the dynamics and physics.
"""

from firedrake import (SpatialCoordinate, PeriodicIntervalMesh, exp, as_vector,
                       norm, Constant, conditional, sqrt, VectorFunctionSpace, acos,
                       cos, sin, FunctionSpace, Function, TestFunction, TrialFunction)
from gusto import *
import pytest
from math import pi


def run_nonsplit_adv_physics(tmpdir, timestepper):
    """
    Runs the advection equation with a physics parametrisation using different timesteppers.
    """

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    dt = 0.5
    tmax = 55
    L = 100
    mesh = PeriodicIntervalMesh(40, L)
    domain = Domain(mesh, dt, "CG", 1)

    # Parameters
    u_max = 1
    C0 = 0.6
    K0 = 0.3
    Csat = 0.75
    Ksat = 0.25
    x1 = 0
    x2 = L/4

    # Equation
    Vu = VectorFunctionSpace(mesh, "CG", 1)
    eltDG = FiniteElement("DG", "interval", 1, variant="equispaced")
    VD = FunctionSpace(mesh, eltDG)
    vapour = WaterVapour(name='water_vapour', space='DG')
    rain = Rain(name='rain', space='DG', transport_eqn=TransportEquationType.no_transport)
    tracers = [vapour, rain]
    equation = CoupledTransportEquation(domain, active_tracers=tracers, Vu=Vu)

    transport_method = [DGUpwind(equation, "water_vapour")]

    x = SpatialCoordinate(mesh)[0]

    # Physics scheme
    msat_expr = Csat + (Ksat * cos(2*pi*(x/L)))
    msat = Function(VD)
    msat.interpolate(msat_expr)

    # I/O
    output = OutputParameters(dirname=str(tmpdir), dumpfreq=25)
    io = IO(domain, output)

    time_varying_velocity = False

    # Time stepper
    time_varying_velocity = False
    if timestepper == 'split':
        physics_schemes = [(InstantRain(equation, msat, rain_name="rain"),
                            ForwardEuler(domain))]
        stepper = SplitPrescribedTransport(equation, SSPRK3(domain,
                                           limiter=DG1Limiter(VD, subspace=0)),
                                            io, time_varying_velocity, transport_method,
                                            physics_schemes=physics_schemes)
    elif timestepper == 'nonsplit_exp_rk_predictor':
        InstantRain(equation, msat, rain_name="rain")
        scheme = SSPRK3(domain, rk_formulation=RungeKuttaFormulation.predictor)
        stepper = PrescribedTransport(equation, scheme,
                                      io, time_varying_velocity,
                                      transport_method=transport_method)
    elif timestepper == 'nonsplit_exp_rk_increment':
        InstantRain(equation, msat, rain_name="rain")
        scheme = SSPRK3(domain, rk_formulation=RungeKuttaFormulation.increment)
        stepper = PrescribedTransport(equation, scheme,
                                      io, time_varying_velocity,
                                      transport_method=transport_method)
    elif timestepper == 'nonsplit_exp_multistep':
        InstantRain(equation, msat, rain_name="rain")
        scheme = AdamsBashforth(domain, order=2)
        stepper = PrescribedTransport(equation, scheme,
                                      io, time_varying_velocity,
                                      transport_method=transport_method)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    # initial moisture and wind profiles
    mexpr = C0 + K0*cos((2*pi*x)/L)
    stepper.fields("u").project(as_vector([u_max]))
    qv = stepper.fields("water_vapour")
    qv.project(mexpr)

    # exact rainfall profile (analytically)
    r_exact_func = Function(VD)
    r_exact = stepper.fields("r_exact", r_exact_func)
    lim1 = L/(2*pi) * acos((C0 + K0 - Csat)/Ksat)
    lim2 = L/2
    coord = (Ksat*cos(2*pi*x/L) + Csat - C0)/K0
    exact_expr = 2*Ksat*sin(2*pi*x/L)*acos(coord)
    r_expr = conditional(x < lim2, conditional(x > lim1, exact_expr, 0), 0)
    r_exact.interpolate(r_expr)

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(0, tmax=tmax)

    error = norm(stepper.fields('rain') - r_exact) / norm(r_exact)
    breakpoint()
    return error


@pytest.mark.parametrize("timestepper", ["split", "nonsplit_exp_rk_predictor",
                                         "nonsplit_exp_rk_increment", "nonsplit_exp_multistep"])
def test_nonsplit_adv_physics(tmpdir, timestepper):
    """
    Test the nonsplit timestepper in the advection equation with source physics.
    """
    tol = 0.2
    error = run_nonsplit_adv_physics(tmpdir, timestepper)
    assert error < tol, 'The nonsplit timestepper in the advection' + \
                        'equation with source physics has an error greater than ' + \
                        'the permitted tolerance'
