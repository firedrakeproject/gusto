"""
This tests the physics routine to provide surface fluxes. The initial fields are
set to correspond to a different temperature at the surface -- we then check
that afterwards the surface temperature is correct.
"""

from gusto import *
import gusto.thermodynamics as td
from gusto.labels import physics_label
from firedrake import (norm, Constant, PeriodicIntervalMesh, as_vector,
                       SpatialCoordinate, ExtrudedMesh, Function, conditional)
import pytest


def run_surface_fluxes(dirname, moist, implicit_formulation):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    dt = 100.0

    # declare grid shape, with length L and height H
    L = 500.
    H = 500.
    nlayers = 5
    ncolumns = 5

    # make mesh and domain
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=(H / nlayers))
    domain = Domain(mesh, dt, "CG", 0)

    _, z = SpatialCoordinate(mesh)

    # Set up equation
    tracers = [WaterVapour()] if moist else None
    vapour_name = 'water_vapour' if moist else None
    parameters = CompressibleParameters()
    eqn = CompressibleEulerEquations(domain, parameters, active_tracers=tracers)

    # I/O
    output = OutputParameters(dirname=dirname+"/surface_fluxes",
                              dumpfreq=1,
                              dumplist=['u'])
    io = IO(domain, output)

    # Physics scheme
    surf_params = BoundaryLayerParameters()
    T_surf = Constant(300.0)
    physics_parametrisation = SurfaceFluxes(eqn, T_surf, vapour_name,
                                            implicit_formulation, surf_params)

    time_discretisation = ForwardEuler(domain) if implicit_formulation else BackwardEuler(domain)

    # time_discretisation = ForwardEuler(domain)
    physics_schemes = [(physics_parametrisation, time_discretisation)]

    # Only want time derivatives and physics terms in equation, so drop the rest
    eqn.residual = eqn.residual.label_map(lambda t: any(t.has_label(time_derivative, physics_label)),
                                          map_if_true=identity, map_if_false=drop)

    # Time stepper
    scheme = ForwardEuler(domain)
    stepper = SplitPhysicsTimestepper(eqn, scheme, io,
                                      physics_schemes=physics_schemes)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    Vt = domain.spaces("theta")
    Vr = domain.spaces("DG")

    surface_mask = Function(Vt)
    surface_mask.interpolate(conditional(z < 100., 0.1, 0.0))

    # Declare prognostic fields
    u0 = stepper.fields("u")
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")

    # Set a background state with constant pressure and temperature
    pressure = Function(Vr).interpolate(Constant(100000.))
    temperature = Function(Vt).interpolate(Constant(295.))
    theta_d = td.theta(parameters, temperature, pressure)
    mv_sat = td.r_sat(parameters, temperature, pressure)

    # Set prognostic variables
    if moist:
        water_v0 = stepper.fields("water_vapour")
        water_v0.interpolate(0.95*mv_sat)
        theta0.project(theta_d*(1 + water_v0 * parameters.R_v / parameters.R_d))
        rho0.interpolate(pressure / (temperature*parameters.R_d * (1 + water_v0 * parameters.R_v / parameters.R_d)))
    else:
        theta0.project(theta_d)
        rho0.interpolate(pressure / (temperature*parameters.R_d))

    T_true = Function(Vt)
    T_true.interpolate(surface_mask*T_surf + (1-surface_mask)*temperature)

    if moist:
        mv_true = Function(Vt)
        mv_true.interpolate(surface_mask*mv_sat + (1-surface_mask)*water_v0)
    else:
        mv_true = None

    # Constant horizontal wind
    u0.project(as_vector([5.0, 0.0]))

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=dt)

    return eqn, stepper, T_true, mv_true


@pytest.mark.parametrize("moist", [False, True])
@pytest.mark.parametrize("implicit_formulation", [False, True])
def test_surface_fluxes(tmpdir, moist, implicit_formulation):

    dirname = str(tmpdir)
    eqn, stepper, T_true, mv_true = run_surface_fluxes(dirname, moist, implicit_formulation)

    # Back out temperature from prognostic fields
    theta_vd = stepper.fields('theta')
    rho = stepper.fields('rho')
    exner = td.exner_pressure(eqn.parameters, rho, theta_vd)
    mv = stepper.fields('water_vapour') if moist else None
    T_expr = td.T(eqn.parameters, theta_vd, exner, r_v=mv)

    # Project T_expr
    T = Function(theta_vd.function_space())
    T.project(T_expr)
    denom = norm(T)
    assert norm(T - T_true) / denom < 0.001, 'Final temperature is incorrect'

    if moist:
        denom = norm(mv_true)
        assert norm(mv - mv_true) / denom < 0.01, 'Final water vapour is incorrect'
