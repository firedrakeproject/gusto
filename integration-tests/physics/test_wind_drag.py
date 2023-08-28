"""
This tests the physics routine to apply drag to the wind.
"""

from gusto import *
import gusto.thermodynamics as td
from gusto.labels import physics
from firedrake import (norm, Constant, PeriodicIntervalMesh, as_vector,
                       SpatialCoordinate, ExtrudedMesh, Function, conditional)
import pytest


def run_wind_drag(dirname, implicit_formulation):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    dt = 100.0

    # declare grid shape, with length L and height H
    L = 500.
    H = 500.
    nlayers = int(H / 5.)
    ncolumns = int(L / 5.)

    # make mesh and domain
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=(H / nlayers))
    domain = Domain(mesh, dt, "CG", 0)

    _, z = SpatialCoordinate(mesh)

    # Set up equation
    parameters = CompressibleParameters()
    eqn = CompressibleEulerEquations(domain, parameters)

    # I/O
    output = OutputParameters(dirname=dirname+"/surface_fluxes",
                              dumpfreq=1,
                              dumplist=['u'])
    io = IO(domain, output)

    # Physics scheme
    surf_params = BoundaryLayerParameters()
    physics_parametrisation = WindDrag(eqn, implicit_formulation, surf_params)

    time_discretisation = ForwardEuler(domain) if implicit_formulation else BackwardEuler(domain)

    # time_discretisation = ForwardEuler(domain)
    physics_schemes = [(physics_parametrisation, time_discretisation)]

    # Only want time derivatives and physics terms in equation, so drop the rest
    eqn.residual = eqn.residual.label_map(lambda t: any(t.has_label(time_derivative, physics)),
                                          map_if_true=identity, map_if_false=drop)

    # Time stepper
    scheme = ForwardEuler(domain)
    stepper = SplitPhysicsTimestepper(eqn, scheme, io,
                                      physics_schemes=physics_schemes)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    Vu = domain.spaces("HDiv")
    Vt = domain.spaces("theta")
    Vr = domain.spaces("DG")

    surface_mask = Function(Vt)
    surface_mask.interpolate(conditional(z < 100., 1.0, 0.0))

    # Declare prognostic fields
    u0 = stepper.fields("u")
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")

    # Set a background state with constant pressure and temperature
    pressure = Function(Vr).interpolate(Constant(100000.))
    temperature = Function(Vt).interpolate(Constant(295.))
    theta_d = td.theta(parameters, temperature, pressure)

    theta0.project(theta_d)
    rho0.interpolate(pressure / (temperature*parameters.R_d))

    # Constant horizontal wind
    u0.project(as_vector([15.0, 0.0]))

    # Answer: slower winds than initially
    u_true = Function(Vu)
    u_true.project(surface_mask*as_vector([14.53, 0.0]) + (1-surface_mask)*u0)

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=dt)

    return mesh, stepper, u_true


@pytest.mark.parametrize("implicit_formulation", [False, True])
def test_wind_drag(tmpdir, implicit_formulation):

    dirname = str(tmpdir)
    mesh, stepper, u_true = run_wind_drag(dirname, implicit_formulation)

    u_final = stepper.fields('u')

    # Project into CG1 to get sensible values
    e_x = as_vector([1.0, 0.0])
    e_z = as_vector([0.0, 1.0])

    DG0 = FunctionSpace(mesh, "DG", 0)
    u_x_final = Function(DG0).project(dot(u_final, e_x))
    u_x_true = Function(DG0).project(dot(u_true, e_x))
    u_z_final = Function(DG0).project(dot(u_final, e_z))
    u_z_true = Function(DG0).project(dot(u_true, e_z))

    denom = norm(u_x_true)
    assert norm(u_x_final - u_x_true) / denom < 0.01, 'Final horizontal wind is incorrect'
    assert norm(u_z_final - u_z_true) < 1e-12, 'Final vertical wind is incorrect'
