"""
This tests the Held-Suarez physics routine to apply Rayleigh friction.
"""

from gusto import *
import gusto.thermodynamics as td
from gusto.labels import physics_label
from firedrake import (norm, Constant, PeriodicIntervalMesh, as_vector, dot,
                       SpatialCoordinate, ExtrudedMesh, Function, conditional)
from firedrake.fml import identity


def run_held_suarez_friction(dirname):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    dt = 3600.0

    # declare grid shape, with length L and height H
    L = 500.
    H = 500.
    nlayers = int(H / 5.)
    ncolumns = int(L / 5.)

    # make mesh and domain
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=(H / nlayers))
    domain = Domain(mesh, dt, "CG", 0)

    # Set up equation
    parameters = CompressibleParameters()
    eqn = CompressibleEulerEquations(domain, parameters)

    # I/O
    output = OutputParameters(dirname=dirname+"/held_suarez_friction",
                              dumpfreq=1,
                              dumplist=['u'])
    io = IO(domain, output)

    # Physics scheme
    physics_parametrisation = RayleighFriction(eqn)

    time_discretisation = BackwardEuler(domain)

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

    Vu = domain.spaces("HDiv")
    Vt = domain.spaces("theta")
    Vr = domain.spaces("DG")

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
    u_true.project(as_vector([14.4, 0.0]))

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=dt)

    return mesh, stepper, u_true


def test_held_suarez_friction(tmpdir):

    dirname = str(tmpdir)
    mesh, stepper, u_true = run_held_suarez_friction(dirname)

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
