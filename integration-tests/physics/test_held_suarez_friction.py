"""
This tests the Rayleigh friction term used in the Held Suarez test case.
"""

from gusto import *
import gusto.equations.thermodynamics as td
from gusto.core.labels import physics_label
from firedrake import (Constant, PeriodicIntervalMesh, as_vector, norm,
                       SpatialCoordinate, ExtrudedMesh, Function, dot)
from firedrake.fml import identity, drop


def run_apply_rayleigh_friction(dirname):
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
    output = OutputParameters(dirname=dirname+"Rayleigh_friction",
                              dumpfreq=1,
                              dumplist=['u'])
    io = IO(domain, output, diagnostic_fields=[XComponent('u')])

    time_discretisation = BackwardEuler(domain)
    physics_parameterisation = RayleighFriction(eqn, parameters)

    physics_scheme = [(physics_parameterisation, time_discretisation)]

    # Only want time derivatives and physics terms in equation, so drop the rest
    eqn.residual = eqn.residual.label_map(lambda t: any(t.has_label(time_derivative, physics_label)),
                                          map_if_true=identity, map_if_false=drop)

    # Time stepper
    scheme = ForwardEuler(domain)
    stepper = SplitPhysicsTimestepper(eqn, scheme, io,
                                      physics_schemes=physics_scheme)

    Vu = domain.spaces("HDiv")
    Vt = domain.spaces("theta")
    Vr = domain.spaces("DG")

    u0 = stepper.fields("u")
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")

    # Set prognostic variables -- there is initially some horizontal wind
    pressure = Function(Vr).interpolate(Constant(100000.))
    temperature = Function(Vt).interpolate(Constant(295.))
    theta_d = td.theta(parameters, temperature, pressure)

    theta0.project(theta_d)
    rho0.interpolate(pressure / (temperature*parameters.R_d))

    u0.project(as_vector([864.0, 0]))

    # Answer: Winds will be slowed by a factor of u/1day so
    day = 24*60*60
    u_true = Function(Vu)
    u_true.project(as_vector([(864 - 864/day), 0]))

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=dt)

    return mesh, stepper, u_true


def test_rayleigh_friction(tmpdir):

    dirname = str(tmpdir)
    mesh, stepper, u_true = run_apply_rayleigh_friction(dirname)

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
