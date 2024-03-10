"""
This tests the Held-Suarez physics routine to apply a temperature relaxation.
"""

from gusto import *
import gusto.thermodynamics as td
from gusto.labels import physics_label
from firedrake import (norm, Constant, PeriodicIntervalMesh, as_vector, dot,
                       SpatialCoordinate, ExtrudedMesh, Function, conditional)
from firedrake.fml import identity


def run_held_suarez_relaxation(dirname):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    dt = 24*60*60

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
    output = OutputParameters(dirname=dirname+"/held_suarez_relaxation",
                              dumpfreq=1,
                              dumplist=['theta'])
    io = IO(domain, output)

    # Physics scheme
    physics_parametrisation = Relaxation(eqn)

    time_discretisation = ForwardEuler(domain)

    # time_discretisation = ForwardEuler(domain)
    physics_schemes = [(physics_parametrisation, time_discretisation)]

    # Only want time derivatives and physics terms in equation, so drop the rest
    eqn.residual = eqn.residual.label_map(lambda t: any(t.has_label(time_derivative, physics_label)),
                                          map_if_true=identity, map_if_false=drop)

    # Time stepper
    scheme = BackwardEuler(domain)
    stepper = SplitPhysicsTimestepper(eqn, scheme, io,
                                      physics_schemes=physics_schemes)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    Vt = domain.spaces("theta")
    Vr = domain.spaces("DG")

    # Declare prognostic fields
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")

    # Set a background state with constant pressure and temperature
    pressure = Function(Vr).interpolate(Constant(100000.))
    temperature = Function(Vt).interpolate(Constant(305.))
    theta_d = td.theta(parameters, temperature, pressure)

    theta0.project(theta_d)
    rho0.interpolate(pressure / (temperature*parameters.R_d))

    # Answer: temperature closer to relaxation value
    theta_eq = Constant(304.)
    theta_true = Function(Vt)
    theta_true.interpolate(theta_eq)

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=dt)

    return stepper, theta_true


def test_held_suarez_relaxation(tmpdir):

    dirname = str(tmpdir)
    stepper, theta_true = run_held_suarez_relaxation(dirname)

    theta_final = stepper.fields('theta')

    denom = norm(theta_true)
    assert norm(theta_final - theta_true) / denom < 0.01, 'Final theta field is incorrect'
