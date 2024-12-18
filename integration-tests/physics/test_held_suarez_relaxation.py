"""
This tests the Held-Suarez physics routine to apply Rayleigh friction.
"""
from gusto import *
import gusto.equations.thermodynamics as td
from gusto.core.labels import physics_label
from firedrake import (Constant, PeriodicIntervalMesh, as_vector, drop,
                       ExtrudedMesh, Function)
from firedrake.fml import identity
import pytest


def run_held_suarez_relaxation(dirname, temp):

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
    physics_parametrisation = Relaxation(eqn, 'theta', parameters)

    time_discretisation = ForwardEuler(domain)

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

    # Declare prognostic fields
    u0 = stepper.fields("u")
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")

    # Set a background state with constant pressure and temperature
    p0 = 100000.
    pressure = Function(Vr).interpolate(Constant(p0))
    temperature = Function(Vt).interpolate(Constant(temp))
    theta_d = td.theta(parameters, temperature, pressure)

    theta0.project(theta_d)
    rho0.interpolate(p0 / (theta_d*parameters.R_d))  # This ensures that exner = 1

    # Constant horizontal wind
    u0.project(as_vector([1.0, 1.0]))
    theta_initial = Function(Vt)
    theta_initial.interpolate(theta_d)

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=dt)

    return stepper, theta_initial


@pytest.mark.parametrize('temp', [280, 290])
def test_held_suarez_relaxation(tmpdir, temp):
    # By configuring the fields we have set the equilibrium temperature to 285K
    # We test a temperature value eith side to check it moves in the right direction
    dirname = str(tmpdir)
    stepper, theta_initial = run_held_suarez_relaxation(dirname, temp)

    theta_final = stepper.fields('theta')
    final_data = theta_final.dat.data
    initial_data = theta_initial.dat.data
    if temp == 280:
        assert np.mean(final_data) > np.mean(initial_data)
    if temp == 290:
        assert np.mean(final_data) < np.mean(initial_data)
