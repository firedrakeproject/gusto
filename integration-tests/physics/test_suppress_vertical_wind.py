"""
This tests the physics routine to apply suppress vertical wind in a model's spin
up period.
"""

from gusto import *
from gusto.core.labels import physics_label
from firedrake import (Constant, PeriodicIntervalMesh, as_vector, sin, norm,
                       SpatialCoordinate, ExtrudedMesh, Function, dot, pi)
from firedrake.fml import identity, drop
import pytest


def run_suppress_vertical_wind(dirname, physics_coupling):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    dt = 100.0
    spin_up_period = 5*dt

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
    tracers = [WaterVapour()]
    parameters = CompressibleParameters(mesh)
    eqn = CompressibleEulerEquations(domain, parameters, active_tracers=tracers)

    # I/O
    output = OutputParameters(dirname=dirname+"/static_adjustment",
                              dumpfreq=1,
                              dumplist=['theta'])
    io = IO(domain, output, diagnostic_fields=[Perturbation('theta')])

    # Physics scheme
    physics_parametrisation = SuppressVerticalWind(eqn, spin_up_period)

    # Only want time derivatives and physics terms in equation, so drop the rest
    eqn.residual = eqn.residual.label_map(lambda t: any(t.has_label(time_derivative, physics_label)),
                                          map_if_true=identity, map_if_false=drop)
    if physics_coupling == "split":
        time_discretisation = ForwardEuler(domain)
        physics_schemes = [(physics_parametrisation, time_discretisation)]
        # Time stepper
        scheme = ForwardEuler(domain)
        stepper = SplitPhysicsTimestepper(eqn, scheme, io,
                                          physics_schemes=physics_schemes)
    else:
        # Time stepper
        scheme = ForwardEuler(domain, rk_formulation=RungeKuttaFormulation.predictor)
        stepper = Timestepper(eqn, scheme, io, physics_parametrisations=[physics_parametrisation])

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    # Declare prognostic fields
    u0 = stepper.fields("u")
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")

    # Set prognostic variables -- there is initially some vertical wind
    theta0.interpolate(Constant(300))
    rho0.interpolate(Constant(1.0))
    u0.project(as_vector([Constant(0.0), sin(pi*z/H)]))

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=dt)

    return domain, stepper


@pytest.mark.parametrize("physics_coupling", ["split", "nonsplit"])
def test_suppress_vertical_wind(tmpdir, physics_coupling):

    dirname = str(tmpdir)
    domain, stepper = run_suppress_vertical_wind(dirname, physics_coupling)

    u = stepper.fields('u')
    vertical_wind = Function(domain.spaces('theta'))
    vertical_wind.interpolate(dot(u, domain.k))

    tol = 1e-10
    assert norm(vertical_wind) < tol
