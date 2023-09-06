"""
This tests the physics routine to apply static adjustment. A column initially
has a decreasing theta profile (which would be unstable). The static adjustment
should then sort this to make it increasing with height.
"""

from gusto import *
from gusto.labels import physics_label
from firedrake import (Constant, PeriodicIntervalMesh,
                       SpatialCoordinate, ExtrudedMesh, Function)
import pytest


def run_static_adjustment(dirname, theta_variable):

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
    domain.coords.register_space(domain, 'theta')

    _, z = SpatialCoordinate(mesh)

    # Set up equation
    tracers = [WaterVapour()]
    parameters = CompressibleParameters()
    eqn = CompressibleEulerEquations(domain, parameters, active_tracers=tracers)

    # I/O
    output = OutputParameters(dirname=dirname+"/static_adjustment",
                              dumpfreq=1,
                              dumplist=['theta'])
    io = IO(domain, output, diagnostic_fields=[Perturbation('theta')])

    # Physics scheme
    physics_parametrisation = StaticAdjustment(eqn, theta_variable)

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

    # Declare prognostic fields
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")

    # Set prognostic variables -- decreasing theta profile
    water_v0 = stepper.fields("water_vapour")
    water_v0.interpolate(Constant(0.01) + 0.001*z/H)
    theta0.interpolate(Constant(300) - 20*z/H)
    rho0.interpolate(Constant(1.0))

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=dt)

    return domain, eqn, stepper


@pytest.mark.parametrize("theta_variable", ["theta", "theta_vd"])
def test_static_adjustment(tmpdir, theta_variable):

    dirname = str(tmpdir)
    domain, eqn, stepper = run_static_adjustment(dirname, theta_variable)

    # Back out temperature from prognostic fields
    theta_vd = stepper.fields('theta')
    if theta_variable == 'theta':
        Rv = eqn.parameters.R_v
        Rd = eqn.parameters.R_d
        mv = stepper.fields('water_vapour')
        theta = Function(theta_vd.function_space())
        theta.interpolate(theta_vd / (1 + Rv*mv/Rd))
    else:
        theta = theta_vd

    column_data, _ = domain.coords.get_column_data(theta)

    # Check first column
    is_increasing = all(i < j for i, j in zip(column_data[0, :], column_data[0, 1:]))
    assert is_increasing, \
        'static adjustment has not worked: data in column is not increasing'
