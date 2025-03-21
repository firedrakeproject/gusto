"""
This tests the scheme that converts water vapour directly to rain. It creates a
bubble of vapour and any above a specified saturation function is converted to
rain. The test passes if the the maximum of the vapour is equivalent to the
saturation function, the minimum of the vapour is unchanged, the maximum of the
vapour above saturation agrees with the maximum of the created rain, and the
minimum of the rain is zero.
"""

from gusto import *
from firedrake import (Constant, PeriodicSquareMesh, SpatialCoordinate,
                       sqrt, conditional, cos, pi, FunctionSpace)
import pytest


def run_instant_rain(dirname, physics_coupling):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # set up mesh and domain
    L = 10
    nx = 10
    mesh = PeriodicSquareMesh(nx, nx, L)
    dt = 0.1
    domain = Domain(mesh, dt, "BDM", 1)
    x, y = SpatialCoordinate(mesh)

    # parameters
    H = 30
    g = 10
    fexpr = Constant(0)

    # Equation
    # TODO: This should become a CoupledAdvectionEquation. Currently we set u
    # and D to 0 in the ShallowWaterEquations so they do not evolve.
    vapour = WaterVapour(name="water_vapour", space='DG')
    rain = Rain(name="rain", space="DG",
                transport_eqn=TransportEquationType.no_transport)

    parameters = ShallowWaterParameters(mesh, H=H, g=g)
    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr,
                                 active_tracers=[vapour, rain])

    # I/O
    output = OutputParameters(dirname=dirname+"/instant_rain",
                              dumpfreq=1,
                              dumplist=['vapour', "rain"])
    diagnostic_fields = [CourantNumber()]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)
    transport_method = [DGUpwind(eqns, "water_vapour")]

    # Physics schemes
    # define saturation function
    saturation = Constant(0.5)
    if physics_coupling == "split":
        physics_schemes = [(InstantRain(eqns, saturation, rain_name="rain"),
                            RK4(domain))]
        # Time stepper
        stepper = SplitPhysicsTimestepper(eqns, RK4(domain), io, transport_method,
                                          physics_schemes=physics_schemes)
    else:
        physics_parametrisation = [InstantRain(eqns, saturation, rain_name="rain")]
        scheme = RK4(domain, rk_formulation=RungeKuttaFormulation.predictor)
        # Time stepper
        stepper = Timestepper(eqns, scheme, io, transport_method,
                              physics_parametrisations=physics_parametrisation)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    vapour0 = stepper.fields("water_vapour")

    # set up vapour
    xc = L/2
    yc = L/2
    rc = L/4
    r = sqrt((x - xc)**2 + (y - yc)**2)
    vapour_expr = conditional(r > rc, 0., 1 * (cos(pi * r / (rc * 2))) ** 2)

    vapour0.interpolate(vapour_expr)

    VD = FunctionSpace(mesh, "DG", 1)
    initial_vapour = Function(VD).interpolate(vapour_expr)

    # TODO: This test is based on the assumption that final vapour should be
    # equal to saturation, which might not be true when timestepping physics.
    vapour_true = Function(VD).interpolate(saturation)
    rain_true = Function(VD).interpolate(vapour0 - saturation)

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=5*dt)
    return stepper, saturation, initial_vapour, vapour_true, rain_true


@pytest.mark.parametrize("physics_coupling", ["split", "nonsplit"])
def test_instant_rain_setup(tmpdir, physics_coupling):
    dirname = str(tmpdir)
    stepper, saturation, initial_vapour, vapour_true, rain_true = run_instant_rain(dirname,
                                                                                   physics_coupling)
    v = stepper.fields("water_vapour")
    r = stepper.fields("rain")

    # check that the maximum of the vapour field is equal to the saturation
    assert v.dat.data.max() - saturation.values() < 0.1, "The maximum of the final vapour field should be equal to saturation"

    # check that the minimum of the vapour field hasn't changed
    assert v.dat.data.min() - initial_vapour.dat.data.min() < 0.1, "The minimum of the vapour field should not change"

    # check that the maximum of the excess vapour agrees with the maximum of the
    # rain
    VD = Function(v.function_space())
    excess_vapour = Function(VD).interpolate(initial_vapour - saturation)
    assert excess_vapour.dat.data.max() - r.dat.data.max() < 0.1

    # check that the minimum of the rain is 0
    assert r.dat.data.min() < 1e-8
