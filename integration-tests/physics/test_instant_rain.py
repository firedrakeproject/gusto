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


def run_instant_rain(dirname):

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
    vapour = WaterVapour(name="water_vapour", space='DG')
    rain = Rain(name="rain", space="DG",
                transport_eqn=TransportEquationType.no_transport)

    parameters = ShallowWaterParameters(H=H, g=g)
    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr,
                                 active_tracers=[vapour, rain])

    # I/O
    output = OutputParameters(dirname=dirname+"/instant_rain",
                              dumpfreq=1,
                              dumplist=['vapour', "rain"])
    diagnostic_fields = [CourantNumber()]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Physics schemes
    # define saturation function
    saturation = Constant(0.5)
    physics_schemes = [(InstantRain(eqns, saturation, rain_name="rain",
                                    set_tau_to_dt=True), ForwardEuler(domain))]

    # Time stepper
    stepper = PrescribedTransport(eqns, RK4(domain), io,
                                  physics_schemes=physics_schemes)

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

    # define expected solutions; vapour should be equal to saturation and rain
    # should be (initial vapour - saturation)
    vapour_true = Function(VD).interpolate(saturation)
    rain_true = Function(VD).interpolate(vapour0 - saturation)

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=5*dt)
    return stepper, saturation, initial_vapour, vapour_true, rain_true


def test_instant_rain_setup(tmpdir):
    dirname = str(tmpdir)
    stepper, saturation, initial_vapour, vapour_true, rain_true = run_instant_rain(dirname)
    v = stepper.fields("water_vapour")
    r = stepper.fields("rain")

    # check that the maximum of the vapour field is equal to the saturation
    assert v.dat.data.max() - saturation.dat.data.max() < 0.001, "The maximum of the final vapour field should be equal to saturation"

    # check that the minimum of the vapour field hasn't changed
    assert v.dat.data.min() - initial_vapour.dat.data.min() < 0.001, "The minimum of the vapour field should not change"

    # check that the maximum of the excess vapour agrees with the maximum of the
    # rain
    VD = Function(v.function_space())
    excess_vapour = Function(VD).interpolate(initial_vapour - saturation)
    assert excess_vapour.dat.data.max() - r.dat.data.max() < 0.001

    # check that the minimum of the rain is 0
    assert r.dat.data.min() < 1e-8
