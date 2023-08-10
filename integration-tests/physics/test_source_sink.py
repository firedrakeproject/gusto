"""
This tests the source/sink
"""

from gusto import *
from firedrake import (Constant, PeriodicSquareMesh, SpatialCoordinate,
                       sqrt, conditional, cos, pi, FunctionSpace)


def run_source_sink(dirname):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # set up mesh and domain
    L = 10
    nx = 10
    mesh = PeriodicSquareMesh(nx, nx, L, quadrilateral=True)
    dt = 0.1
    domain = Domain(mesh, dt, "RTCF", 1)
    x, y = SpatialCoordinate(mesh)

    # Source is a Gaussian centred on a point
    centre_x = L / 4.0

    # Equation
    eqns = AdvectionEquation(domain, V, "ash")

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
    physics_schemes = [(SourceSink(eqns, saturation, rain_name="rain"),
                        ForwardEuler(domain))]

    # Time stepper
    stepper = PrescribedTransport(eqns, SSPRK3(domain), io, transport_method,
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

    # TODO: This test is based on the assumption that final vapour should be
    # equal to saturation, which might not be true when timestepping physics.
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
