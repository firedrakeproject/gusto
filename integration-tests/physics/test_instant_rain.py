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

    # set up mesh
    L = 10
    nx = 10
    mesh = PeriodicSquareMesh(nx, nx, L)
    x, y = SpatialCoordinate(mesh)

    # parameters
    H = 30
    g = 10
    fexpr = Constant(0)
    dt = 0.1

    output = OutputParameters(dirname=dirname+"/instant_rain",
                              dumpfreq=1,
                              dumplist=['vapour', "rain"])

    parameters = ShallowWaterParameters(H=H, g=g)

    diagnostic_fields = [CourantNumber()]

    state = State(mesh,
                  dt=dt,
                  output=output,
                  diagnostic_fields=diagnostic_fields,
                  parameters=parameters)

    vapour = WaterVapour(name="water_vapour", space='DG')
    rain = Rain(name="rain", space="DG",
                transport_eqn=TransportEquationType.no_transport)

    VD = FunctionSpace(mesh, "DG", 1)

    eqns = ShallowWaterEquations(state, "BDM", 1, fexpr=fexpr,
                                 active_tracers=[vapour, rain])

    vapour0 = state.fields("water_vapour")

    # set up vapour
    xc = L/2
    yc = L/2
    rc = L/4
    r = sqrt((x - xc)**2 + (y - yc)**2)
    vapour_expr = conditional(r > rc, 0., 1 * (cos(pi * r / (rc * 2))) ** 2)

    vapour0.interpolate(vapour_expr)

    initial_vapour = Function(VD).interpolate(vapour_expr)

    # define saturation function
    saturation = Constant(0.5)

    # define expected solutions; vapour should be equal to saturation and rain
    # should be (initial vapour - saturation)
    vapour_true = Function(VD).interpolate(Constant(saturation))
    rain_true = Function(VD).interpolate(vapour0 - saturation)

    physics_schemes = [(InstantRain(eqns, saturation, rain_name="rain",
                                    set_tau_to_dt=True), ForwardEuler(state))]

    stepper = PrescribedTransport(eqns, RK4(state), state,
                                  physics_schemes=physics_schemes)

    stepper.run(t=0, tmax=5*dt)
    return state, saturation, initial_vapour, vapour_true, rain_true


def test_instant_rain_setup(tmpdir):
    dirname = str(tmpdir)
    state, saturation, initial_vapour, vapour_true, rain_true = run_instant_rain(dirname)
    v = state.fields("water_vapour")
    r = state.fields("rain")

    # check that the maximum of the vapour field is equal to the saturation
    assert v.dat.data.max() - saturation.values() < 0.001, "The maximum of the final vapour field should be equal to saturation"

    # check that the minimum of the vapour field hasn't changed
    assert v.dat.data.min() - initial_vapour.dat.data.min() < 0.001, "The minimum of the vapour field should not change"

    # check that the maximum of the excess vapour agrees with the maximum of the
    # rain
    VD = Function(v.function_space())
    excess_vapour = Function(VD).interpolate(initial_vapour - saturation)
    assert excess_vapour.dat.data.max() - r.dat.data.max() < 0.001

    # check that the minimum of the rain is 0
    assert r.dat.data.min() < 1e-8
