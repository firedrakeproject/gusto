from os import path
from gusto import *
from firedrake import (SpatialCoordinate, sqrt, assemble,
                       conditional, cos, norm)
from netCDF4 import Dataset
from math import pi

# This setup creates a cloud of rain that falls at its
# terminal velocity, which is prescribed in the fallout
# method in physics.py. The test passes if there is no
# rain remaining at the end of the test.

def run(setup):

    state = setup.state
    tmax = setup.tmax
    Ld = setup.Ld
    x, z = SpatialCoordinate(state.mesh)

    u = state.fields("u", space=state.spaces("HDiv"))
    rho = state.fields("rho", space=state.spaces("DG"))
    rain = state.fields("rain", space=state.spaces("HDiv_v"), dump=True)
    rainfall_velocity = state.fields("rainfall_velocity", space=state.spaces("HDiv"))

    # set up rain
    xc = Ld / 4
    zc = Ld / 2
    rc = Ld / 4
    r = sqrt((x - xc) ** 2 + (z - zc) ** 2)
    rain_expr = conditional(r > rc, 5e-4, 5e-4 + 1e-3 * (cos(pi * r / (rc * 2))) ** 2)

    rho.assign(1.0)
    rain.interpolate(rain_expr)

    schemes = []

    physics_list = [Fallout(state, moments=AdvectedMoments.M0)]

    timestepper = PrescribedAdvectionTimestepper(
        state, schemes,
        physics_list=physics_list)
    timestepper.run(t=0, tmax=tmax)

    one = Function(rain.function_space()).assign(1.0)
    total_rain_final = assemble(rain * dx) / assemble(one * dx)
    total_rms_rain = norm(rain) / assemble(one * dx)

    return total_rain_final, total_rms_rain

def test_precipitation(tmpdir, moist_setup):

    setup = moist_setup(tmpdir, "narrow")
    total_rain_final, total_rms_rain = run(setup)
    # check final amount of rain is very small
    assert total_rain_final < 1e-4
    # check final rms rain
    assert total_rms_rain < 1e-4
