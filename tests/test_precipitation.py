from os import path
from gusto import *
from firedrake import PeriodicIntervalMesh, \
    SpatialCoordinate, ExtrudedMesh, sqrt, \
    conditional, cos
from netCDF4 import Dataset
from math import pi

# This setup creates a cloud of rain that falls at its
# terminal velocity, which is prescribed in the fallout
# method in physics.py. The test passes if there is no
# rain remaining at the end of the test.


def setup_fallout(dirname):

    # declare grid shape, with length L and height H
    L = 400.
    H = 1000.
    nlayers = 25
    ncolumns = 10

    # make mesh
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=(H / nlayers))
    x = SpatialCoordinate(mesh)

    fieldlist = ['u', 'rho', 'theta', 'rain']
    timestepping = TimesteppingParameters(dt=1.0, maxk=4, maxi=1)
    output = OutputParameters(dirname=dirname+"/fallout",
                              dumpfreq=1,
                              dumplist=['rain'])
    parameters = CompressibleParameters()
    diagnostic_fields = [Precipitation()]
    state = State(mesh, vertical_degree=1, horizontal_degree=1,
                  family="CG",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  fieldlist=fieldlist,
                  diagnostic_fields=diagnostic_fields)

    # declare initial fields
    u0 = state.fields("u")
    rho0 = state.fields("rho")
    theta0 = state.fields("theta")

    # spaces
    Vt = theta0.function_space()

    # declare tracer field and a background field
    rain0 = state.fields("rain", Vt)

    # set up rain
    xc = L / 2
    zc = 800.
    rc = 150.
    r = sqrt((x[0] - xc) ** 2 + (x[1] - zc) ** 2)
    rain_expr = conditional(r > rc, 0., 1e-5 * (cos(pi * r / (rc * 2))) ** 2)

    rain0.interpolate(rain_expr)

    rho0.assign(1.0)

    state.initialise([('u', u0),
                      ('rho', rho0),
                      ('rain', rain0)])

    # build advection dictionary
    advected_fields = []
    advected_fields.append(("u", NoAdvection(state, u0, None)))
    advected_fields.append(("rho", NoAdvection(state, rho0, None)))
    advected_fields.append(("rain", NoAdvection(state, rain0, None)))

    physics_list = [Fallout(state)]

    # build time stepper
    stepper = AdvectionDiffusion(state, advected_fields, physics_list=physics_list)

    return stepper, 50.0


def run_fallout(dirname):

    stepper, tmax = setup_fallout(dirname)
    stepper.run(t=0, tmax=tmax)


def test_fallout_setup(tmpdir):

    dirname = str(tmpdir)
    run_fallout(dirname)
    filename = path.join(dirname, "fallout/diagnostics.nc")
    data = Dataset(filename, "r")

    rain = data.groups["rain"]
    final_rain = rain.variables["total"][-1]
    final_rms_rain = rain.variables["rms"][-1]

    assert abs(final_rain) < 1e-12
    assert abs(final_rms_rain) < 1e-12
