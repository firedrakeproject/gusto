from os import path
from gusto import *
from firedrake import PeriodicIntervalMesh, Constant, \
    SpatialCoordinate, ExtrudedMesh, sqrt, \
    conditional, cos, as_vector
from netCDF4 import Dataset
from math import pi

# This setup creates a cloud of rain that falls at its
# terminal velocity, which is prescribed in the fallout
# method in physics.py. The test passes if there is no
# rain remaining at the end of the test.


def setup_fallout(dirname):

    # declare grid shape, with length L and height H
    L = 10.
    H = 10.
    nlayers = 10
    ncolumns = 10

    # make mesh
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=(H / nlayers))
    x = SpatialCoordinate(mesh)

    dt = 0.1
    output = OutputParameters(dirname=dirname+"/fallout",
                              dumpfreq=10,
                              dumplist=['rain'])
    parameters = CompressibleParameters()
    diagnostic_fields = [Precipitation()]
    state = State(mesh,
                  dt=dt,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=diagnostic_fields)

    Vt = state.spaces("theta", degree=1)
    Vrho = state.spaces("DG", "DG", 1)
    problem = [(AdvectionEquation(state, Vrho, "rho", ufamily="CG", udegree=1), ForwardEuler(state))]
    rho0 = state.fields("rho").assign(1.)

    physics_list = [Fallout(state)]
    rain0 = state.fields("rain")

    # set up rain
    xc = L / 2
    zc = H / 2
    rc = H / 4
    r = sqrt((x[0] - xc) ** 2 + (x[1] - zc) ** 2)
    rain_expr = conditional(r > rc, 0., 1e-3 * (cos(pi * r / (rc * 2))) ** 2)

    rain0.interpolate(rain_expr)

    def zero_u(t):
        return as_vector((Constant(0.), Constant(0.)))

    # build time stepper
    stepper = PrescribedAdvection(state, problem,
                                  physics_list=physics_list,
                                  prescribed_advecting_velocity=zero_u)

    return stepper, 10.0


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

    assert abs(final_rain) < 1e-4
    assert abs(final_rms_rain) < 1e-4
