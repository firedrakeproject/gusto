from os import path
from gusto import *
from firedrake import SpatialCoordinate, as_vector
from math import pi
from netCDF4 import Dataset
import pytest


def setup_sw(dirname, euler_poincare):

    refinements = 3  # number of horizontal cells = 20*(4^refinements)

    R = 6371220.
    H = 5960.
    day = 24.*60.*60.

    physical_domain = Sphere(radius=R, ref_level=refinements)

    output = OutputParameters(dirname=dirname+"/sw", dumplist_latlon=['D', 'D_error'], steady_state_error_fields=['D', 'u'])

    diagnostic_fields = [PotentialVorticity()]

    state = ShallowWaterState(physical_domain.mesh,
                              output=output,
                              diagnostic_fields=diagnostic_fields)

    timestepping = TimesteppingParameters(dt=1500.)
    advected_fields = []
    if euler_poincare:
        ueqn = EulerPoincare(physical_domain, state.spaces("HDiv"), state.spaces("HDiv"))
        advected_fields.append(("u", ThetaMethod(state.fields("u"),
                                                 timestepping.dt,
                                                 ueqn)))

    model = ShallowWaterModel(state,
                              physical_domain,
                              parameters=ShallowWaterParameters(H=H),
                              timestepping=timestepping,
                              advected_fields=advected_fields)

    # interpolate initial conditions
    x = SpatialCoordinate(physical_domain.mesh)
    u0 = state.fields("u")
    D0 = state.fields("D")
    u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    g = model.parameters.g
    Omega = model.parameters.Omega
    Dexpr = H - ((R * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R*R)))/g

    u0.project(uexpr)
    D0.interpolate(Dexpr)
    state.initialise([('u', u0),
                      ('D', D0)])

    # build time stepper
    stepper = Timestepper(model)

    return stepper, 0.25*day


def run_sw(dirname, euler_poincare):

    stepper, tmax = setup_sw(dirname, euler_poincare)
    stepper.run(t=0, tmax=tmax)


@pytest.mark.parametrize("euler_poincare", [True, False])
def test_sw_setup(tmpdir, euler_poincare):

    dirname = str(tmpdir)
    run_sw(dirname, euler_poincare=euler_poincare)
    filename = path.join(dirname, "sw/diagnostics.nc")
    data = Dataset(filename, "r")

    Derr = data.groups["D_error"]
    D = data.groups["D"]
    Dl2 = Derr["l2"][-1]/D["l2"][0]
    uerr = data.groups["u_error"]
    u = data.groups["u"]
    ul2 = uerr["l2"][-1]/u["l2"][0]

    assert Dl2 < 5.e-4
    assert ul2 < 5.e-3
