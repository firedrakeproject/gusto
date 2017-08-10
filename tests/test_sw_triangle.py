from os import path
from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, as_vector, \
    FunctionSpace, Function
from math import pi
import json
import pytest


def setup_sw(dirname, euler_poincare):

    refinements = 3  # number of horizontal cells = 20*(4^refinements)

    R = 6371220.
    H = 5960.
    day = 24.*60.*60.

    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=refinements)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    timestepping = TimesteppingParameters(dt=1500.)
    output = OutputParameters(dirname=dirname+"/sw", dumplist_latlon=['D', 'D_error'], steady_state_error_fields=['D', 'u'])
    parameters = {"H": H}
    diagnostic_fields = [PotentialVorticity()]

    model = ShallowWaterModel(mesh,
                              timestepping=timestepping,
                              output=output,
                              parameters=parameters,
                              diagnostic_fields=diagnostic_fields)
    state = model.state
    # Coriolis
    Omega = model.parameters.Omega
    fexpr = 2*Omega*x[2]/R
    V = FunctionSpace(mesh, "CG", 1)
    f = state.fields("coriolis", Function(V))
    f.interpolate(fexpr)  # Coriolis frequency (1/s)

    if euler_poincare:
        field_equations = {"u": EulerPoincare(state, state.spaces("HDiv"))}
        model.setup(field_equations=field_equations)
    else:
        sw_forcing = ShallowWaterForcing(state, model.parameters, euler_poincare=False)
        model.setup(forcing=sw_forcing)

    # interpolate initial conditions
    u0 = state.fields("u")
    D0 = state.fields("D")
    u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    g = model.parameters.g
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
    with open(path.join(dirname, "sw/diagnostics.json"), "r") as f:
        data = json.load(f)
    Dl2 = data["D_error"]["l2"][-1]/data["D"]["l2"][0]
    ul2 = data["u_error"]["l2"][-1]/data["u"]["l2"][0]

    assert Dl2 < 5.e-4
    assert ul2 < 5.e-3
