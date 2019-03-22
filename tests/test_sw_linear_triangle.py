from os import path
from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector)
from math import pi
from netCDF4 import Dataset
import pytest


def setup_sw(dirname, scheme):
    refinements = 3  # number of horizontal cells = 20*(4^refinements)

    R = 6371220.
    H = 2000.
    day = 24.*60.*60.
    if scheme == "CrankNicolson":
        dt = 3600.
    elif scheme == "SSPRK3":
        dt = 500.

    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=refinements, degree=3)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    output = OutputParameters(dirname=dirname+"/sw_linear_w2", steady_state_error_fields=['u', 'D'], dumpfreq=12)
    parameters = ShallowWaterParameters(H=H)

    state = State(mesh, dt=dt,
                  output=output,
                  parameters=parameters)

    eqns = ShallowWaterEquations(state, family="BDM", degree=1, linear=True)

    # interpolate initial conditions
    # Initial/current conditions
    u0 = state.fields("u")
    D0 = state.fields("D")
    u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    Omega = parameters.Omega
    g = parameters.g
    Dexpr = - ((R * Omega * u_max)*(x[2]*x[2]/(R*R)))/g
    u0.project(uexpr)
    D0.interpolate(Dexpr)
    state.initialise([('u', u0),
                      ('D', D0)])

    # build time stepper
    if scheme == "CrankNicolson":
        advected_fields = []
        advected_fields.append(ForwardEuler(state, eqns, advection, field_name="D"))
        stepper = CrankNicolson(state, equation_set=eqns,
                                schemes=advected_fields)
    elif scheme == "SSPRK3":
        scheme = SSPRK3(state, eqns)
        stepper = Timestepper(state, scheme)

    return stepper, 2*day


def run_sw(dirname, scheme):

    stepper, tmax = setup_sw(dirname, scheme)
    stepper.run(t=0, tmax=tmax)


@pytest.mark.parametrize("scheme", ["CrankNicolson", "SSPRK3"])
def test_sw_linear(tmpdir, scheme):
    dirname = str(tmpdir)
    run_sw(dirname, scheme)
    filename = path.join(dirname, "sw_linear_w2/diagnostics.nc")
    data = Dataset(filename, "r")

    Derr = data.groups["D_error"]
    D = data.groups["D"]
    Dl2 = Derr["l2"][-1]/D["l2"][0]
    uerr = data.groups["u_error"]
    u = data.groups["u"]
    ul2 = uerr["l2"][-1]/u["l2"][0]

    assert Dl2 < 3.e-3
    assert ul2 < 6.e-2
