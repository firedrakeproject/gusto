from os import path
from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, FunctionSpace)
from math import pi
from netCDF4 import Dataset
import pytest


def setup_sw(dirname, scheme):

    refinements = 3  # number of horizontal cells = 20*(4^refinements)

    R = 6371220.
    H = 5960.
    day = 24.*60.*60.

    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=refinements)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    if scheme == "CrankNicolson":
        dt = 1500.
    else:
        dt = 500.
    output = OutputParameters(dirname=dirname+"/sw", steady_state_error_fields=['D', 'u'])
    parameters = ShallowWaterParameters(H=H)
    diagnostic_fields = [RelativeVorticity(), AbsoluteVorticity(),
                         PotentialVorticity(), CourantNumber(),
                         ShallowWaterPotentialEnstrophy('RelativeVorticity'),
                         ShallowWaterPotentialEnstrophy('AbsoluteVorticity'),
                         ShallowWaterPotentialEnstrophy('PotentialVorticity'),
                         Difference('RelativeVorticity',
                                    'AnalyticalRelativeVorticity'),
                         Difference('AbsoluteVorticity',
                                    'AnalyticalAbsoluteVorticity'),
                         Difference('PotentialVorticity',
                                    'AnalyticalPotentialVorticity'),
                         Difference('SWPotentialEnstrophy_from_PotentialVorticity',
                                    'SWPotentialEnstrophy_from_RelativeVorticity'),
                         Difference('SWPotentialEnstrophy_from_PotentialVorticity',
                                    'SWPotentialEnstrophy_from_AbsoluteVorticity')]

    state = State(mesh, dt,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=diagnostic_fields)

    eqns = ShallowWaterEquations(state, family="BDM", degree=1)

    # interpolate initial conditions
    u0 = state.fields("u")
    D0 = state.fields("D")
    u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    Omega = parameters.Omega
    g = parameters.g
    Dexpr = H - ((R * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R*R)))/g

    u0.project(uexpr)
    D0.interpolate(Dexpr)
    state.initialise([('u', u0),
                      ('D', D0)])

    # build time stepper
    if scheme == "CrankNicolson":
        advected_fields = []
        advected_fields.append(("u", ThetaMethod(state, u0, eqns)))
        advected_fields.append(("D", SSPRK3(state, D0, eqns)))
        stepper = CrankNicolson(state, equations=eqns,
                                advected_fields=advected_fields)
    elif scheme == "ImplicitMidpoint":
        scheme = ThetaMethod(state, state.fields.X, eqns)
        stepper = Timestepper(state, equations=eqns, schemes=[("X", scheme)])
    elif scheme == "SSPRK3":
        scheme = SSPRK3(state, state.fields.X, eqns)
        stepper = Timestepper(state, equations=eqns, schemes=[("X", scheme)])

    vspace = FunctionSpace(state.mesh, "CG", 3)
    vexpr = (2*u_max/R)*x[2]/R
    f = state.fields("coriolis")
    vrel_analytical = state.fields("AnalyticalRelativeVorticity", vspace)
    vrel_analytical.interpolate(vexpr)
    vabs_analytical = state.fields("AnalyticalAbsoluteVorticity", vspace)
    vabs_analytical.interpolate(vexpr + f)
    pv_analytical = state.fields("AnalyticalPotentialVorticity", vspace)
    pv_analytical.interpolate((vexpr+f)/D0)

    return stepper, 0.25*day


def run_sw(dirname, scheme):

    stepper, tmax = setup_sw(dirname, scheme)
    stepper.run(t=0, tmax=tmax)


@pytest.mark.parametrize("scheme", ["CrankNicolson", "ImplicitMidpoint", "SSPRK3"])
def test_sw_setup(tmpdir, scheme):

    dirname = str(tmpdir)
    run_sw(dirname, scheme)
    filename = path.join(dirname, "sw/diagnostics.nc")
    data = Dataset(filename, "r")

    Derr = data.groups["D_error"]
    D = data.groups["D"]
    Dl2 = Derr["l2"][-1]/D["l2"][0]
    assert Dl2 < 5.e-4

    uerr = data.groups["u_error"]
    u = data.groups["u"]
    ul2 = uerr["l2"][-1]/u["l2"][0]
    assert ul2 < 5.e-3

    # these 3 checks are for the diagnostic field so the checks are
    # made for values at the beginning of the run:
    vrel_err = data.groups["RelativeVorticity_minus_AnalyticalRelativeVorticity"]
    assert vrel_err["max"][0] < 6.e-7

    vabs_err = data.groups["AbsoluteVorticity_minus_AnalyticalAbsoluteVorticity"]
    assert vabs_err["max"][0] < 6.e-7

    pv_err = data.groups["PotentialVorticity_minus_AnalyticalPotentialVorticity"]
    assert pv_err["max"][0] < 1.e-10

    # these 2 checks confirm that the potential enstrophy is the same
    # when it is calculated using the pv field, the relative vorticity
    # field or the absolute vorticity field
    enstrophy_diff = data.groups["SWPotentialEnstrophy_from_PotentialVorticity_minus_SWPotentialEnstrophy_from_RelativeVorticity"]
    assert enstrophy_diff["max"][-1] < 1.e-15

    enstrophy_diff = data.groups["SWPotentialEnstrophy_from_PotentialVorticity_minus_SWPotentialEnstrophy_from_AbsoluteVorticity"]
    assert enstrophy_diff["max"][-1] < 1.e-15
