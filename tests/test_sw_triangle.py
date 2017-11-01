from os import path
from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, as_vector, \
    FunctionSpace, Function
from math import pi
from netCDF4 import Dataset
import pytest


def setup_sw(dirname, euler_poincare, timestepper):

    refinements = 3  # number of horizontal cells = 20*(4^refinements)

    R = 6371220.
    H = 5960.
    day = 24.*60.*60.

    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=refinements)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    fieldlist = ['u', 'D']
    timestepping = TimesteppingParameters(dt=1500.)
    output = OutputParameters(dirname=dirname+"/sw", dumplist_latlon=['D', 'D_error'], steady_state_error_fields=['D', 'u'])
    parameters = ShallowWaterParameters(H=H)
    diagnostic_fields = [RelativeVorticity(), AbsoluteVorticity(),
                         PotentialVorticity(),
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

    state = State(mesh, vertical_degree=None, horizontal_degree=1,
                  family="BDM",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=diagnostic_fields,
                  fieldlist=fieldlist)

    # interpolate initial conditions
    u0 = state.fields("u")
    D0 = state.fields("D")
    u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    Omega = parameters.Omega
    g = parameters.g
    Dexpr = H - ((R * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R*R)))/g
    # Coriolis
    fexpr = 2*Omega*x[2]/R
    V = FunctionSpace(mesh, "CG", 1)
    f = state.fields("coriolis", Function(V))
    f.interpolate(fexpr)  # Coriolis frequency (1/s)

    u0.project(uexpr)
    D0.interpolate(Dexpr)
    state.initialise([('u', u0),
                      ('D', D0)])

    if euler_poincare:
        ueqn = EulerPoincare(state, u0.function_space())
        sw_forcing = ShallowWaterForcing(state)
    else:
        ueqn = VectorInvariant(state, state.spaces("HDiv"))
        sw_forcing = ShallowWaterForcing(state, euler_poincare=False)

    Deqn = AdvectionEquation(state, D0.function_space(), equation_form="continuity")
    advected_fields = []
    advected_fields.append(("D", SSPRK3(state, D0, Deqn)))
    if timestepper == "implicit_midpoint":
        advected_fields.append(("u", SSPRK3(state, u0, ueqn, forcing=sw_forcing)))
    else:
        advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))

    linear_solver = ShallowWaterSolver(state)

    # build time stepper
    if timestepper == "implicit_midpoint":
        stepper = ImplicitMidpoint(state, advected_fields, linear_solver)
    else:
        stepper = CrankNicolson(state, advected_fields, linear_solver,
                                sw_forcing)

    vspace = FunctionSpace(state.mesh, "CG", 3)
    vexpr = (2*u_max/R)*x[2]/R
    vrel_analytical = state.fields("AnalyticalRelativeVorticity", vspace)
    vrel_analytical.interpolate(vexpr)
    vabs_analytical = state.fields("AnalyticalAbsoluteVorticity", vspace)
    vabs_analytical.interpolate(vexpr + f)
    pv_analytical = state.fields("AnalyticalPotentialVorticity", vspace)
    pv_analytical.interpolate((vexpr+f)/D0)

    return stepper, 0.25*day


def run_sw(dirname, euler_poincare, timestepper=None):

    stepper, tmax = setup_sw(dirname, euler_poincare, timestepper)
    stepper.run(t=0, tmax=tmax)


def check_vorticity(data):
    # these 3 checks are for the diagnostic field so the checks are
    # made for values at the beginning of the run:
    vrel_err = data.groups["RelativeVorticity_minus_AnalyticalRelativeVorticity"]
    assert vrel_err["max"][0] < 6.e-7

    vabs_err = data.groups["AbsoluteVorticity_minus_AnalyticalAbsoluteVorticity"]
    assert vabs_err["max"][0] < 6.e-7

    pv_err = data.groups["PotentialVorticity_minus_AnalyticalPotentialVorticity"]
    assert pv_err["max"][0] < 1.e-10


def check_enstrophy(data):
    # these 2 checks confirm that the potential enstrophy is the same
    # when it is calculated using the pv field, the relative vorticity
    # field or the absolute vorticity field
    enstrophy_diff = data.groups["SWPotentialEnstrophy_from_PotentialVorticity_minus_SWPotentialEnstrophy_from_RelativeVorticity"]
    assert enstrophy_diff["max"][-1] < 1.e-15

    enstrophy_diff = data.groups["SWPotentialEnstrophy_from_PotentialVorticity_minus_SWPotentialEnstrophy_from_AbsoluteVorticity"]
    assert enstrophy_diff["max"][-1] < 1.e-15


def check_output(data):

    Derr = data.groups["D_error"]
    D = data.groups["D"]
    Dl2 = Derr["l2"][-1]/D["l2"][0]
    assert Dl2 < 5.e-4

    uerr = data.groups["u_error"]
    u = data.groups["u"]
    ul2 = uerr["l2"][-1]/u["l2"][0]
    assert ul2 < 5.e-3


@pytest.mark.parametrize("euler_poincare", [True, False])
def test_sw_setup(tmpdir, euler_poincare):

    dirname = str(tmpdir)
    run_sw(dirname, euler_poincare=euler_poincare)
    filename = path.join(dirname, "sw/diagnostics.nc")
    data = Dataset(filename, "r")
    check_output(data)
    check_vorticity(data)
    check_enstrophy(data)


def test_sw_implicit_midpoint(tmpdir):

    dirname = str(tmpdir)
    run_sw(dirname, euler_poincare=False, timestepper="implicit_midpoint")
    filename = path.join(dirname, "sw/diagnostics.nc")
    data = Dataset(filename, "r")
    check_output(data)
    check_vorticity(data)
    check_enstrophy(data)
