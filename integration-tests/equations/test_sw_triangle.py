"""
This tests the shallow water equations, with various combinations of settings.
The test uses the initial conditions of the Williamson 2 test and checks the
errors in the fields.

TODO: there's a lot of diagnostics! These should be tested individually using
unit-tests
"""

from os import path
from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, FunctionSpace, pi)
from netCDF4 import Dataset
import pytest

R = 6371220.
H = 5960.
day = 24.*60.*60.
u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)


def setup_sw(dirname, dt, u_transport_option):

    refinements = 3  # number of horizontal cells = 20*(4^refinements)

    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=refinements)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

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
                                    'SWPotentialEnstrophy_from_AbsoluteVorticity'),
                         MeridionalComponent('u'),
                         ZonalComponent('u'),
                         RadialComponent('u')]

    state = State(mesh,
                  dt=dt,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=diagnostic_fields)

    Omega = parameters.Omega
    fexpr = 2*Omega*x[2]/R
    eqns = ShallowWaterEquations(state, family="BDM", degree=1,
                                 fexpr=fexpr,
                                 u_transport_option=u_transport_option)

    # interpolate initial conditions
    u0 = state.fields("u")
    D0 = state.fields("D")
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    g = parameters.g
    Dexpr = H - ((R * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R*R)))/g

    u0.project(uexpr)
    D0.interpolate(Dexpr)

    vspace = FunctionSpace(state.mesh, "CG", 3)
    vexpr = (2*u_max/R)*x[2]/R
    f = state.fields("coriolis")
    vrel_analytical = state.fields("AnalyticalRelativeVorticity", vspace)
    vrel_analytical.interpolate(vexpr)
    vabs_analytical = state.fields("AnalyticalAbsoluteVorticity", vspace)
    vabs_analytical.interpolate(vexpr + f)
    pv_analytical = state.fields("AnalyticalPotentialVorticity", vspace)
    pv_analytical.interpolate((vexpr+f)/D0)

    return state, eqns


def check_results(dirname):
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

    # these checks are for the diagnostics of the velocity in spherical components
    tolerance = 0.05

    u_meridional = data.groups["u_meridional"]
    assert u_meridional["max"][0] < tolerance * u_max

    u_radial = data.groups["u_radial"]
    assert u_radial["max"][0] < tolerance * u_max

    u_zonal = data.groups["u_zonal"]
    assert u_max * (1 - tolerance) < u_zonal["max"][0] < u_max * (1 + tolerance)


@pytest.mark.parametrize("u_transport_option",
                         ["vector_invariant_form", "circulation_form",
                          "vector_advection_form"])
def test_sw_setup(tmpdir, u_transport_option):

    dirname = str(tmpdir)
    dt = 1500
    state, eqns = setup_sw(dirname, dt, u_transport_option)

    transported_fields = []
    transported_fields.append((ImplicitMidpoint(state, "u")))
    transported_fields.append((SSPRK3(state, "D")))
    stepper = SemiImplicitQuasiNewton(eqns, state, transported_fields)
    stepper.run(t=0, tmax=0.25*day)

    check_results(dirname)


@pytest.mark.parametrize("u_transport_option",
                         ["vector_invariant_form", "circulation_form",
                          "vector_advection_form"])
def test_sw_ssprk3(tmpdir, u_transport_option):

    dirname = str(tmpdir)
    dt = 100
    state, eqns = setup_sw(dirname, dt, u_transport_option)

    stepper = Timestepper(eqns, SSPRK3(state), state)
    stepper.run(t=0, tmax=0.01*day)

    check_results(dirname)
