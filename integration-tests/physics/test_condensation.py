"""
# This tests the Condensation routine. It creates a bubble of water vapour that
# is advected by a prescribed velocity. The test passes if the integral
# of the water mixing ratio is conserved.
"""

from os import path
from gusto import *
import gusto.thermodynamics as td
from firedrake import (norm, Constant, PeriodicIntervalMesh,
                       SpatialCoordinate, ExtrudedMesh, Function, sqrt,
                       conditional)
from netCDF4 import Dataset
import pytest


def run_cond_evap(dirname, process):
    # declare grid shape, with length L and height H
    L = 1000.
    H = 1000.
    nlayers = int(H / 10.)
    ncolumns = int(L / 10.)

    # make mesh
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=(H / nlayers))
    x, z = SpatialCoordinate(mesh)

    dt = 2.0
    tmax = dt
    output = OutputParameters(dirname=dirname+"/cond_evap",
                              dumpfreq=1,
                              dumplist=['u'])
    parameters = CompressibleParameters()

    state = State(mesh,
                  dt=dt,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=[Sum('vapour_mixing_ratio', 'cloud_liquid_mixing_ratio')])

    # spaces
    Vt = state.spaces("theta", degree=1)
    Vr = state.spaces("DG", "DG", degree=1)

    # Set up equation -- use compressible to set up these spaces
    # However the equation itself will be unused
    _ = CompressibleEulerEquations(state, "CG", 1)

    # Declare prognostic fields
    rho0 = state.fields("rho")
    theta0 = state.fields("theta")
    water_v0 = state.fields("vapour_mixing_ratio", Vt)
    water_c0 = state.fields("cloud_liquid_mixing_ratio", Vt)

    # Set a background state with constant pressure and temperature
    pressure = Function(Vr).interpolate(Constant(100000.))
    temperature = Function(Vt).interpolate(Constant(300.))
    theta_d = td.theta(parameters, temperature, pressure)
    mv_sat = td.r_v(parameters, Constant(1.0), temperature, pressure)
    Lv_over_cpT = td.Lv(parameters, temperature) / (parameters.cp * temperature)

    # Apply perturbation
    xc = L / 2.
    zc = H / 2.
    rc = L / 4.
    r = sqrt((x-xc)**2 + (z-zc)**2)
    pert = conditional(r < rc, 1.0, 0.0)

    if process == "evaporation":
        water_v0.interpolate(0.96*mv_sat)
        water_c0.interpolate(0.005*mv_sat*pert)
        # Approximate answers
        # Rate of change is roughly (m_sat - m_v) / 4 so should evaporate everything
        mc_true = Function(Vt).interpolate(Constant(0.0))
        theta_d_true = Function(Vt).interpolate(theta_d + 0.005*mv_sat*pert*Lv_over_cpT)
        mv_true = Function(Vt).interpolate(mv_sat*(0.96 + 0.005*pert))
    elif process == "condensation":
        water_v0.interpolate(mv_sat*(1.0 + 0.04*pert))
        # Approximate answers -- rate of change is roughly (m_v - m_sat) / 4
        mc_true = Function(Vt).interpolate(0.01*mv_sat*pert)
        theta_d_true = Function(Vt).interpolate(theta_d - 0.01*mv_sat*pert*Lv_over_cpT)
        mv_true = Function(Vt).interpolate(mv_sat*(1.0 + 0.03*pert))

    # Set prognostic variables
    theta0.project(theta_d*(1 + water_v0 * parameters.R_v / parameters.R_d))
    rho0.interpolate(pressure / (temperature*parameters.R_d * (1 + water_v0 * parameters.R_v / parameters.R_d)))
    mc_init = Function(Vt).assign(water_c0)

    # Have empty problem as only thing is condensation / evaporation
    problem = []
    physics_list = [Condensation(state)]

    # build time stepper
    stepper = PrescribedTransport(state, problem,
                                  physics_list=physics_list)

    stepper.run(t=0, tmax=tmax)

    return state, mv_true, mc_true, theta_d_true, mc_init


@pytest.mark.parametrize("process", ["evaporation", "condensation"])
def test_cond_evap(tmpdir, process):

    dirname = str(tmpdir)
    state, mv_true, mc_true, theta_d_true, mc_init = run_cond_evap(dirname, process)

    water_v = state.fields('vapour_mixing_ratio')
    water_c = state.fields('cloud_liquid_mixing_ratio')
    theta_vd = state.fields('theta')
    theta_d = Function(theta_vd.function_space())
    theta_d.interpolate(theta_vd/(1 + water_v * state.parameters.R_v / state.parameters.R_d))

    # Check that water vapour is approximately equal to saturation amount
    assert norm(water_v - mv_true) / norm(mv_true) < 0.01, \
        f'Final vapour field is incorrect for {process}'

    # Check that cloud has been created / removed
    denom = norm(mc_true) if process == "condensation" else norm(mc_init)
    assert norm(water_c - mc_true) / denom < 0.1, \
        f'Final cloud field is incorrect for {process}'

    # Check that theta pertubation has correct sign and size
    assert norm(theta_d - theta_d_true) / norm(theta_d_true) < 0.01, \
        f'Latent heating is incorrect for {process}'

    # Check that total moisture conserved
    filename = path.join(dirname, "cond_evap/diagnostics.nc")
    data = Dataset(filename, "r")

    water = data.groups["vapour_mixing_ratio_plus_cloud_liquid_mixing_ratio"]
    total = water.variables["total"]
    water_t_0 = total[0]
    water_t_T = total[-1]

    assert abs(water_t_0 - water_t_T) / water_t_0 < 1e-12, \
        f'Total amount of water should be conserved by {process}'
