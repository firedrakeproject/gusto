"""
# This tests the SW_AdjustableSaturation physics class. In the first scenario
# it creates a cloud in a subsaturated atmosphere that should evaporate.
# In the second test it creates a bubble of water vapour that is advected by
# a prescribed velocity and should be converted to cloud where it exceeds a
# saturation threshold. The first test passes if the cloud is zero, the
# vapour has increased, the buoyancy field has increased and the total moisture
# is conserved. The second test passes if cloud is non-zero, vapour has
# decreased, buoyancy has decreased and total moisture is conserved.
"""

from os import path
from gusto import *
from firedrake import (IcosahedralSphereMesh, acos, sin, cos, Constant, norm,
                       max_value, min_value)
from netCDF4 import Dataset
import pytest


def run_sw_cond_evap(dirname, process):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Parameters
    dt = 100
    R = 6371220.
    H = 100
    theta_c = pi
    lamda_c = pi/2
    rc = R/4
    beta2 = 10

    # Domain
    mesh = IcosahedralSphereMesh(radius=R, refinement_level=3, degree=2)
    degree = 1
    domain = Domain(mesh, dt, 'BDM', degree)
    x = SpatialCoordinate(mesh)
    lamda, theta, _ = lonlatr_from_xyz(x[0], x[1], x[2])

    # saturation field (constant everywhere)
    sat = 100

    # Equation
    parameters = ShallowWaterParameters(H=H)
    Omega = parameters.Omega
    fexpr = 2*Omega*x[2]/R

    tracers = [WaterVapour(space='DG'), CloudWater(space='DG')]

    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr,
                                 u_transport_option='vector_advection_form',
                                 thermal=True, active_tracers=tracers)

    # I/O
    output = OutputParameters(dirname=dirname+"/sw_cond_evap",
                              dumpfreq=1)
    io = IO(domain, output,
            diagnostic_fields=[Sum('water_vapour', 'cloud_water')])

    # Physics schemes
    physics_schemes = [(SWSaturationAdjustment(eqns, sat,
                                               parameters=parameters,
                                               thermal_feedback=True,
                                               beta2=beta2),
                        ForwardEuler(domain))]

    # Timestepper
    stepper = SplitPhysicsTimestepper(eqns, RK4(domain), io,
                                      physics_schemes=physics_schemes)

    # Initial conditions
    b0 = stepper.fields("b")
    v0 = stepper.fields("water_vapour")
    c0 = stepper.fields("cloud_water")

    # perturbation
    r = R * (
        acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lamda-lamda_c)))
    pert = conditional(r < rc, 1.0, 0.0)

    if process == "evaporation":
        # atmosphere is subsaturated and cloud is present
        v0.interpolate(0.96*Constant(sat))
        c0.interpolate(0.005*sat*pert)
        # lose cloud and add this to vapour
        v_true = Function(v0.function_space()).interpolate(sat*(0.96+0.005*pert))
        c_true = Function(c0.function_space()).interpolate(Constant(0.0))
        # gain buoyancy
        factor = parameters.g*beta2
        sat_adj_expr = (v0 - sat) / dt
        sat_adj_expr = conditional(sat_adj_expr < 0,
                                   max_value(sat_adj_expr, -c0 / dt),
                                   min_value(sat_adj_expr, v0 / dt))
        # include factor of -1 in true solution to compare term to LHS in Gusto
        b_true = Function(b0.function_space()).interpolate(-dt*sat_adj_expr*factor)

    elif process == "condensation":
        # vapour is above saturation
        v0.interpolate(sat*(1.0 + 0.04*pert))
        # lose vapour and add this to cloud
        v_true = Function(v0.function_space()).interpolate(Constant(sat))
        c_true = Function(c0.function_space()).interpolate(v0 - sat)
        # lose buoyancy
        factor = parameters.g*beta2
        sat_adj_expr = (v0 - sat) / dt
        sat_adj_expr = conditional(sat_adj_expr < 0,
                                   max_value(sat_adj_expr, -c0 / dt),
                                   min_value(sat_adj_expr, v0 / dt))
        # include factor of -1 in true solution to compare term to LHS in Gusto
        b_true = Function(b0.function_space()).interpolate(-dt*sat_adj_expr*factor)

    c_init = Function(c0.function_space()).interpolate(c0)

    # Run
    stepper.run(t=0, tmax=dt)

    return eqns, stepper, v_true, c_true, b_true, c_init


@pytest.mark.parametrize("process", ["evaporation", "condensation"])
def test_cond_evap(tmpdir, process):

    dirname = str(tmpdir)
    eqns, stepper, v_true, c_true, b_true, c_init = run_sw_cond_evap(dirname, process)

    vapour = stepper.fields("water_vapour")
    cloud = stepper.fields("cloud_water")
    buoyancy = stepper.fields("b")

    # Check that the water vapour is correct
    assert norm(vapour - v_true) / norm(v_true) < 0.001, \
        f'Final vapour field is incorrect for {process}'

    # Check that cloud has been created/removed
    denom = norm(c_true) if process == "condensation" else norm(c_init)
    assert norm(cloud - c_true) / denom < 0.001, \
        f'Final cloud field is incorrect for {process}'

    # Check that the buoyancy perturbation has the correct sign and size
    assert norm(buoyancy - b_true) / norm(b_true) < 0.01, \
        f'Latent heating is incorrect for {process}'

    # Check that total moisture conserved
    filename = path.join(dirname, "sw_cond_evap/diagnostics.nc")
    data = Dataset(filename, "r")
    water = data.groups["water_vapour_plus_cloud_water"]
    total = water.variables["total"]
    water_t_0 = total[0]
    water_t_T = total[-1]

    assert abs(water_t_0 - water_t_T) / water_t_0 < 1e-12, \
        f'Total amount of water should be conserved by {process}'
