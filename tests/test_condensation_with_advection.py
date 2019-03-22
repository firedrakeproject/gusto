from os import path
from gusto import *
from gusto import thermodynamics
from firedrake import (as_vector, Constant, sin, PeriodicIntervalMesh,
                       SpatialCoordinate, ExtrudedMesh, FunctionSpace,
                       Function, sqrt, conditional, cos, assemble)
from netCDF4 import Dataset
from math import pi

# This setup creates a bubble of water vapour that is advected
# by a prescribed velocity. The test passes if the integral
# of the water mixing ratio is conserved.

def run(setup):

    state = setup.state
    tmax = 10 * setup.tmax
    Ld = setup.Ld
    x, z = SpatialCoordinate(state.mesh)

    u = state.fields("u", space=state.spaces("HDiv"), dump=True)
    rho = state.fields("rho", space=state.spaces("DG"))
    theta_vd = state.fields("theta", space=state.spaces("HDiv_v"), dump=True)
    water_v = state.fields("water_v", space=state.spaces("HDiv_v"), dump=True)
    water_c = state.fields("water_c", space=state.spaces("HDiv_v"), dump=True)

    # Isentropic background state
    Vrho = rho.function_space()
    Vtheta = theta_vd.function_space()
    Tmin = Constant(280.)
    Tmax = Constant(290.)
    rhosurf = Constant(1.0)

    parameters = state.parameters
    theta_d = Function(Vtheta).interpolate(Tmin + (Tmax - Tmin) * sin(2 * pi * x  / Ld))
    rho.interpolate(rhosurf)
    pie = thermodynamics.pi(parameters, rho, theta_d)
    p = thermodynamics.p(parameters, pie)
    T = thermodynamics.T(parameters, theta_d, pie)
    r_sat = thermodynamics.r_sat(parameters, T, p)

    # this sets up the water vapour to be r_sat = 0.2 everywhere
    # apart from a smooth cosine blob which reaches saturation
    # this should avoid formation of any negative concentrations by advection
    r_sat_bar = 0.2
    xc = 0.25 * Ld
    zc = 0.5 * Ld
    rc = 0.1 * Ld
    r = sqrt((x-xc)**2 + (z-zc)**2)
    w_expr = conditional(r > rc, r_sat_bar * r_sat,
                         r_sat_bar * r_sat + (1 - r_sat_bar) * r_sat * cos(pi * r / (2 * rc)))

    water_v.interpolate(w_expr)
    theta_vd.assign(theta_d * (1 + water_v * parameters.R_v / parameters.R_d))

    water_v_eqn = AdvectionEquation(state, Vtheta, "water_v")
    water_c_eqn = AdvectionEquation(state, Vtheta, "water_c")

    supg_opts = SUPGOptions()

    schemes = [SSPRK3(state, water_v_eqn, options=supg_opts),
               SSPRK3(state, water_c_eqn, options=supg_opts)]

    u.project(as_vector([0.4, 0.0]))

    physics_list = [Condensation(state)]

    water_t_initial = assemble((water_c + water_v) * dx)
    total_theta_initial = assemble(theta_vd * dx)

    timestepper = PrescribedAdvectionTimestepper(state, schemes, physics_list=physics_list)
    timestepper.run(t=0, tmax=tmax)

    # want to check that water is conserved
    water_t_final = assemble((water_c + water_v) * dx)

    # also want to check that some condensation has happened!
    one = Function(Vtheta).interpolate(Constant(1.0))
    total_cloud = assemble(water_c * dx) / assemble(one * dx)
    total_theta_final = assemble(theta_vd * dx)

    return (abs(water_t_initial - water_t_final) / water_t_initial,
            total_cloud,
            (total_theta_final - total_theta_initial) / total_theta_initial)


def test_advect_condensation(tmpdir, moist_setup):

    setup = moist_setup(tmpdir, "normal")
    water_change, cloud, theta_change = run(setup)
    assert water_change < 1e-12
    assert cloud > 1e-12
    assert theta_change > 1e-12
