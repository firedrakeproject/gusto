from os import path
from gusto import *
from firedrake import (Constant, SpatialCoordinate, FunctionSpace,
                       Function, conditional, cos, assemble, norm,
                       min_value)
from netCDF4 import Dataset
from math import pi

# Like test_condensation_alone.py, this set up tests the condensation procedure.
# A blob of cloud is initialised, with no water vapour.
# The excess cloud should evaporate, absorbing latent heat.
# There is no advection.

# This tests that the evaporation happens, consevering mass of water,
# not releasing heat, and not forming negative water concentrations.

def run(setup):

    state = setup.state
    tmax = setup.tmax
    Ld = setup.Ld
    x = SpatialCoordinate(state.mesh)

    u = state.fields("u", space=state.spaces("HDiv"), dump=False)
    rho = state.fields("rho", space=state.spaces("DG"), dump=False)
    theta = state.fields("theta", space=state.spaces("HDiv_v"), dump=True)
    water_v = state.fields("water_v", space=state.spaces("HDiv_v"), dump=True)
    water_c = state.fields("water_c", space=state.spaces("HDiv_v"), dump=True)

    # Isentropic background state
    Vrho = rho.function_space()
    Vtheta = theta.function_space()
    Tsurf = Constant(280.)
    rhosurf = Constant(1.0)

    theta.interpolate(Tsurf)
    rho.interpolate(rhosurf)

    # set up blob  of cloud
    xc = 0.5 * Ld
    zc = 0.35 * Ld
    rc = 0.25 * Ld
    r = sqrt((x[0]-xc)**2 + (x[1]-zc)**2)
    w_expr = conditional(r > rc, 0., 0.01*(1. + cos((pi/rc)*r)))

    water_c.interpolate(w_expr)

    schemes = []
    physics_list = [Condensation(state)]

    water_t_initial = assemble((water_c + water_v) * dx)
    total_theta_initial = assemble(theta * dx)

    timestepper = PrescribedAdvectionTimestepper(
        state, schemes,
        physics_list=physics_list)
    timestepper.run(t=0, tmax=tmax)

    # want to check that water is conserved
    water_t_final = assemble((water_c + water_v) * dx)

    # also want to check that some condensation has happened!
    one = Function(Vtheta).interpolate(Constant(1.0))
    total_water_v = assemble(water_v * dx) / assemble(one * dx)
    total_theta_final = assemble(theta * dx)
    negative_water_v = Function(Vtheta).interpolate(min_value(0, water_v))
    negative_cloud = Function(Vtheta).interpolate(min_value(0, water_c))

    return (abs(water_t_initial - water_t_final) / water_t_initial,
            total_water_v,
            (total_theta_final - total_theta_initial) / total_theta_initial,
            norm(negative_water_v),
            norm(negative_cloud))


def test_evaporation(tmpdir, moist_setup):

    setup = moist_setup(tmpdir, "normal")
    water_change, water_vapour, theta_change, negative_water_v, negative_cloud = run(setup)
    # check that the total amount of water hasn't changed
    assert water_change < 1e-12
    # check that a cloud has indeed formed
    assert water_vapour > 1e-12
    # check that heat has been absorbed
    assert theta_change < -1e-12
    # check that there is no negative water vapour
    assert negative_water_v < 1e-12
    # check that there is no negative cloud
    assert negative_cloud < 1e-12
