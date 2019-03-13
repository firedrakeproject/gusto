from os import path
from gusto import *
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
    tmax = setup.tmax
    L = 10.
    x = SpatialCoordinate(state.mesh)

    u = state.fields("u", space=state.spaces("HDiv"), dump=True)
    rho = state.fields("rho", space=state.spaces("DG"))
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

    # set up water_v
    xc = 5.
    zc = 3.5
    rc = 2.5
    r = sqrt((x[0]-xc)**2 + (x[1]-zc)**2)
    w_expr = conditional(r > rc, 0., 0.01*(1. + cos((pi/rc)*r)))

    water_v.interpolate(w_expr)

    rho_eqn = ContinuityEquation(state, Vrho, "rho")
    theta_eqn = AdvectionEquation(state, Vtheta, "theta")
    water_v_eqn = AdvectionEquation(state, Vtheta, "water_v")
    water_c_eqn = AdvectionEquation(state, Vtheta, "water_c")

    supg_opts = SUPGOptions()

    schemes = [SSPRK3(state, rho_eqn, advection),
               SSPRK3(state, theta_eqn, options=supg_opts),
               SSPRK3(state, water_v_eqn, options=supg_opts),
               SSPRK3(state, water_c_eqn, options=supg_opts)]

    # make a gradperp
    Vpsi = FunctionSpace(state.mesh, "CG", 2)
    gradperp = lambda u: as_vector([-u.dx(1), u.dx(0)])

    u_max = 0.1

    def u_evaluation(t):
        psi_expr = ((-u_max * L / pi)
                    * sin(2 * pi * x[0] / L)
                    * sin(pi * x[1] / L)) * sin(2 * pi * t / tmax)

        psi0 = Function(Vpsi).interpolate(psi_expr)

        return gradperp(psi0)

    u.project(u_evaluation(0))

    prescribed_fields = [('u', u_evaluation)]
    physics_list = [Condensation(state)]

    water_t_0 = assemble((water_c + water_v) * dx)

    timestepper = PrescribedAdvectionTimestepper(
        state, schemes,
        physics_list=physics_list, prescribed_fields=prescribed_fields)
    timestepper.run(t=0, tmax=tmax)

    # want to check that water is conserved
    water_t_T = assemble((water_c + water_v) * dx)

    # also want to check that some condensation has happened!
    one = Function(Vtheta).interpolate(Constant(1.0))
    cloud = assemble(water_c * dx) / assemble(one * dx)

    return abs(water_t_0 - water_t_T) / water_t_0, cloud


def test_condensation(tmpdir, tracer_setup):

    setup = tracer_setup(tmpdir, "slice", blob=True, atmosphere=True)
    err, cloud = run(setup)
    assert err < 1e-12
    assert cloud > 1e-12
