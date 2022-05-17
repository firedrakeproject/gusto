from os import path
from gusto import *
from firedrake import (as_vector, Constant, sin, PeriodicIntervalMesh,
                       SpatialCoordinate, ExtrudedMesh, FunctionSpace,
                       Function, sqrt, conditional, cos)
from netCDF4 import Dataset
from math import pi

# This setup creates a bubble of water vapour that is advected
# by a prescribed velocity. The test passes if the integral
# of the water mixing ratio is conserved.


def setup_condens(dirname):

    # declare grid shape, with length L and height H
    L = 1000.
    H = 1000.
    nlayers = int(H / 100.)
    ncolumns = int(L / 100.)

    tmax = 10.0

    # make mesh
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=(H / nlayers))
    x = SpatialCoordinate(mesh)

    dt = 1.0
    output = OutputParameters(dirname=dirname+"/condens",
                              dumpfreq=1,
                              dumplist=['u'])
    parameters = CompressibleParameters()

    state = State(mesh,
                  dt=dt,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=[Sum('vapour_mixing_ratio', 'cloud_liquid_mixing_ratio')])

    # spaces
    Vpsi = FunctionSpace(mesh, "CG", 2)
    Vt = state.spaces("theta", degree=1)
    Vr = state.spaces("DG", "DG", degree=1)

    # set up equations
    rhoeqn = ContinuityEquation(state, Vr, "rho", ufamily="CG", udegree=1)
    thetaeqn = AdvectionEquation(state, Vt, "theta")
    wveqn = AdvectionEquation(state, Vt, "vapour_mixing_ratio")
    wceqn = AdvectionEquation(state, Vt, "cloud_liquid_mixing_ratio")

    # declare initial fields
    u0 = state.fields("u")
    rho0 = state.fields("rho")
    theta0 = state.fields("theta")
    water_v0 = state.fields("vapour_mixing_ratio")

    # make a gradperp
    gradperp = lambda u: as_vector([-u.dx(1), u.dx(0)])

    # Isentropic background state
    Tsurf = Constant(300.)

    theta_b = Function(Vt).interpolate(Tsurf)
    rho_b = Function(Vr)

    # Calculate initial rho
    compressible_hydrostatic_balance(state, theta_b, rho_b,
                                     solve_for_rho=True)

    # set up water_v
    xc = 500.
    zc = 350.
    rc = 250.
    r = sqrt((x[0]-xc)**2 + (x[1]-zc)**2)
    w_expr = conditional(r > rc, 0., 0.25*(1. + cos((pi/rc)*r)))

    # set up velocity field
    u_max = 20.0

    def u_evaluation(t):
        psi_expr = ((-u_max * L / pi)
                    * sin(2 * pi * x[0] / L)
                    * sin(pi * x[1] / L)) * sin(2 * pi * t / tmax)

        psi0 = Function(Vpsi).interpolate(psi_expr)

        return gradperp(psi0)

    u0.project(u_evaluation(0))
    theta0.interpolate(theta_b)
    rho0.interpolate(rho_b)
    water_v0.interpolate(w_expr)

    # build probem
    problem = []
    problem.append((rhoeqn, SSPRK3(state)))
    problem.append((thetaeqn, SSPRK3(state, options=SUPGOptions())))
    problem.append((wveqn, SSPRK3(state)))
    problem.append((wceqn, SSPRK3(state)))

    physics_list = [Condensation(state)]

    # build time stepper
    stepper = PrescribedAdvection(state, problem,
                                  physics_list=physics_list,
                                  prescribed_advecting_velocity=u_evaluation)

    return stepper, tmax


def run_condens(dirname):

    stepper, tmax = setup_condens(dirname)
    stepper.run(t=0, tmax=tmax)


def test_condens_setup(tmpdir):

    dirname = str(tmpdir)
    run_condens(dirname)
    filename = path.join(dirname, "condens/diagnostics.nc")
    data = Dataset(filename, "r")

    water = data.groups["vapour_mixing_ratio_plus_cloud_liquid_mixing_ratio"]
    total = water.variables["total"]
    water_t_0 = total[0]
    water_t_T = total[-1]

    assert abs(water_t_0 - water_t_T) / water_t_0 < 1e-12
