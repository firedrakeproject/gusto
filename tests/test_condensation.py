from gusto import *
from firedrake import as_vector, Constant, sin, PeriodicIntervalMesh, \
    SpatialCoordinate, ExtrudedMesh, Expression
import json
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

    # make mesh
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=(H / nlayers))
    x = SpatialCoordinate(mesh)

    fieldlist = ['u', 'rho', 'theta']
    timestepping = TimesteppingParameters(dt=1.0, maxk=4, maxi=1)
    output = OutputParameters(dirname=dirname+"/condens",
                              dumpfreq=1,
                              dumplist=['u'],
                              perturbation_fields=['theta', 'rho'])
    parameters = CompressibleParameters()

    state = State(mesh, vertical_degree=1, horizontal_degree=1,
                  family="CG",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  fieldlist=fieldlist,
                  diagnostic_fields=[Sum('water_v', 'water_c')])

    # declare initial fields
    u0 = state.fields("u")
    rho0 = state.fields("rho")
    theta0 = state.fields("theta")

    # spaces
    Vpsi = FunctionSpace(mesh, "CG", 2)
    Vt = theta0.function_space()
    Vr = rho0.function_space()

    # make a gradperp
    gradperp = lambda u: as_vector([-u.dx(1), u.dx(0)])

    # declare tracer field and a background field
    water_v0 = state.fields("water_v", Vt)
    water_c0 = state.fields("water_c", Vt)

    # Isentropic background state
    Tsurf = 300.
    thetab = Constant(Tsurf)

    theta_b = Function(Vt).interpolate(thetab)
    rho_b = Function(Vr)

    # Calculate initial rho
    compressible_hydrostatic_balance(state, theta_b, rho_b,
                                     solve_for_rho=True)

    # set up water_v
    w_expr = Function(Vt).interpolate(
        Expression("sqrt(pow(x[0]-xc,2)+pow(x[1]-zc,2))" +
                   "> rc ? 0.0 : 0.25*(1. + cos((pi/rc)*" +
                   "(sqrt(pow((x[0]-xc),2)+pow((x[1]-zc),2)))))",
                   xc=500., zc=350., rc=250.))

    # set up velocity field
    u_max = Constant(10.0)

    psi_expr = ((-u_max * L / pi) *
                sin(2 * pi * x[0] / L) *
                sin(pi * x[1] / L))

    psi0 = Function(Vpsi).interpolate(psi_expr)
    u0.project(gradperp(psi0))
    theta0.interpolate(theta_b)
    rho0.interpolate(rho_b)
    water_v0.interpolate(w_expr)

    state.initialise({'u': u0, 'rho': rho0, 'theta': theta0,
                      'water_v': water_v0, 'water_c': water_c0})
    state.set_reference_profiles({'rho': rho_b, 'theta': theta_b})

    # set up advection schemes
    rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
    thetaeqn = SUPGAdvection(state, Vt,
                             supg_params={"dg_direction": "horizontal"},
                             equation_form="advective")

    # build advection dictionary
    advection_dict = {}
    advection_dict["u"] = NoAdvection(state, u0, None)
    advection_dict["rho"] = SSPRK3(state, rho0, rhoeqn)
    advection_dict["theta"] = SSPRK3(state, theta0, thetaeqn)
    advection_dict["water_v"] = SSPRK3(state, water_v0, thetaeqn)
    advection_dict["water_c"] = SSPRK3(state, water_c0, thetaeqn)

    physics_list = [Condensation(state)]

    # build time stepper
    stepper = AdvectionTimestepper(state, advection_dict, physics_list=physics_list)

    return stepper, 5.0


def run_condens(dirname):

    stepper, tmax = setup_condens(dirname)
    stepper.run(t=0, tmax=tmax)


def test_condens_setup(tmpdir):

    dirname = str(tmpdir)
    run_condens(dirname)
    with open(path.join(dirname, "condens/diagnostics.json"), "r") as f:
        data = json.load(f)
    print data.keys()

    water_t_0 = data["water_v_plus_water_c"]["total"][0]
    water_t_T = data["water_v_plus_water_c"]["total"][-1]

    assert abs(water_t_0 - water_t_T) / water_t_0 < 1e-12
