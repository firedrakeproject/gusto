from gusto import *
from firedrake import as_vector, Constant, sin, cos, \
    PeriodicIntervalMesh, ExtrudedMesh, Expression, \
    SpatialCoordinate
import json
from math import pi

def setup_condens(dirname):

    # declare grid shape, with length L and height H
    L = 1000.
    H = 1000.
    nlayers = int(H / 100.)
    ncolumns = int(L / 100.)

    # make mesh
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=(H / nlayers))

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
                  diagnostic_fields=[Sum('water_v','water_c')])

    # declare initial fields
    u0 = state.fields("u")
    rho0 = state.fields("rho")
    theta0 = state.fields("theta")

    # spaces
    Vu = u0.function_space()
    Vt = theta0.function_space()
    Vr = rho0.function_space()

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
    u_max = Constant(1.0) 
    u_expr = as_vector([u_max * sin(2 * pi * x[0] / L) *
                        cos(pi * x[1] / H),
                        - 2 * u_max * cos(2 * pi * x[0] / L) *
                        sin(pi * x[1] / H)])
    
    u0.project(u_expr)
    theta0.interpolate(theta_b)
    rho0.interpolate(rho_b)
    water_v0.interpolate(theta0)

    state.initialise({'u': u0, 'rho': rho0, 'theta': theta0,
                      'water_v': water_v0, 'water_c':water_c0})
    state.set_reference_profiles({'rho': rho_b, 'theta': theta_b})

    # set up advection schemes
    rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
    thetaeqn = SUPGAdvection(state, Vt,
                             supg_params={"dg_direction":"horizontal"},
                             equation_form="advective")

    # build advection dictionary
    advection_dict = {}
    advection_dict["u"] = NoAdvection(state, u0, None)
    advection_dict["rho"] = SSPRK3(state, rho0, rhoeqn)
    advection_dict["theta"] = SSPRK3(state, theta0, thetaeqn)
    advection_dict["tracer"] = SSPRK3(state, tracer0, thetaeqn)

    # Set up linear solver
    schur_params = {'pc_type': 'fieldsplit',
                    'pc_fieldsplit_type': 'schur',
                    'ksp_type': 'gmres',
                    'ksp_monitor_true_residual': True,
                    'ksp_max_it': 100,
                    'ksp_gmres_restart': 50,
                    'pc_fieldsplit_schur_fact_type': 'FULL',
                    'pc_fieldsplit_schur_precondition': 'selfp',
                    'fieldsplit_0_ksp_type': 'richardson',
                    'fieldsplit_0_ksp_max_it': 5,
                    'fieldsplit_0_pc_type': 'bjacobi',
                    'fieldsplit_0_sub_pc_type': 'ilu',
                    'fieldsplit_1_ksp_type': 'richardson',
                    'fieldsplit_1_ksp_max_it': 5,
                    'fieldsplit_1_ksp_monitor_true_residual': True,
                    'fieldsplit_1_pc_type': 'gamg',
                    'fieldsplit_1_pc_gamg_sym_graph': True,
                    'fieldsplit_1_mg_levels_ksp_type': 'chebyshev',
                    'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues': True,
                    'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues_random': True,
                    'fieldsplit_1_mg_levels_ksp_max_it': 5,
                    'fieldsplit_1_mg_levels_pc_type': 'bjacobi',
                    'fieldsplit_1_mg_levels_sub_pc_type': 'ilu'}

    linear_solver = CompressibleSolver(state, params=schur_params)

    compressible_forcing = NoForcing(state)

    physics_list = [Condensation(state)]

    # build time stepper
    stepper = Timestepper(state, advection_dict, linear_solver,
                          compressible_forcing, physics_list=physics_list)

    return stepper, 100.0


def run_condens(dirname):

    stepper, tmax = setup_condens(dirname)
    stepper.run(t=0, tmax=tmax)


def test_condens_setup(tmpdir):

    dirname = str(tmpdir)
    run_condens(dirname)
    with open(path.join(dirname, "condens/diagnostics.json"), "r") as f:
        data = json.load(f)
    print data.keys()

    water_t_l20 = data["water_v_plus_water_c"]["l2"][0]
    water_t_l2T = data["water_v_plus_water_c"]["l2"][-1]

    assert water_t_l20 == water_t_l2T
