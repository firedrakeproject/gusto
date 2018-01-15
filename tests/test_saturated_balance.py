from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, Constant, Function
from firedrake import NonlinearVariationalProblem, NonlinearVariationalSolver, TestFunction, dx, FunctionSpace, MixedFunctionSpace, TestFunctions, split
from os import path
from netCDF4 import Dataset


def setup_saturated(dirname):

    solve_for_rho = True
    solve_again_for_rho = True
    solve_again_for_water_v = False
    weak_condensation = True
    condensation_off = False

    # set up grid and time stepping parameters
    dt = 1.
    tmax = 3.
    deltax = 200
    L = 2000.
    H = 10000.

    nlayers = int(H/deltax)
    ncolumns = int(L/deltax)

    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

    fieldlist = ['u', 'rho', 'theta']
    timestepping = TimesteppingParameters(dt=dt, maxk=4, maxi=1)
    output = OutputParameters(dirname=dirname+'/saturated_balance', dumpfreq=1, dumplist=['u','rho','theta'], perturbation_fields=['rho','theta','water_v'])
    parameters = CompressibleParameters()
    diagnostics = Diagnostics(*fieldlist)
    diagnostic_fields = [Theta_e(), Perturbation("Theta_e")]

    state = State(mesh, vertical_degree=1, horizontal_degree=1,
                  family="CG",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  diagnostics=diagnostics,
                  fieldlist=fieldlist,
                  diagnostic_fields=diagnostic_fields)

    # Initial conditions
    u0 = state.fields("u")
    rho0 = state.fields("rho")
    theta0 = state.fields("theta")
    water_v0 = state.fields("water_v", theta0.function_space())
    water_c0 = state.fields("water_c", theta0.function_space())
    moisture = ['water_v', 'water_c']

    # spaces
    Vu = u0.function_space()
    Vt = theta0.function_space()
    Vr = rho0.function_space()

    # Isentropic background state
    Tsurf = Constant(300.)
    total_water = Constant(0.02)
    theta_e = Function(Vt).interpolate(Tsurf)
    water_t = Function(Vt).interpolate(total_water)
    theta_e_bar = state.fields("Theta_ebar", FunctionSpace(mesh, "CG", 1)).interpolate(Tsurf)

    # Calculate hydrostatic Pi
    saturated_hydrostatic_balance(state, theta_e, water_t, solve_for_rho=solve_for_rho, solve_for_rho_again=solve_again_for_rho)

    if solve_again_for_water_v:
        # solve again for water_v
        quadrature_degree = (4, 4)
        dxp = dx(degree=(quadrature_degree))
        w_v = Function(Vt)
        phi = TestFunction(Vt)
        pi = pi_expr(state.parameters, rho0, theta0)
        p = p_expr(state.parameters, pi)
        T = T_expr(state.parameters, theta0, pi, r_v=w_v)
        w_sat = r_sat_expr(state.parameters, T, p)
        w_functional = (phi * w_v * dxp - phi * w_sat * dxp)
        w_problem = NonlinearVariationalProblem(w_functional, w_v)
        w_solver = NonlinearVariationalSolver(w_problem)
        w_solver.solve()
        water_v0.assign(w_v)

    water_c0.assign(water_t - water_v0)

    state.initialise([('u', u0),
                      ('rho', rho0),
                      ('theta', theta0),
                      ('water_v', water_v0),
                      ('water_c', water_c0)])
    state.set_reference_profiles([('rho', rho0),
                                  ('theta', theta0),
                                  ('water_v', water_v0)])

    # Set up advection schemes
    ueqn = EulerPoincare(state, Vu)
    rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
    thetaeqn = EmbeddedDGAdvection(state, Vt, equation_form="advective")
    
    advected_fields = [("u", ThetaMethod(state, u0, ueqn)),
                       ("rho", SSPRK3(state, rho0, rhoeqn)),
                       ("theta", SSPRK3(state, theta0, thetaeqn)),
                       ("water_v", SSPRK3(state, water_v0, thetaeqn)),
                       ("water_c", SSPRK3(state, water_c0, thetaeqn))]

    # Set up linear solver
    linear_solver = CompressibleSolver(state, moisture=moisture)

    # Set up forcing
    compressible_forcing = CompressibleForcing(state, moisture=moisture)

    # Set up physics
    if condensation_off:
        physics_list = []
    else:
        physics_list = [Condensation(state, weak=weak_condensation)]

    # build time stepper
    stepper = CrankNicolson(state, advected_fields, linear_solver,
                            compressible_forcing, physics_list=physics_list)

    return stepper, tmax


def run_saturated(dirname):

    stepper, tmax = setup_saturated(dirname)
    stepper.run(t=0, tmax=tmax)


def test_saturated_setup(tmpdir):

    dirname = str(tmpdir)
    run_saturated(dirname)
    filename = path.join(dirname, "saturated_balance/diagnostics.nc")
    data = Dataset(filename, "r")

    u = data.groups['u']
    umax = u.variables['max']
    theta_e = data.groups['Theta_e_perturbation']
    theta_e_max = theta_e.variables['max']
    theta_e_min = theta_e.variables['min']
    water_v = data.groups['water_v_perturbation']
    w_v_max = water_v.variables['max']
    w_v_min = water_v.variables['min']

    max_theta = max(abs(theta_e_max[-1]), abs(theta_e_min[-1]))
    max_w_v = max(abs(w_v_max[-1]), abs(w_v_min[-1]))

    print(umax[-1], max_theta, max_w_v)

    assert umax[-1] < 1e-12
