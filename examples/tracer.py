from gusto import *
from firedrake import Expression, PeriodicIntervalMesh, ExtrudedMesh, \
    SpatialCoordinate, DirichletBC
import sys

if '--running-tests' in sys.argv:
    res_dt = {800.:4.}
    tmax = 4.
else:
    res_dt = {800.:4.,400.:2.,200.:1.,100.:0.5,50.:0.25}
    tmax = 15.*60.

L = 51200.

# build volume mesh
H = 6400.  # Height position of the model top

for delta, dt in res_dt.iteritems():

    dirname = "tracer_dx%s_dt%s" % (delta, dt)
    nlayers = int(H/delta)  # horizontal layers
    columns = int(L/delta)  # number of columns

    m = PeriodicIntervalMesh(columns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

    fieldlist = ['u', 'rho', 'theta']
    timestepping = TimesteppingParameters(dt=dt, maxk=4, maxi=1)
    output = OutputParameters(dirname=dirname, dumpfreq=5, dumplist=['u'], perturbation_fields=['theta', 'rho'])
    parameters = CompressibleParameters()
    diagnostics = Diagnostics(*fieldlist)
    diagnostic_fields = [CourantNumber()]

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
    water0 = state.fields("water", theta0.function_space())

    # spaces
    Vu = u0.function_space()
    Vt = theta0.function_space()
    Vr = rho0.function_space()

    # Isentropic background state
    Tsurf = 300.
    thetab = Constant(Tsurf)

    theta_b = Function(Vt).interpolate(thetab)
    rho_b = Function(Vr)

    # Calculate hydrostatic Pi
    compressible_hydrostatic_balance(state, theta_b, rho_b, solve_for_rho=True)

    x = SpatialCoordinate(mesh)
    a = 5.0e3
    deltaTheta = 1.0e-2
    theta_pert = Function(theta0.function_space()).interpolate(Expression("sqrt(pow((x[0]-xc)/xr,2)+pow((x[1]-zc)/zr,2)) > 1. ? 0.0 : -7.5*(cos(pi*(sqrt(pow((x[0]-xc)/xr,2)+pow((x[1]-zc)/zr,2))))+1)", xc=0.5*L, xr=4000., zc=3000., zr=2000., g=parameters.g))
    theta0.interpolate(theta_b + theta_pert)
    water0.interpolate(theta_pert)
    rho0.assign(rho_b)

    state.initialise({'u':u0, 'rho':rho0, 'theta': theta0, 'water': water0})
    state.set_reference_profiles({'rho':rho_b, 'theta':theta_b})

    # Set up advection schemes
    ueqn = EulerPoincare(state, Vu)
    rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
    supg = True
    if supg:
        thetaeqn = SUPGAdvection(state, Vt,
                                 supg_params={"dg_direction":"horizontal"},
                                 equation_form="advective")
        watereqn = SUPGAdvection(state, Vt,
                                 supg_params={"dg_direction":"horizontal"},
                                 equation_form="advective")
    else:
        thetaeqn = EmbeddedDGAdvection(state, Vt,
                                       equation_form="advective")
        watereqn = EmbeddedDGAdvection(state, Vt,
                                       equation_form="advective")
    advection_dict = {}
    advection_dict["u"] = ThetaMethod(state, u0, ueqn)
    advection_dict["rho"] = SSPRK3(state, rho0, rhoeqn)
    advection_dict["theta"] = SSPRK3(state, theta0, thetaeqn)
    advection_dict["water"] = SSPRK3(state, water0, watereqn)

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
                    "fieldsplit_1_ksp_monitor_true_residual": True,
                    'fieldsplit_1_pc_type': 'gamg',
                    'fieldsplit_1_pc_gamg_sym_graph': True,
                    'fieldsplit_1_mg_levels_ksp_type': 'chebyshev',
                    'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues': True,
                    'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues_random': True,
                    'fieldsplit_1_mg_levels_ksp_max_it': 5,
                    'fieldsplit_1_mg_levels_pc_type': 'bjacobi',
                    'fieldsplit_1_mg_levels_sub_pc_type': 'ilu'}

    linear_solver = CompressibleSolver(state, params=schur_params)

    # Set up forcing
    compressible_forcing = CompressibleForcing(state)

    bcs = [DirichletBC(Vu, 0.0, "bottom"),
           DirichletBC(Vu, 0.0, "top")]
    diffusion_dict = {"u": InteriorPenalty(state, Vu, kappa=Constant(75.), mu=Constant(10./delta), bcs=bcs),
                      "theta": InteriorPenalty(state, Vt, kappa=Constant(75.), mu=Constant(10./delta))}

    # build time stepper
    stepper = Timestepper(state, advection_dict, linear_solver,
                          compressible_forcing, diffusion_dict)

    stepper.run(t=0, tmax=tmax)
