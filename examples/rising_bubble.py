from gusto import *
from firedrake import Expression, PeriodicIntervalMesh, ExtrudedMesh, \
    SpatialCoordinate
import sys

dt = 1.
if '--running-tests' in sys.argv:
    tmax = dt
else:
    tmax = 700.

L = 1000.
H = 1000.
nlayers = int(H/10.)
ncolumns = int(L/10.)

m = PeriodicIntervalMesh(ncolumns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt, maxk=4, maxi=1)
output = OutputParameters(dirname='rb', dumpfreq=10, dumplist=['u'], perturbation_fields=['theta', 'rho'])
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
theta_pert = Function(Vt).interpolate(Expression("sqrt(pow(x[0]-xc,2)+pow(x[1]-zc,2)) > rc ? 0.0 : 0.25*(1. + cos((pi/rc)*(sqrt(pow((x[0]-xc),2)+pow((x[1]-zc),2)))))", xc=500., zc=350., rc=250.))

theta0.interpolate(theta_b + theta_pert)
rho0.interpolate(rho_b)

state.initialise({'u': u0, 'rho': rho0, 'theta': theta0})
state.set_reference_profiles({'rho': rho_b, 'theta': theta_b})

# Set up advection schemes
ueqn = EulerPoincare(state, Vu)
rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
supg = True
if supg:
    thetaeqn = SUPGAdvection(state, Vt,
                             supg_params={"dg_direction": "horizontal"},
                             equation_form="advective")
else:
    thetaeqn = EmbeddedDGAdvection(state, Vt,
                                   equation_form="advective")
advection_dict = {}
advection_dict["u"] = ThetaMethod(state, u0, ueqn)
advection_dict["rho"] = SSPRK3(state, rho0, rhoeqn)
advection_dict["theta"] = SSPRK3(state, theta0, thetaeqn)

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

# build time stepper
stepper = Timestepper(state, advection_dict, linear_solver,
                      compressible_forcing)

stepper.run(t=0, tmax=tmax)
