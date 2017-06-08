from __future__ import absolute_import
from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, \
    SpatialCoordinate, conditional, cos, pi, sqrt, exp, NonlinearVariationalProblem, \
    NonlinearVariationalSolver, TestFunction, dx, grad, inner
from firedrake.slope_limiter import VertexBasedLimiter
import sys

dt = 1.0
if '--running-tests' in sys.argv:
    tmax = 4.
    deltax = 1000.
else:
    deltax = 20.
    tmax = 1080.

L = 1000.
H = 1500.
nlayers = int(H/deltax)
ncolumns = int(L/deltax)

m = PeriodicIntervalMesh(ncolumns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt, maxk=4, maxi=1)
output = OutputParameters(dirname='robert_gauss_lowres_SUPG', dumpfreq=5, dumplist=['u','theta','rho'], perturbation_fields=['theta', 'rho'])
params = CompressibleParameters()
diagnostics = Diagnostics(*fieldlist)
diagnostic_fields = []

state = State(mesh, vertical_degree=1, horizontal_degree=1,
              family="CG",
              timestepping=timestepping,
              output=output,
              parameters=params,
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
x = SpatialCoordinate(mesh)
quadrature_degree = (5,5)
dxp = dx(degree=(quadrature_degree))

# declare some parameters
p_0 = params.p_0
R_d = params.R_d
kappa = params.kappa
T_0 = params.T_0
g = params.g
cp = params.cp

# Define constant theta_e and water_t
Tsurf = 30.0 + T_0
theta0 = Function(Vt).interpolate(Constant(Tsurf))

# Calculate hydrostatic fields
compressible_hydrostatic_balance(state, theta0, rho0,
                                 solve_for_rho=True)

# make mean fields
theta_b = Function(Vt).assign(theta0)
rho_b = Function(Vr).assign(rho0)

# define perturbation
A = 0.5
a = 50.
s = 100.
xc = 500.
zc = 260.
r = sqrt((x[0] - xc) ** 2.0 + (x[1] - zc) ** 2.0)

theta_pert = Function(Vt).interpolate(conditional(r > a,
                                                  A * exp(-(r - a) ** 2.0 / (s ** 2.0)),
                                                  A))

# define initial theta
theta0.assign(theta_b + theta_pert)

# find perturbed rho
gamma = TestFunction(Vr)
rho_trial = Function(Vr)
rho_functional = gamma * theta0 * rho_trial * dxp - gamma * rho_b * theta_b * dxp
rho_problem = NonlinearVariationalProblem(rho_functional, rho_trial)
rho_solver = NonlinearVariationalSolver(rho_problem)
rho_solver.solve()
rho0.assign(rho_trial)

# initialise fields
state.initialise({'u': u0, 'rho': rho0, 'theta': theta0})

state.set_reference_profiles({'rho':rho_b, 'theta':theta_b})

# Set up advection schemes
ueqn = EulerPoincare(state, Vu)
rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
thetaeqn = SUPGAdvection(state, Vt,
                             supg_params={"dg_direction":"horizontal"},
                             equation_form="advective")

advection_dict = {}
advection_dict["u"] = ThetaMethod(state, u0, ueqn)
advection_dict["rho"] = SSPRK3(state, rho0, rhoeqn, limiter=None)
advection_dict["theta"] = SSPRK3(state, theta0, thetaeqn, limiter=None)

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
