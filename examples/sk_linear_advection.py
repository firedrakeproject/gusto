from dcore import *
from firedrake import Mesh, Expression, \
    VectorFunctionSpace, PeriodicIntervalMesh, ExtrudedMesh
from firedrake import exp, acos, cos, sin, ds_b
import numpy as np

nlayers = 10 #10 horizontal layers
refinements = 3 # number of horizontal cells = 2**refinements
L = 3.0e5
m = PeriodicIntervalMesh(2**refinements, L)

#build volume mesh
H = 1.0e4  # Height position of the model top
mesh = ExtrudedMesh(m, layers = nlayers, layer_height = H/nlayers)

# Space for initialising velocity
W_VectorCG1 = VectorFunctionSpace(mesh, "CG", 1)
W_CG1 = FunctionSpace(mesh, "CG", 1)

Omega = None

#vertical coordinate and normal
z = Function(W_CG1).interpolate(Expression("x[1]"))
k = Function(W_VectorCG1).interpolate(
    Expression(("0.","1.")))

#Thermodynamic constants
g = 9.810616
N = 0.01  # Brunt-Vaisala frequency (1/s)
p_0 = 1000.0 * 100.0  # Reference pressure (Pa, not hPa)
c_p = 1004.5  # SHC of dry air at constant pressure (J/kg/K)
R_d = 287.0  # Gas constant for dry air (J/kg/K)
kappa = 2.0/7.0  # R_d/c_p

state = State(mesh,vertical_degree = 0, horizontal_degree = 0,
              family = "CG",
              dt = 10.0,
              alpha = 0.5,
              g = g,
              cp = c_p,
              R_d = R_d,
              p_0 = p_0,
              z=z,
              k=k,
              Omega=Omega,
              Verbose=True, dumpfreq=1)

# Initial conditions
u0, theta0, rho0 = Function(state.V[0]), Function(state.V[2]), Function(state.V[1])

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = 300.
thetab = Tsurf*exp(N**2*g*z)

theta_b = Function(state.V[2]).interpolate(thetab)
rho_b = Function(state.V[1])

#Calculate hydrostatic Pi
W = MixedFunctionSpace((state.Vv,state.V[1]))
v, pi = TrialFunctions(W)
dv, dpi = TestFunctions(W)

n = FacetNormal(mesh)

alhs = (
    (c_p*inner(v,dv) - c_p*div(dv*theta_b)*pi)*dx
    + dpi*div(v)*dx
)

arhs = (
    -g*inner(dv,k)*dx
    -c_p*inner(dv,n)*theta_b*ds_b # bottom surface value pi = 1.
)
bcs = [DirichletBC(W.sub(0), Expression(("0.", "0.")), "top")]

w = Function(W)
PiProblem = LinearVariationalProblem(alhs, arhs, w, bcs=bcs, nest = False)

params={'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'schur',
        'ksp_type': 'gmres',
        'ksp_max_it': 100,
        'ksp_gmres_restart': 50,
        'pc_fieldsplit_schur_fact_type': 'FULL',
        'pc_fieldsplit_schur_precondition': 'selfp',
        'fieldsplit_0_ksp_type': 'preonly',
        'fieldsplit_0_pc_type': 'bjacobi',
        'fieldsplit_0_sub_pc_type': 'ilu',
        'fieldsplit_1_ksp_type': 'preonly',
        'fieldsplit_1_pc_type': 'gamg',
        'fieldsplit_1_mg_levels_ksp_type': 'chebyshev',
        'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues': True,
        'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues_random': True,
        'fieldsplit_1_mg_levels_ksp_max_it': 1,
        'fieldsplit_1_mg_levels_pc_type': 'bjacobi',
        'fieldsplit_1_mg_levels_sub_pc_type': 'ilu'}

PiSolver = LinearVariationalSolver(PiProblem,
                                   solver_parameters = params)

PiSolver.solve()

v, Pi = w.split()

x = Function(W_CG1).interpolate(Expression("x[0]"))
a = 5.0e3
deltaTheta = 1.0e-2
theta_pert = deltaTheta*sin(2*np.pi*z/H)/(1 + (x - L/2))
theta0.interpolate(theta_b + theta_pert)
rho0.assign(rho_b)

state.initialise(u0, rho0, theta0)
state.set_reference_profiles(rho_b, theta_b)

#Set up advection schemes
advection_list = []
velocity_advection = NoAdvection(state)
advection_list.append((velocity_advection, 0))
rho_advection = LinearAdvection_V3(state, state.V[1], rho_b)
advection_list.append((rho_advection, 1))
theta_advection = LinearAdvection_Vt(state, state.V[2], theta_b)
advection_list.append((theta_advection, 2))

#Set up linear solver
schur_params={'pc_type': 'fieldsplit',
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
        'fieldsplit_1_mg_levels_sub_pc_type': 'ilu'
}

linear_solver = CompressibleSolver(state, alpha = 0.5)

#Set up forcing
compressible_forcing = CompressibleForcing(state)

#build time stepper
stepper = Timestepper(state, advection_list, linear_solver,
                      compressible_forcing)

stepper.run(t = 0, tmax = 3600.0)

