from gusto import *
from firedrake import Expression, \
    VectorFunctionSpace, PeriodicRectangleMesh, ExtrudedMesh, \
    exp, sin, SpatialCoordinate
import numpy as np

nlayers = 10  # horizontal layers
columns = 100  # number of columns
L = 6.0e6
m = PeriodicRectangleMesh(columns, 10, L, 1.e4, quadrilateral=True)

# build volume mesh
H = 1.0e4  # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

# Space for initialising velocity
W_VectorCG1 = VectorFunctionSpace(mesh, "CG", 1)
W_CG1 = FunctionSpace(mesh, "CG", 1)

# vertical coordinate and normal
z = Function(W_CG1).interpolate(Expression("x[2]"))
k = Function(W_VectorCG1).interpolate(Expression(("0.","0.","1.")))
Omega = Function(W_VectorCG1).interpolate(Expression(("0.","0.","1.e-4")))

fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=100.0)
output = OutputParameters(dirname='sk_linear_hydros', dumplist=['u','rho','theta'])
parameters = CompressibleParameters()

state = CompressibleState(mesh, vertical_degree=1, horizontal_degree=1,
                          family="RTCF",
                          z=z, k=k, Omega=Omega,
                          timestepping=timestepping,
                          output=output,
                          parameters=parameters,
                          fieldlist=fieldlist,
                          on_sphere=False)

# Initial conditions
u0, rho0, theta0 = Function(state.V[0]), Function(state.V[1]), Function(state.V[2])

# Thermodynamic constants required for setting initial conditions
# and reference profiles
g = parameters.g
N = parameters.N
p_0 = parameters.p_0
c_p = parameters.cp
R_d = parameters.R_d
kappa = parameters.kappa

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = 300.
thetab = Tsurf*exp(N**2*z/g)

theta_b = Function(state.V[2]).interpolate(thetab)
rho_b = Function(state.V[1])

# Calculate hydrostatic Pi
compressible_hydrostatic_balance(state, theta_b, rho_b)

W_DG1 = FunctionSpace(mesh, "DG", 1)
x = SpatialCoordinate(mesh)
a = 1.0e5
deltaTheta = 1.0e-2
theta_pert = deltaTheta*sin(np.pi*x[2]/H)/(1 + (x[0] - L/2)**2/a**2)
theta0.interpolate(theta_b + theta_pert)
rho0.assign(rho_b)

state.initialise([u0, rho0, theta0])
state.set_reference_profiles(rho_b, theta_b)
state.output.meanfields = {'rho':state.rhobar, 'theta':state.thetabar}

# Set up advection schemes
advection_dict = {}
advection_dict["u"] = NoAdvection(state)
advection_dict["rho"] = LinearAdvection_V3(state, state.V[1], rho_b)
advection_dict["theta"] = LinearAdvection_Vt(state, state.V[2], theta_b)

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
compressible_forcing = CompressibleForcing(state, linear=True)

# build time stepper
stepper = Timestepper(state, advection_dict, linear_solver,
                      compressible_forcing)

stepper.run(t=0, tmax=6000.0)
