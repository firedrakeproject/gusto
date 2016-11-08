from gusto import *
from firedrake import Expression, FunctionSpace,\
    VectorFunctionSpace, PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate
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

# Space for initialising velocity
W_VectorCG1 = VectorFunctionSpace(mesh, "CG", 1)
W_CG1 = FunctionSpace(mesh, "CG", 1)

# vertical coordinate and normal
z = Function(W_CG1).interpolate(Expression("x[1]"))
k = Function(W_VectorCG1).interpolate(Expression(("0.","1.")))

fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=2.0, maxk=4, maxi=1)
output = OutputParameters(dirname='rb', dumpfreq=1, dumplist=['u'])
parameters = CompressibleParameters()
diagnostics = Diagnostics(*fieldlist)
diagnostic_fields = [CourantNumber()]

state = CompressibleState(mesh, vertical_degree=1, horizontal_degree=1,
                          family="CG",
                          z=z, k=k,
                          timestepping=timestepping,
                          output=output,
                          parameters=parameters,
                          diagnostics=diagnostics,
                          fieldlist=fieldlist,
                          diagnostic_fields=diagnostic_fields,
                          on_sphere=False)

# Initial conditions
u0, rho0, theta0 = Function(state.V[0]), Function(state.V[1]), Function(state.V[2])

# Isentropic background state
Tsurf = 300.
thetab = Constant(Tsurf)

theta_b = Function(state.V[2]).interpolate(thetab)
rho_b = Function(state.V[1])

# Calculate hydrostatic Pi
compressible_hydrostatic_balance(state, theta_b, rho_b, solve_for_rho=True)

x = SpatialCoordinate(mesh)
theta_pert = Function(state.V[2]).interpolate(Expression("sqrt(pow(x[0]-xc,2)+pow(x[1]-zc,2)) > rc ? 0.0 : 0.25*(1. + cos((pi/rc)*(sqrt(pow((x[0]-xc),2)+pow((x[1]-zc),2)))))", xc=500., zc=350., rc=250.))

theta0.interpolate(theta_b + theta_pert)
rho0.interpolate(rho_b)

state.initialise([u0, rho0, theta0])
state.set_reference_profiles(rho_b, theta_b)
state.output.meanfields = {'rho':state.rhobar, 'theta':state.thetabar}

# Set up advection schemes
Vtdg = FunctionSpace(mesh, "DG", 2)
advection_dict = {}
advection_dict["u"] = EulerPoincareForm(state, state.V[0])
advection_dict["rho"] = DGAdvection(state, state.V[1], continuity=True)
advection_dict["theta"] = SUPGAdvection(state, state.V[2], direction=[1])
# theta_advection = EmbeddedDGAdvection(state, state.V[2], Vdg=Vtdg, continuity=False)

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
