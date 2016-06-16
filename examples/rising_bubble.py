from dcore import *
from firedrake import Expression, FunctionSpace,\
    VectorFunctionSpace, PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate
from firedrake import ds_b, NonlinearVariationalProblem, NonlinearVariationalSolver

L = 1000.
H = 1000.
nlayers = int(H/10.)
ncolumns = int(L/10.)
dt = 1.

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
                          diagnostic_fields=diagnostic_fields)

# Initial conditions
u0, theta0, rho0 = Function(state.V[0]), Function(state.V[2]), Function(state.V[1])


# Thermodynamic constants required for setting initial conditions
# and reference profiles
g = parameters.g
N = parameters.N
p_0 = parameters.p_0
c_p = parameters.cp
R_d = parameters.R_d
kappa = parameters.kappa

# Isentropic background state
Tsurf = 300.
thetab = Constant(Tsurf)

theta_b = Function(state.V[2]).interpolate(thetab)
rho_b = Function(state.V[1])

# Calculate hydrostatic Pi
W = MixedFunctionSpace((state.Vv,state.V[1]))
v, pi = TrialFunctions(W)
dv, dpi = TestFunctions(W)

n = FacetNormal(mesh)

alhs = (
    (c_p*inner(v,dv) - c_p*div(dv*theta_b)*pi)*dx
    + dpi*div(theta_b*v)*dx
)

arhs = (
    - g*inner(dv,k)*dx
    - c_p*inner(dv,n)*theta_b*ds_b  # bottom surface value pi = 1.
)
bcs = [DirichletBC(W.sub(0), Expression(("0.", "0.")), "top")]

w = Function(W)
PiProblem = LinearVariationalProblem(alhs, arhs, w, bcs=bcs)

params = {'pc_type': 'fieldsplit',
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
          'fieldsplit_1_pc_type': 'bjacobi',
          'fieldsplit_1_sub_pc_type': 'ilu'}

PiSolver = LinearVariationalSolver(PiProblem,
                                   solver_parameters=params)

PiSolver.solve()
v, Pi = w.split()

w1 = Function(W)
v, rho = w1.split()
rho.interpolate(p_0*(Pi**((1-kappa)/kappa))/R_d/theta_b)
v, rho = split(w1)
dv, dpi = TestFunctions(W)
pi = ((R_d/p_0)*rho*theta_b)**(kappa/(1.-kappa))
F = (
    (c_p*inner(v,dv) - c_p*div(dv*theta_b)*pi)*dx
    + dpi*div(theta_b*v)*dx
    + g*inner(dv,k)*dx
    + c_p*inner(dv,n)*theta_b*ds_b  # bottom surface value pi = 1.
)
rhoproblem = NonlinearVariationalProblem(F, w1, bcs=bcs)
rhosolver = NonlinearVariationalSolver(rhoproblem, solver_parameters=params)
rhosolver.solve()
v, rho = w1.split()
rho_b.interpolate(rho)

W_DG1 = FunctionSpace(mesh, "DG", 1)
x = SpatialCoordinate(mesh)
theta_pert = Function(state.V[2]).interpolate(Expression("sqrt(pow(x[0]-xc,2)+pow(x[1]-zc,2)) > rc ? 0.0 : 0.25*(1. + cos((pi/rc)*(sqrt(pow((x[0]-xc),2)+pow((x[1]-zc),2)))))", xc=500., zc=350., rc = 250.))

theta0.interpolate(theta_b + theta_pert)
rho0.interpolate(rho_b)

state.initialise([u0, rho0, theta0])
state.set_reference_profiles(rho_b, theta_b)
state.output.meanfields = {'rho':state.rhobar, 'theta':state.thetabar}

# Set up advection schemes
Vtdg = FunctionSpace(mesh, "DG", 2)
advection_list = []
velocity_advection = EulerPoincareForm(state, state.V[0])
advection_list.append((velocity_advection, 0))
rho_advection = DGAdvection(state, state.V[1], continuity=True)
advection_list.append((rho_advection, 1))
# theta_advection = EmbeddedDGAdvection(state, Vtdg, continuity=False)
theta_advection = SUPGAdvection(state, state.V[2], direction=[1])
advection_list.append((theta_advection, 2))

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
stepper = Timestepper(state, advection_list, linear_solver,
                      compressible_forcing)

stepper.run(t=0, tmax=700.)


