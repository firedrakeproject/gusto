from gusto import *
from firedrake import Expression, FunctionSpace, as_vector,\
    VectorFunctionSpace, PeriodicIntervalMesh, ExtrudedMesh, Constant, SpatialCoordinate, NonlinearVariationalProblem, NonlinearVariationalSolver, exp, ds_t

nlayers = 70  # horizontal layers
columns = 180  # number of columns
L = 144000.
m = PeriodicIntervalMesh(columns, L)

# build volume mesh
H = 35000.  # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
Vc = mesh.coordinates.function_space()
x = SpatialCoordinate(mesh)
H = Constant(H)
a = Constant(1000.)
xc = Constant(L/2.)
new_coords = Function(Vc).interpolate(as_vector([x[0], x[1]+(H-x[1])*a**2/(H*((x[0]-xc)**2+a**2))]))
mesh.coordinates.assign(new_coords)

# Space for initialising velocity
W_VectorCG1 = VectorFunctionSpace(mesh, "CG", 1)
W_CG1 = FunctionSpace(mesh, "CG", 1)
W_DG1 = FunctionSpace(mesh, "DG", 1)

# vertical coordinate and normal
z = Function(W_CG1).interpolate(Expression("x[1]"))
k = Function(W_VectorCG1).interpolate(Expression(("0.","1.")))

dt = 5.0
mu_top = Expression("x[1] <= zc ? 0.0 : mubar*pow(sin((pi/2.)*(x[1]-zc)/(H-zc)),2)", H=H, zc=(H-10000.), mubar=0.15/dt)
mu = Function(W_DG1).interpolate(mu_top)
fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt)
output = OutputParameters(dirname='nh_mountain', dumpfreq=1, dumplist=['u'])
parameters = CompressibleParameters(g=9.80665, cp=1004., mu=mu)
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

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = 300.
thetab = Tsurf*exp(N**2*z/g)
theta_b = Function(state.V[2]).interpolate(thetab)
# theta_top = Tsurf*exp(N**2*H/g)
# pi_top = 1. - (g**2/(c_p*N**2))*(theta_top - Tsurf)/(theta_top*Tsurf)

# Calculate hydrostatic Pi
W = MixedFunctionSpace((state.Vv,state.V[1]))
v, pi = TrialFunctions(W)
dv, dpi = TestFunctions(W)

n = FacetNormal(mesh)

alhs = (
    (c_p*inner(v,dv) - c_p*div(dv*theta_b)*pi)*dx
    + dpi*div(theta_b*v)*dx
)
pi_top = 0.5
arhs = (- g*inner(dv,k)*dx - pi_top*c_p*inner(dv,n)*theta_b*ds_t)
bcs = [DirichletBC(W.sub(0), Expression(("0.", "0.")), "bottom")]
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
pfile = File('pi.pvd')
PiSolver.solve()
v, Pi = w.split()
pfile.write(Pi)
p0 = Pi.dat.data[0]
print "JEMMA: ", p0
pi_top = 1.0
arhs = (- g*inner(dv,k)*dx - pi_top*c_p*inner(dv,n)*theta_b*ds_t)
PiProblem = LinearVariationalProblem(alhs, arhs, w, bcs=bcs)
PiSolver = LinearVariationalSolver(PiProblem,
                                   solver_parameters=params)
PiSolver.solve()
v, Pi = w.split()
pfile.write(Pi)
p1 = Pi.dat.data[0]
print "JEMMA: ", p1
alpha = 2.*(p1-p0)
beta = p1-alpha
pi_top = (1.-beta)/alpha
print "JEMMA: ", alpha, beta, pi_top
arhs = (- g*inner(dv,k)*dx - pi_top*c_p*inner(dv,n)*theta_b*ds_t)
PiProblem = LinearVariationalProblem(alhs, arhs, w, bcs=bcs)
PiSolver = LinearVariationalSolver(PiProblem,
                                   solver_parameters=params)
PiSolver.solve()
v, Pi = w.split()
print "JEMMA: ", Pi.dat.data[0]
pfile.write(Pi)

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
    + pi_top*c_p*inner(dv,n)*theta_b*ds_t
)
rhoproblem = NonlinearVariationalProblem(F, w1, bcs=bcs)
rhosolver = NonlinearVariationalSolver(rhoproblem, solver_parameters=params)
rhosolver.solve()
v, rho = w1.split()
rho_b = Function(state.V[1]).interpolate(rho)

theta0.interpolate(theta_b)
rho0.assign(rho_b)
u0.project(as_vector([10.0,0.0]))

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

stepper.run(t=0, tmax=9000.0)
