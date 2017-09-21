from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, \
    SpatialCoordinate, conditional, cos, pi, sqrt, exp, NonlinearVariationalProblem, \
    NonlinearVariationalSolver, TestFunction, dx, TrialFunction, Constant, Function, \
    LinearVariationalProblem, LinearVariationalSolver, DirichletBC
import sys

dt = 1.0
if '--running-tests' in sys.argv:
    tmax = 10.
    deltax = 1000.
else:
    deltax = 100.
    tmax = 1000.

L = 20000.
H = 10000.
nlayers = int(H/deltax)
ncolumns = int(L/deltax)

m = PeriodicIntervalMesh(ncolumns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
diffusion = True

fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt, maxk=4, maxi=1)
output = OutputParameters(dirname='moist_bf', dumpfreq=20, dumplist=['u'], perturbation_fields=[])
params = CompressibleParameters()
diagnostics = Diagnostics(*fieldlist)
diagnostic_fields = [Theta_e()]

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
water_v0 = state.fields("water_v", theta0.function_space())
water_c0 = state.fields("water_c", theta0.function_space())
moisture = ["water_v", "water_c"]

# spaces
Vu = u0.function_space()
Vt = theta0.function_space()
Vr = rho0.function_space()
x = SpatialCoordinate(mesh)
quadrature_degree = (5, 5)
dxp = dx(degree=(quadrature_degree))

# declare some parameters
p_0 = params.p_0
R_d = params.R_d
R_v = params.R_v
cp = params.cp
c_pl = params.c_pl
c_pv = params.c_pv
L_v0 = params.L_v0
kappa = params.kappa
w_sat1 = params.w_sat1
w_sat2 = params.w_sat2
w_sat3 = params.w_sat3
w_sat4 = params.w_sat4
T_0 = params.T_0
g = params.g
cp = params.cp

# Define constant theta_e and water_t
Tsurf = 320.0
total_water = 0.02
theta_e = Function(Vt).assign(Tsurf)
water_t = Function(Vt).assign(total_water)

# Calculate hydrostatic fields
moist_hydrostatic_balance(state, theta_e, water_t)

# make mean fields
theta_b = Function(Vt).assign(theta0)
rho_b = Function(Vr).assign(rho0)
water_vb = Function(Vt).assign(water_v0)
water_cb = Function(Vt).assign(water_t - water_vb)

# define perturbation
xc = 10000.
zc = 2000.
rc = 2000.
Tdash = 2.0
theta_pert = Function(Vt).interpolate(conditional(sqrt((x[0] - xc) ** 2 + (x[1] - zc) ** 2) > rc,
                                                  0.0, Tdash *
                                                  (cos(pi * sqrt(((x[0] - xc) / rc) ** 2 + ((x[1] - zc) / rc) ** 2) / 2.0))
                                                  ** 2))

# define initial theta
theta0.assign(theta_b * (theta_pert / 300.0 + 1.0))

# find perturbed rho
gamma = TestFunction(Vr)
rho_trial = TrialFunction(Vr)
a = gamma * rho_trial * dxp
L = gamma * (rho_b * theta_b / theta0) * dxp
rho_problem = LinearVariationalProblem(a, L, rho0)
rho_solver = LinearVariationalSolver(rho_problem)
rho_solver.solve()

# find perturbed water_v
w_v = Function(Vt)
phi = TestFunction(Vt)

p = p_0 * (R_d * theta0 * rho0 / p_0) ** (1.0 / (1.0 - kappa))
T = theta0 * (R_d * theta0 * rho0 / p_0) ** (kappa / (1.0 - kappa)) / (1.0 + w_v * R_v / R_d)
w_sat = w_sat1 / (p * exp(w_sat2 * ((T - T_0) / (T - w_sat3))) - w_sat4)

w_functional = (phi * w_v * dxp - phi * w_sat * dxp)
w_problem = NonlinearVariationalProblem(w_functional, w_v)
w_solver = NonlinearVariationalSolver(w_problem)
w_solver.solve()

water_v0.assign(w_v)
water_c0.assign(water_t - water_v0)

# initialise fields
state.initialise([('u', u0),
                  ('rho', rho0),
                  ('theta', theta0),
                  ('water_v', water_v0),
                  ('water_c', water_c0)])
state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b),
                              ('water_v', water_vb)])

# Set up advection schemes
ueqn = EulerPoincare(state, Vu)
rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
thetaeqn = EmbeddedDGAdvection(state, Vt, equation_form="advective")

advected_fields = [('u', ThetaMethod(state, u0, ueqn)),
                   ('rho', SSPRK3(state, rho0, rhoeqn)),
                   ('theta', SSPRK3(state, theta0, thetaeqn)),
                   ('water_v', SSPRK3(state, water_v0, thetaeqn)),
                   ('water_c', SSPRK3(state, water_c0, thetaeqn))]

linear_solver = CompressibleSolver(state, moisture=moisture)

# Set up forcing
compressible_forcing = CompressibleForcing(state, moisture=moisture)

# diffusion
bcs = [DirichletBC(Vu, 0.0, "bottom"),
       DirichletBC(Vu, 0.0, "top")]

diffused_fields = []

if diffusion:
    diffused_fields.append(('u', InteriorPenalty(state, Vu, kappa=Constant(60.),
                                                 mu=Constant(10./deltax), bcs=bcs)))

# define condensation
physics_list = [Condensation(state)]

# build time stepper
stepper = Timestepper(state, advected_fields, linear_solver,
                      compressible_forcing, physics_list=physics_list,
                      diffused_fields=diffused_fields)

stepper.run(t=0, tmax=tmax)
