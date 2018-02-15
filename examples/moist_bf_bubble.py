from gusto import *
from gusto import thermodynamics
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, \
    SpatialCoordinate, conditional, cos, pi, sqrt, NonlinearVariationalProblem, \
    NonlinearVariationalSolver, TestFunction, dx, TrialFunction, Constant, Function, \
    LinearVariationalProblem, LinearVariationalSolver, DirichletBC, \
    FunctionSpace, BrokenElement, VectorFunctionSpace
import sys

dt = 1.0
if '--running-tests' in sys.argv:
    tmax = 10.
    deltax = 1000.
else:
    deltax = 100.
    tmax = 1000.

L = 10000.
H = 10000.
nlayers = int(H/deltax)
ncolumns = int(L/deltax)

m = PeriodicIntervalMesh(ncolumns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
diffusion = False
recovered = True
degree = 0 if recovered else 1

fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt, maxk=4, maxi=1)
output = OutputParameters(dirname='moist_bf_test', dumpfreq=20, dumplist=['u'], perturbation_fields=[], log_level='INFO')
params = CompressibleParameters()
diagnostics = Diagnostics(*fieldlist)
diagnostic_fields = [Theta_e(), InternalEnergy(), Perturbation("InternalEnergy")]

state = State(mesh, vertical_degree=degree, horizontal_degree=degree,
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

if recovered:
    VDG1 = FunctionSpace(mesh, "DG", 1)
    VCG1 = FunctionSpace(mesh, "CG", 1)
    Vt_brok = FunctionSpace(mesh, BrokenElement(Vt.ufl_element()))
    Vu_DG1 = VectorFunctionSpace(mesh, "DG", 1)
    Vu_CG1 = VectorFunctionSpace(mesh, "CG", 1)

    u_spaces = (Vu_DG1, Vu_CG1, Vu)
    rho_spaces = (VDG1, VCG1, Vr)
    theta_spaces = (VDG1, VCG1, Vt_brok)

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
pibar = thermodynamics.pi(state.parameters, rho_b, theta_b)
Tb = thermodynamics.T(state.parameters, theta_b, pibar, r_v=water_vb)
Ibar = state.fields("InternalEnergybar", FunctionSpace(mesh, "CG", 1))
Ibar.interpolate(thermodynamics.internal_energy(state.parameters, rho_b, Tb, r_v=water_vb, r_l=water_cb))

# define perturbation
xc = L / 2
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

pi = thermodynamics.pi(state.parameters, rho0, theta0)
p = thermodynamics.p(state.parameters, pi)
T = thermodynamics.T(state.parameters, theta0, pi, r_v=w_v)
w_sat = thermodynamics.r_sat(state.parameters, T, p)

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
if recovered:
    ueqn = EmbeddedDGAdvection(state, Vu, equation_form="advective", recovered_spaces=u_spaces)
    rhoeqn = EmbeddedDGAdvection(state, Vr, equation_form="continuity", recovered_spaces=rho_spaces)
    thetaeqn = EmbeddedDGAdvection(state, Vt, equation_form="advective", recovered_spaces=theta_spaces)
else:
    ueqn = EulerPoincare(state, Vu)
    rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
    thetaeqn = EmbeddedDGAdvection(state, Vt, equation_form="advective")

advected_fields = [('rho', SSPRK3(state, rho0, rhoeqn)),
                   ('theta', SSPRK3(state, theta0, thetaeqn)),
                   ('water_v', SSPRK3(state, water_v0, thetaeqn)),
                   ('water_c', SSPRK3(state, water_c0, thetaeqn))]
if recovered:
    advected_fields.append(('u', SSPRK3(state, u0, ueqn)))
else:
    advected_fields.append(('u', ThetaMethod(state, u0, ueqn)))

linear_solver = CompressibleSolver(state, moisture=moisture)

# Set up forcing
if recovered:
    compressible_forcing = CompressibleForcing(state, moisture=moisture, euler_poincare=False)
else:
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
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        compressible_forcing, physics_list=physics_list,
                        diffused_fields=diffused_fields)

stepper.run(t=0, tmax=tmax)
