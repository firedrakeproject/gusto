"""
The moist rising bubble test from Bryan & Fritsch (2002), in a cloudy
atmosphere.

The rise of the thermal is fueled by latent heating from condensation.
"""

from gusto import *
from gusto import thermodynamics
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh,
                       SpatialCoordinate, conditional, cos, pi, sqrt,
                       NonlinearVariationalProblem,
                       NonlinearVariationalSolver, TestFunction, dx,
                       TrialFunction, Function,
                       LinearVariationalProblem, LinearVariationalSolver)
import sys

dt = 1.0
if '--running-tests' in sys.argv:
    deltax = 1000.
    tmax = 5.
else:
    deltax = 200
    tmax = 1000.

L = 10000.
H = 10000.
nlayers = int(H/deltax)
ncolumns = int(L/deltax)

m = PeriodicIntervalMesh(ncolumns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
degree = 1

dirname = 'moist_bryan_fritsch'

output = OutputParameters(dirname=dirname,
                          dumpfreq=int(tmax / (5*dt)),
                          dumplist=['u'],
                          perturbation_fields=[],
                          log_level='INFO')

params = CompressibleParameters()
diagnostic_fields = [Theta_e()]
tracers = [WaterVapour(), CloudWater()]

state = State(mesh,
              dt=dt,
              output=output,
              parameters=params,
              diagnostic_fields=diagnostic_fields)

eqns = CompressibleEulerEquations(state, "CG", degree, active_tracers=tracers)

# Initial conditions
u0 = state.fields("u")
rho0 = state.fields("rho")
theta0 = state.fields("theta")
water_v0 = state.fields("vapour_mixing_ratio")
water_c0 = state.fields("cloud_liquid_mixing_ratio")
moisture = ["vapour_mixing_ratio", "cloud_liquid_mixing_ratio"]

# spaces
Vu = state.spaces("HDiv")
Vt = state.spaces("theta")
Vr = state.spaces("DG")
x, z = SpatialCoordinate(mesh)
quadrature_degree = (4, 4)
dxp = dx(degree=(quadrature_degree))

# Define constant theta_e and water_t
Tsurf = 320.0
total_water = 0.02
theta_e = Function(Vt).assign(Tsurf)
water_t = Function(Vt).assign(total_water)

# Calculate hydrostatic fields
saturated_hydrostatic_balance(state, theta_e, water_t)

# make mean fields
theta_b = Function(Vt).assign(theta0)
rho_b = Function(Vr).assign(rho0)
water_vb = Function(Vt).assign(water_v0)
water_cb = Function(Vt).assign(water_t - water_vb)
exner_b = thermodynamics.exner_pressure(state.parameters, rho_b, theta_b)
Tb = thermodynamics.T(state.parameters, theta_b, exner_b, r_v=water_vb)

# define perturbation
xc = L / 2
zc = 2000.
rc = 2000.
Tdash = 2.0
r = sqrt((x - xc) ** 2 + (z - zc) ** 2)
theta_pert = Function(Vt).interpolate(
    conditional(r > rc,
                0.0,
                Tdash * (cos(pi * r / (2.0 * rc))) ** 2))

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
rho_averaged = Function(Vt)
rho_recoverer = Recoverer(rho0, rho_averaged)
rho_recoverer.project()

exner = thermodynamics.exner_pressure(state.parameters, rho_averaged, theta0)
p = thermodynamics.p(state.parameters, exner)
T = thermodynamics.T(state.parameters, theta0, exner, r_v=w_v)
w_sat = thermodynamics.r_sat(state.parameters, T, p)

w_functional = (phi * w_v * dxp - phi * w_sat * dxp)
w_problem = NonlinearVariationalProblem(w_functional, w_v)
w_solver = NonlinearVariationalSolver(w_problem)
w_solver.solve()

water_v0.assign(w_v)
water_c0.assign(water_t - water_v0)

state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b),
                              ('vapour_mixing_ratio', water_vb)])

rho_opts = None
theta_opts = EmbeddedDGOptions()
u_transport = ImplicitMidpoint(state, "u")

transported_fields = [SSPRK3(state, "rho", options=rho_opts),
                      SSPRK3(state, "theta", options=theta_opts),
                      SSPRK3(state, "vapour_mixing_ratio", options=theta_opts),
                      SSPRK3(state, "cloud_liquid_mixing_ratio", options=theta_opts),
                      u_transport]

# Set up linear solver
linear_solver = CompressibleSolver(state, eqns, moisture=moisture)

# define condensation
physics_list = [Condensation(state)]

# build time stepper
stepper = CrankNicolson(state, eqns, transported_fields,
                        linear_solver=linear_solver,
                        physics_list=physics_list)

stepper.run(t=0, tmax=tmax)
