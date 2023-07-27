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

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

dt = 1.0
L = 10000.
H = 10000.

if '--running-tests' in sys.argv:
    deltax = 1000.
    tmax = 5.
else:
    deltax = 200
    tmax = 1000.

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain
nlayers = int(H/deltax)
ncolumns = int(L/deltax)

m = PeriodicIntervalMesh(ncolumns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
degree = 1
domain = Domain(mesh, dt, 'CG', degree)

# Equation
params = CompressibleParameters()
tracers = [WaterVapour(), CloudWater()]
eqns = CompressibleEulerEquations(domain, params, active_tracers=tracers)

# I/O
dirname = 'moist_bryan_fritsch'
output = OutputParameters(dirname=dirname,
                          dumpfreq=int(tmax / (5*dt)),
                          dumplist=['u'],
                          log_level='INFO')
diagnostic_fields = [Theta_e(eqns)]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
transported_fields = [SSPRK3(domain, "rho"),
                      SSPRK3(domain, "theta", options=EmbeddedDGOptions()),
                      SSPRK3(domain, "water_vapour", options=EmbeddedDGOptions()),
                      SSPRK3(domain, "cloud_water", options=EmbeddedDGOptions()),
                      TrapeziumRule(domain, "u")]

transport_methods = [DGUpwind(eqns, field) for field in ["u", "rho", "theta", "water_vapour", "cloud_water"]]

# Linear solver
linear_solver = CompressibleSolver(eqns)

# Physics schemes (condensation/evaporation)
physics_schemes = [(SaturationAdjustment(eqns), ForwardEuler(domain))]

# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                  transport_methods,
                                  linear_solver=linear_solver,
                                  physics_schemes=physics_schemes)

# ---------------------------------------------------------------------------- #
# Initial conditions
# ---------------------------------------------------------------------------- #

u0 = stepper.fields("u")
rho0 = stepper.fields("rho")
theta0 = stepper.fields("theta")
water_v0 = stepper.fields("water_vapour")
water_c0 = stepper.fields("cloud_water")

# spaces
Vu = domain.spaces("HDiv")
Vt = domain.spaces("theta")
Vr = domain.spaces("DG")
x, z = SpatialCoordinate(mesh)
quadrature_degree = (4, 4)
dxp = dx(degree=(quadrature_degree))

# Define constant theta_e and water_t
Tsurf = 320.0
total_water = 0.02
theta_e = Function(Vt).assign(Tsurf)
water_t = Function(Vt).assign(total_water)

# Calculate hydrostatic fields
saturated_hydrostatic_balance(eqns, stepper.fields, theta_e, water_t)

# make mean fields
theta_b = Function(Vt).assign(theta0)
rho_b = Function(Vr).assign(rho0)
water_vb = Function(Vt).assign(water_v0)
water_cb = Function(Vt).assign(water_t - water_vb)
exner_b = thermodynamics.exner_pressure(eqns.parameters, rho_b, theta_b)
Tb = thermodynamics.T(eqns.parameters, theta_b, exner_b, r_v=water_vb)

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
theta0.interpolate(theta_b * (theta_pert / 300.0 + 1.0))

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

exner = thermodynamics.exner_pressure(eqns.parameters, rho_averaged, theta0)
p = thermodynamics.p(eqns.parameters, exner)
T = thermodynamics.T(eqns.parameters, theta0, exner, r_v=w_v)
w_sat = thermodynamics.r_sat(eqns.parameters, T, p)

w_functional = (phi * w_v * dxp - phi * w_sat * dxp)
w_problem = NonlinearVariationalProblem(w_functional, w_v)
w_solver = NonlinearVariationalSolver(w_problem)
w_solver.solve()

water_v0.assign(w_v)
water_c0.assign(water_t - water_v0)

stepper.set_reference_profiles([('rho', rho_b),
                                ('theta', theta_b),
                                ('water_vapour', water_vb),
                                ('cloud_water', water_cb)])

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)
