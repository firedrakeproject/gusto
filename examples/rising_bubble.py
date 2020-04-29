from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate,
                       Constant, pi, cos, Function, sqrt, conditional)
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

dirname = 'rb'

output = OutputParameters(dirname=dirname,
                          dumpfreq=10,
                          dumplist=['u'],
                          perturbation_fields=['theta', 'rho'],
                          log_level='INFO')

parameters = CompressibleParameters()
diagnostic_fields = [CourantNumber()]

state = State(mesh,
              dt=dt,
              output=output,
              parameters=parameters,
              diagnostic_fields=diagnostic_fields)

eqns = CompressibleEulerEquations(state, "CG", 1)

# Initial conditions
u0 = state.fields("u")
rho0 = state.fields("rho")
theta0 = state.fields("theta")

# spaces
Vu = u0.function_space()
Vt = theta0.function_space()
Vr = rho0.function_space()

# Isentropic background state
Tsurf = Constant(300.)

theta_b = Function(Vt).interpolate(Tsurf)
rho_b = Function(Vr)

# Calculate hydrostatic Pi
compressible_hydrostatic_balance(state, theta_b, rho_b, solve_for_rho=True)

x = SpatialCoordinate(mesh)
xc = 500.
zc = 350.
rc = 250.
r = sqrt((x[0]-xc)**2 + (x[1]-zc)**2)
theta_pert = conditional(r > rc, 0., 0.25*(1. + cos((pi/rc)*r)))

theta0.interpolate(theta_b + theta_pert)
rho0.interpolate(rho_b)

state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b)])

# Set up advection schemes
supg = True
if supg:
    theta_opts = SUPGOptions()
else:
    theta_opts = EmbeddedDGOptions()
advected_fields = []
advected_fields.append(ImplicitMidpoint(state, "u"))
advected_fields.append(SSPRK3(state, "rho"))
advected_fields.append(SSPRK3(state, "theta", options=theta_opts))

# Set up linear solver
linear_solver = CompressibleSolver(state, eqns)

# build time stepper
stepper = CrankNicolson(state, eqns, advected_fields, linear_solver=linear_solver)

stepper.run(t=0, tmax=tmax)
