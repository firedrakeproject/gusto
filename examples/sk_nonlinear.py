from gusto import *
import itertools
from firedrake import (as_vector, SpatialCoordinate, PeriodicIntervalMesh,
                       ExtrudedMesh, exp, sin, Function)
import numpy as np
import sys

dt = 6.
if '--running-tests' in sys.argv:
    tmax = dt
else:
    tmax = 3600.

if '--hybridization' in sys.argv:
    hybridization = True
else:
    hybridization = False

nlayers = 10  # horizontal layers
columns = 150  # number of columns
L = 3.0e5
m = PeriodicIntervalMesh(columns, L)

# build volume mesh
H = 1.0e4  # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

points_x = np.linspace(0., L, 100)
points_z = [H/2.]
points = np.array([p for p in itertools.product(points_x, points_z)])

dirname = 'sk_nonlinear'
if hybridization:
    dirname += '_hybridization'

output = OutputParameters(dirname=dirname,
                          dumpfreq=1,
                          dumplist=['u'],
                          perturbation_fields=['theta', 'rho'],
                          point_data=[('theta_perturbation', points)],
                          log_level='INFO')

parameters = CompressibleParameters()
g = parameters.g
Tsurf = 300.

diagnostic_fields = [CourantNumber(), Gradient("u"), Gradient("theta_perturbation"), RichardsonNumber("theta", g/Tsurf), Gradient("theta")]

state = State(mesh, dt=dt,
              output=output,
              parameters=parameters,
              diagnostic_fields=diagnostic_fields)

eqns = CompressibleEulerEquations(state, "CG", 1, 1)

# Initial conditions
u0 = state.fields("u")
rho0 = state.fields("rho")
theta0 = state.fields("theta")

# spaces
Vu = u0.function_space()
Vt = theta0.function_space()
Vr = rho0.function_space()

# Thermodynamic constants required for setting initial conditions
# and reference profiles
g = parameters.g
N = parameters.N
p_0 = parameters.p_0
c_p = parameters.cp
R_d = parameters.R_d
kappa = parameters.kappa

x, z = SpatialCoordinate(mesh)

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = 300.
thetab = Tsurf*exp(N**2*z/g)

theta_b = Function(Vt).interpolate(thetab)
rho_b = Function(Vr)

# Calculate hydrostatic Pi
compressible_hydrostatic_balance(state, theta_b, rho_b)

a = 5.0e3
deltaTheta = 1.0e-2
theta_pert = deltaTheta*sin(np.pi*z/H)/(1 + (x - L/2)**2/a**2)
theta0.interpolate(theta_b + theta_pert)
rho0.assign(rho_b)
u0.project(as_vector([20.0, 0.0]))

state.initialise([('u', u0),
                  ('rho', rho0),
                  ('theta', theta0)])
state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b)])

# Set up advection schemes
supg = True
if supg:
    theta_opts = SUPGOptions()
else:
    theta_opts = EmbeddedDGOptions()
advected_fields = []
advected_fields.append(ImplicitMidpoint(state, eqns, advection, field_name="u"))
advected_fields.append(SSPRK3(state, eqns, advection, field_name="rho"))
advected_fields.append(SSPRK3(state, eqns, advection, field_name="theta", options=theta_opts))

# build time stepper
stepper = CrankNicolson(state, equation_set=eqns, schemes=advected_fields)

stepper.run(t=0, tmax=tmax)
