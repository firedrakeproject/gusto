"""
The non-linear gravity wave test case of Skamarock and Klemp (1994), but solved
with the hydrostatic Compressible Euler equations.

Potential temperature is transported using SUPG.
"""

from gusto import *
from firedrake import (as_vector, SpatialCoordinate, PeriodicRectangleMesh,
                       ExtrudedMesh, exp, sin, Function)
import numpy as np
import sys

dt = 25.
if '--running-tests' in sys.argv:
    nlayers = 5  # horizontal layers
    columns = 50  # number of columns
    tmax = dt
    dumpfreq = 1
else:
    nlayers = 10  # horizontal layers
    columns = 150  # number of columns
    tmax = 60000.0
    dumpfreq = int(tmax / (2*dt))


L = 6.0e6
m = PeriodicRectangleMesh(columns, 1, L, 1.e4, quadrilateral=True)

# build volume mesh
H = 1.0e4  # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

dirname = 'skamarock_klemp_hydrostatic'

output = OutputParameters(dirname=dirname,
                          dumpfreq=dumpfreq,
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

Omega = as_vector((0., 0., 0.5e-4))
balanced_pg = as_vector((0., -1.0e-4*20, 0.))
eqns = CompressibleEulerEquations(state, "RTCF", 1, Omega=Omega,
                                  extra_terms=[("u", balanced_pg)])

# Initial conditions
u0 = state.fields("u")
rho0 = state.fields("rho")
theta0 = state.fields("theta")

# spaces
Vu = state.spaces("HDiv")
Vt = state.spaces("theta")
Vr = state.spaces("DG")

# Thermodynamic constants required for setting initial conditions
# and reference profiles
g = parameters.g
N = parameters.N
p_0 = parameters.p_0
c_p = parameters.cp
R_d = parameters.R_d
kappa = parameters.kappa

x, y, z = SpatialCoordinate(mesh)

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = 300.
thetab = Tsurf*exp(N**2*z/g)

theta_b = Function(Vt).interpolate(thetab)
rho_b = Function(Vr)

a = 1.0e5
deltaTheta = 1.0e-2
theta_pert = deltaTheta*sin(np.pi*z/H)/(1 + (x - L/2)**2/a**2)
theta0.interpolate(theta_b + theta_pert)

compressible_hydrostatic_balance(state, theta_b, rho_b,
                                 solve_for_rho=True)

rho0.assign(rho_b)
u0.project(as_vector([20.0, 0.0, 0.0]))

state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b)])

# Set up transport schemes
transported_fields = []
transported_fields.append(ImplicitMidpoint(state, "u"))
transported_fields.append(SSPRK3(state, "rho"))
transported_fields.append(SSPRK3(state, "theta", options=SUPGOptions()))

# Set up linear solver
linear_solver = CompressibleSolver(state, eqns)

# build time stepper
stepper = CrankNicolson(state, eqns, transported_fields,
                        linear_solver=linear_solver)

stepper.run(t=0, tmax=tmax)
