from petsc4py import PETSc
PETSc.Sys.popErrorHandler()
from gusto import *
import itertools
from firedrake import (as_vector, SpatialCoordinate, PeriodicIntervalMesh,
                       ExtrudedMesh, exp, sin, Function)
import numpy as np
import sys

dt = 0.1
tmax = 800.

nlayers = 10  # horizontal layers
columns = 150  # number of columns
L = 3.0e5
m = PeriodicIntervalMesh(columns, L)

# build volume mesh
H = 1.0e4  # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

dirname = 'sk_sdc'

output = OutputParameters(dirname=dirname,
                          dumplist=['u'],
                          perturbation_fields=['theta', 'rho'])

parameters = CompressibleParameters()
g = parameters.g
Tsurf = 300.

state = State(mesh,
              dt=dt,
              output=output,
              parameters=parameters)

eqns = CompressibleEulerEquations(state, "CG", 1)

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

state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b)])

# build time stepper
# scheme = SSPRK3(state)
M = 3
maxk = 2
scheme = FE_SDC(state, M, maxk)
stepper = Timestepper(state, ((eqns, scheme),))

stepper.run(t=0, tmax=tmax)
