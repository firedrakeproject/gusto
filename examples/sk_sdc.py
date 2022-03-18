from petsc4py import PETSc
PETSc.Sys.popErrorHandler()
from gusto import *
import itertools
from firedrake import (as_vector, SpatialCoordinate, PeriodicIntervalMesh,
                       ExtrudedMesh, exp, sin, Function)
import numpy as np
import sys

class MyCompressibleEulerEquations(PrognosticEquation):

    field_names = ['u', 'rho', 'theta']

    def __init__(self, state):

        spaces = state.spaces.build_compatible_spaces("CG", 1)
        W = MixedFunctionSpace(spaces)

        field_name = "_".join(self.field_names)
        super().__init__(state, W, field_name)

        Vu = W.sub(0)
        self.bcs['u'].append(DirichletBC(Vu, 0.0, "bottom"))
        self.bcs['u'].append(DirichletBC(Vu, 0.0, "top"))

        g = state.parameters.g
        cp = state.parameters.cp

        w, phi, gamma = TestFunctions(W)
        X = Function(W)
        u, rho, theta = X.split()
        rhobar = state.fields("rhobar", space=rho.function_space(), dump=False)
        thetabar = state.fields("thetabar", space=theta.function_space(), dump=False)
        pi = Pi(state.parameters, rho, theta)
        n = FacetNormal(state.mesh)

        u_mass = subject(prognostic(inner(u, w)*dx, "u"), X)

        rho_mass = subject(prognostic(inner(rho, phi)*dx, "rho"), X)

        theta_mass = subject(prognostic(inner(theta, gamma)*dx, "theta"), X)

        mass_form = time_derivative(u_mass + rho_mass + theta_mass)

        u_adv = prognostic(vector_invariant_form(state, w, u), "u")

        rho_adv = prognostic(continuity_form(state, phi, rho), "rho")

        theta_adv = prognostic(advection_form(state, gamma, theta, ibp=IntegrateByParts.TWICE), "theta")

        adv_form = subject(u_adv + rho_adv + theta_adv, X)

        pressure_gradient_form = name(subject(prognostic(
            cp*(-div(theta*w)*pi*dx
                + jump(theta*w, n)*avg(pi)*dS_v), "u"), X), "pressure_gradient")

        gravity_form = subject(prognostic(Term(g*inner(state.k, w)*dx), "u"), X)

        self.residual = (mass_form + adv_form
                         + pressure_gradient_form + gravity_form)


dt = 2.
if '--running-tests' in sys.argv:
    tmax = dt
else:
    tmax = 3600.


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
scheme = IMEX_SDC(state, M, maxk)
stepper = Timestepper(state, ((eqns, scheme),))

stepper.run(t=0, tmax=tmax)
