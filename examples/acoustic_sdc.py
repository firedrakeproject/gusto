from firedrake import (PeriodicIntervalMesh, FunctionSpace, MixedFunctionSpace,
                       TestFunctions, Function, dx, Constant, split,
                       SpatialCoordinate, NonlinearVariationalProblem,
                       NonlinearVariationalSolver, File, exp, cos, assemble)
from gusto import State, PrognosticEquation, OutputParameters, IMEX_Euler, Timestepper, IMEX_SDC
from gusto.fml.form_manipulation_labelling import Label, drop, all_terms
from gusto.labels import time_derivative, subject, replace_subject, implicit, explicit
import numpy as np
import scipy
from scipy.special import legendre


class AcousticEquation(PrognosticEquation):

    field_names = ["u", "p"]

    def __init__(self, state):

        Vu = FunctionSpace(mesh, 'Lagrange', 1)
        Vp = FunctionSpace(mesh, 'CG', 1)
        W = MixedFunctionSpace((Vu, Vp))

        super().__init__(state, W, "u_p")

        w, phi = TestFunctions(W)
        X = Function(W)
        u, p = split(X)

        c_s = Constant(1)                   # speed of sound
        u_mean = Constant(0.05)                  # mean flow

        mass_form = time_derivative(subject(w * u * dx + phi * p * dx, X))
        implicit_form = implicit(subject(c_s * (w * p.dx(0) + phi * u.dx(0)) * dx, X))
        explicit_form = implicit(subject(u_mean * (w * u.dx(0) + phi * p.dx(0)) * dx, X))
        self.residual = mass_form + implicit_form + explicit_form



a = 0                           # time of start
b = 3                           # time of end
n = 512                         # number of spatial nodes
n_steps = 154.                  # number of time steps
dt = (b-a)/(n_steps)            # delta t

mesh = PeriodicIntervalMesh(n, 1)

output = OutputParameters(dirname="acoustic")

state = State(mesh, dt=dt, output=output)

eqn = AcousticEquation(state)

p0 = state.fields("p")

x1 = 0.25
x0 = 0.75
sigma = 0.1

def p_0(x, sigma=sigma):
    return exp(-x**2/sigma**2)
                 
def p_1(x, p0=p_0, sigma=sigma, k=7.2*np.pi):
    return p0(x)*cos(k*x/sigma)

def p_init(x, p0=p_0, p1=p_1, x0=x0, x1=x1, coeff=1.):
    return p_0(x-x0) + coeff*p_1(x-x1)

x = SpatialCoordinate(mesh)[0]
p0.interpolate(p_init(x))

M = 3
maxk = 2
scheme = IMEX_SDC(state, M, maxk)
timestepper = Timestepper(state, ((eqn, scheme),))
timestepper.run(a, b)
