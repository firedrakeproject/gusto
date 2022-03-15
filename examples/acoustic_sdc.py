from firedrake import (PeriodicIntervalMesh, FunctionSpace, MixedFunctionSpace,
                       TestFunctions, Function, dx, Constant, split,
                       SpatialCoordinate, NonlinearVariationalProblem,
                       NonlinearVariationalSolver, File, exp, cos)
from gusto import State, PrognosticEquation, OutputParameters, IMEX_Euler, Timestepper
from gusto.fml.form_manipulation_labelling import Label, drop
from gusto.labels import time_derivative, subject, replace_subject, fast, slow
import numpy as np

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
        fast_form = fast(subject(c_s * (w * p.dx(0) + phi * u.dx(0)) * dx, X))
        slow_form = slow(subject(u_mean * (w * u.dx(0) + phi * p.dx(0)) * dx, X))
        self.residual = mass_form + fast_form + slow_form


class IMEX_SDC(object):

    def __init__(self, M, dt, equation):
        self.M = M

        # set up SDC form and solver
        W = equation.function_space
        U_SDC = Function(W)
        U01 = Function(W)
        Q = Function(W)
        u11, p11 = split(U_SDC)
        F = equation.residual.label_map(lambda t: t.has_label(time_derivative),
                                        map_if_false=lambda t: dt*t)
        F_imp = F.label_map(lambda t: any(t.has_label(time_derivative, fast)),
                            replace_subject(U_SDC),
                            drop)

        F_exp = F.label_map(lambda t: any(t.has_label(time_derivative, slow)),
                            replace_subject(un.split()),
                            drop)
        F_exp = F_exp.label_map(lambda t: t.has_label(time_derivative),
                                lambda t: -1*t)

        F01 = F.label_map(lambda t: t.has_label(fast),
                          replace_subject(U01.split()),
                          drop)

        F01 = F01.label_map(all_terms, lambda t: -1*t)
        
        F0 = F.label_map(lambda t: t.has_label(slow),
                          replace_subject(U0.split()),
                          drop)
        F0 = F0.label_map(all_terms, lambda t: -1*t)
        Q = F.label_map(lambda t: t.has_label(time_derivative),
                        replace_subject(Q),
                        drop)

        F_SDC = F_imp + F_exp + F01 + F0 + Q
        prob_SDC = NonlinearVariationalProblem(F_SDC.form, U_SDC)
        self.solver_SDC = NonlinearVariationalSolver(prob_SDC)

        # set up RHS evaluation
        Urhs = Function(W)
        Uin = Function(W)
        a = equation.residual.label_map(lambda t: t.has_label(time_derivative),
                                        replace_subject(Urhs),
                                        drop)
        L = equation.residual.label_map(lambda t: t.has_label(time_derivative),
                                        drop,
                                        replace_subject(Uin.split()))
        Frhs = a - L
        prob_rhs = NonlinearVariationalProblem(Frhs.form, Urhs)
        self.solver_rhs = NonlinearVariationalSolver(prob_rhs)

    def rnw_r(n, a, b, A=-1, B=1):
        # nodes and weights for gauss - radau IIA quadrature
        # See Abramowitz & Stegun p 888
        nodes = np.zeros(n)
        nodes[0] = A
        p = np.poly1d([1, 1])
        pn = legendre(n)
        pn1 = legendre(n-1)
        poly, remainder = (pn + pn1)/p  # [1] returns remainder from polynomial division
        nodes[1:] = np.sort(poly.roots)
        weights = 1/n**2 * (1-nodes[1:])/(pn1(nodes[1:]))**2
        weights = np.append(2/n**2, weights)
        nodes = ((b - a) * nodes + a * B - b * A) / (B - A)
        weights = (b - a)/(B - A)*weights
        self.nodes = ((b + a) - nodes)[::-1]  # reverse nodes
        self.weights = weights[::-1]  # reverse weights


    def NewtonVM(t):
        """
        t: array or list containing nodes.
        returns: array Newton Vandermode Matrix. Entries are in the lower
        triangle
        Polynomial can be created with
        scipy.linalg.solve_triangular(NewtonVM(t),y,lower=True) where y
        contains the points the polynomial need to pass through
        """
        t = np.asarray(t)
        dim = len(t)
        VM = np.zeros([dim, dim])
        VM[:, 0] = 1
        for i in range(1, dim):
            VM[:, i] = (t[:] - t[(i - 1)]) * VM[:, i - 1]

        return VM


    def Horner_newton(weights, xi, x):
        """
        Horner scheme to evaluate polynomials based on newton basis
        """
        y = np.zeros_like(x)
        for i in range(len(weights)):
            y = y * (x - xi[(-i - 1)]) + weights[(-i - 1)]

        return y


    def gauss_legendre(n, a, b, A=-1, B=1):
        # nodes and weights for gauss legendre quadrature
        from scipy.special import legendre
        poly = legendre(n)
        polyd = poly.deriv()
        nodes= poly.roots
        nodes = np.sort(nodes)
        weights = 2/((1-nodes**2)*(np.polyval(polyd,nodes))**2)
        self.gl_nodes = ((b - a) * nodes + a * B - b * A) / (B - A)
        self.gl_weights=(b-a)/(B-A)*weights


    def get_weights(n, a, b, nodes):
        # This calculates for equation 2.4 FWSW - called from Q
        # integrates lagrange polynomials to the points [nodes]
        nodes_m, weights_m=nodes_weights(np.ceil(n/2), a, b)  # use gauss-legendre quadrature to integrate polynomials
        weights = np.zeros(n)
        for j in np.arange(n):
            coeff = np.zeros(n)
            coeff[j] = 1.0  # is unity because it needs to be scaled with y_j for interpolation we have  sum y_j*l_j
            poly_coeffs = scipy.linalg.solve_triangular(NewtonVM(nodes), coeff, lower=True)
            eval_newt_poly = self.Horner_newton(poly_coeffs, nodes, self.gl_nodes)
            weights[j] = np.dot(self.gl_weights, eval_newt_poly)
        return weights


    def Qmatrix(nodes, a):
        """
        Integration Matrix 
        """
        M = len(nodes)
        Q = np.zeros([M, M])

        # for all nodes, get weights for the interval [tleft,node]
        for m in np.arange(M):
            w = get_weights(M, a, nodes[m],nodes)
            Q[m, 0:] = w

        return Q


    def Smatrix(Q):
        """
        Integration matrix based on Q: sum(S@vector) returns integration
        """
        from copy import deepcopy
        M = len(Q)
        S = np.zeros([M, M])

        S[0, :] = deepcopy(Q[0, :])
        for m in np.arange(1, M):
            S[m, :] = Q[m, :] - Q[m - 1, :]

        return S


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

scheme = IMEX_Euler(state)
timestepper = Timestepper(state, ((eqn, scheme),))
timestepper.run(a, b)


# To Do:
# DONE1. Make IMEX_Euler class based on above
# DONE2. Check that Timestepper class will do IMEX_Euler
# 3. Instantiate SDC class with all the right parts
# DONE4. Create form for SDC solve
# 5. Make SDC class and try with Timestepper class
