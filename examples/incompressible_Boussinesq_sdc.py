from firedrake import (PeriodicIntervalMesh, FunctionSpace, MixedFunctionSpace,
                       TestFunctions, Function, dx, Constant, split,
                       SpatialCoordinate, NonlinearVariationalProblem,
                       NonlinearVariationalSolver, File, exp, cos, assemble,
                       ExtrudedMesh, DirichletBC)
from gusto import State, PrognosticEquation, OutputParameters, IMEX_Euler, Timestepper
from gusto.fml.form_manipulation_labelling import Label, drop, all_terms
from gusto.labels import time_derivative, subject, replace_subject, fast, slow
import numpy as np
import scipy
from scipy.special import legendre


class IncompressibleBoussinesqEquation(PrognosticEquation):

    field_names = ["u", "p", "b"]

    def __init__(self, state):

        spaces = state.spaces.build_compatible_spaces("CG", 1)
        W = MixedFunctionSpace(spaces)

        field_name = "_".join(self.field_names)
        super().__init__(state, W, field_name)

        Vu = state.spaces("HDiv")
        self.bcs['u'].append(DirichletBC(Vu, 0.0, "bottom"))
        self.bcs['u'].append(DirichletBC(Vu, 0.0, "top"))


class IMEX_SDC(object):

    def __init__(self, state, M, maxk):

        self.state = state
        self.dt = Constant(state.dt)
        self.M = M
        self.maxk = maxk
        self.IMEX = IMEX_Euler(state)

        self.rnw_r(state.dt)
        self.Qmatrix()
        self.Smatrix()
        self.dtau = np.diff(np.append(0, self.nodes))

    def setup(self, equation, uadv=None):

        self.IMEX.setup(equation, uadv)

        # set up SDC form and solver
        W = equation.function_space
        dt = self.dt
        self.W = W
        self.Unodes = [Function(W) for _ in range(M+1)]
        self.Unodes1 = [Function(W) for _ in range(M+1)]
        self.fUnodes = [Function(W) for _ in range(M+1)]

        self.U_SDC = Function(W)
        self.U0 = Function(W)
        self.U01 = Function(W)
        self.Un = Function(W)
        self.Q_ = Function(W)

        F = equation.residual.label_map(lambda t: t.has_label(time_derivative),
                                        map_if_false=lambda t: dt*t)
        F_imp = F.label_map(lambda t: any(t.has_label(time_derivative, fast)),
                            replace_subject(self.U_SDC),
                            drop)

        F_exp = F.label_map(lambda t: any(t.has_label(time_derivative, slow)),
                            replace_subject(self.Un.split()),
                            drop)
        F_exp = F_exp.label_map(lambda t: t.has_label(time_derivative),
                                lambda t: -1*t)

        F01 = F.label_map(lambda t: t.has_label(fast),
                          replace_subject(self.U01.split()),
                          drop)

        F01 = F01.label_map(all_terms, lambda t: -1*t)
        
        F0 = F.label_map(lambda t: t.has_label(slow),
                          replace_subject(self.U0.split()),
                          drop)
        F0 = F0.label_map(all_terms, lambda t: -1*t)
        Q = F.label_map(lambda t: t.has_label(time_derivative),
                        replace_subject(self.Q_),
                        drop)

        F_SDC = F_imp + F_exp + F01 + F0 + Q
        prob_SDC = NonlinearVariationalProblem(F_SDC.form, self.U_SDC)
        self.solver_SDC = NonlinearVariationalSolver(prob_SDC)

        # set up RHS evaluation
        self.Urhs = Function(W)
        self.Uin = Function(W)
        a = equation.residual.label_map(lambda t: t.has_label(time_derivative),
                                        replace_subject(self.Urhs),
                                        drop)
        L = equation.residual.label_map(lambda t: t.has_label(time_derivative),
                                        drop,
                                        replace_subject(self.Uin.split()))
        Frhs = a - L
        prob_rhs = NonlinearVariationalProblem(Frhs.form, self.Urhs)
        self.solver_rhs = NonlinearVariationalSolver(prob_rhs)

    def rnw_r(self, b, A=-1, B=1):
        # nodes and weights for gauss - radau IIA quadrature
        # See Abramowitz & Stegun p 888
        M = self.M
        a = 0
        nodes = np.zeros(M)
        nodes[0] = A
        p = np.poly1d([1, 1])
        pn = legendre(M)
        pn1 = legendre(M-1)
        poly, remainder = (pn + pn1)/p  # [1] returns remainder from polynomial division
        nodes[1:] = np.sort(poly.roots)
        weights = 1/M**2 * (1-nodes[1:])/(pn1(nodes[1:]))**2
        weights = np.append(2/M**2, weights)
        nodes = ((b - a) * nodes + a * B - b * A) / (B - A)
        weights = (b - a)/(B - A)*weights
        self.nodes = ((b + a) - nodes)[::-1]  # reverse nodes
        self.weights = weights[::-1]  # reverse weights

    def NewtonVM(self, t):
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

    def Horner_newton(self, weights, xi, x):
        """
        Horner scheme to evaluate polynomials based on newton basis
        """
        y = np.zeros_like(x)
        for i in range(len(weights)):
            y = y * (x - xi[(-i - 1)]) + weights[(-i - 1)]

        return y

    def gauss_legendre(self, n, b, A=-1, B=1):
        # nodes and weights for gauss legendre quadrature
        a = 0
        poly = legendre(n)
        polyd = poly.deriv()
        nodes= poly.roots
        nodes = np.sort(nodes)
        weights = 2/((1-nodes**2)*(np.polyval(polyd,nodes))**2)
        gl_nodes = ((b - a) * nodes + a * B - b * A) / (B - A)
        gl_weights=(b-a)/(B-A)*weights
        return gl_nodes, gl_weights

    def get_weights(self, b):
        # This calculates for equation 2.4 FWSW - called from Q
        # integrates lagrange polynomials to the points [nodes]
        M = self.M
        nodes_m, weights_m=self.gauss_legendre(np.ceil(M/2), b)  # use gauss-legendre quadrature to integrate polynomials
        weights = np.zeros(M)
        for j in np.arange(M):
            coeff = np.zeros(M)
            coeff[j] = 1.0  # is unity because it needs to be scaled with y_j for interpolation we have  sum y_j*l_j
            poly_coeffs = scipy.linalg.solve_triangular(self.NewtonVM(self.nodes), coeff, lower=True)
            eval_newt_poly = self.Horner_newton(poly_coeffs, self.nodes, nodes_m)
            weights[j] = np.dot(weights_m, eval_newt_poly)
        return weights

    def Qmatrix(self):
        """
        Integration Matrix 
        """
        M = self.M
        self.Q = np.zeros([M, M])

        # for all nodes, get weights for the interval [tleft,node]
        for m in np.arange(M):
            w = self.get_weights(self.nodes[m])
            self.Q[m, 0:] = w

    def Smatrix(self):
        """
        Integration matrix based on Q: sum(S@vector) returns integration
        """
        from copy import deepcopy
        M = self.M
        self.S = np.zeros([M, M])

        self.S[0, :] = deepcopy(self.Q[0, :])
        for m in np.arange(1, M):
            self.S[m, :] = self.Q[m, :] - self.Q[m - 1, :]

    def matmul_UFL(self, a, b):
        # b is nx1 array!
        n = np.shape(a)[0]
        result = [float(0)]*n
        for j in range(n):
            for k in range(n):
                result[j] += float(a[j,k])*b[k]
            result[j] = assemble(result[j])
        return result


    def apply(self, xin, xout):
        self.Un.assign(xin)

        self.Unodes[0].assign(self.Un)
        for m in range(self.M):
            self.IMEX.dt.assign(self.dtau[m])
            self.IMEX.apply(self.Unodes[m], self.Unodes[m+1])

        k = 0
        while k < self.maxk:
            k += 1

            fUnodes = []
            for m in range(1, M+1):
                self.Uin.assign(self.Unodes[m])
                self.solver_rhs.solve()
                UU = Function(self.W).assign(self.Urhs)
                fUnodes.append(UU)

            quad = self.matmul_UFL(self.S, fUnodes)
        
            #quad = dot(as_matrix(S),
            #            as_vector([f(Unodes[1]), f(Unodes[2]), f(Unodes[3])]))

            self.Unodes1[0].assign(self.Unodes[0])
            for m in range(1, M+1):
                self.dt.assign(self.dtau[m-1])
                self.U0.assign(self.Unodes[m-1])
                self.U01.assign(self.Unodes[m])
                self.Un.assign(self.Unodes1[m-1])
                self.Q_.assign(quad[m-1])
                self.solver_SDC.solve()
                self.Unodes1[m].assign(self.U_SDC)
            for m in range(1, M+1):
                self.Unodes[m].assign(self.Unodes1[m])

            self.Un.assign(self.Unodes1[-1])
        xout.assign(self.Un)

a = 0                           # time of start
b = 3                           # time of end
n = 512                         # number of spatial nodes
n_steps = 154.                  # number of time steps
dt = (b-a)/(n_steps)            # delta t

m = PeriodicIntervalMesh(n, 1)
H = 1.0e4  # Height position of the model top
mesh = ExtrudedMesh(m, layers=10, layer_height=H/10)

output = OutputParameters(dirname="acoustic")

state = State(mesh, dt=dt, output=output)

eqn = IncompressibleBoussinesqEquation(state)

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


# To Do:
# 1. Write weak form
# 2. Pass boundary conditions into variational problem
# 3. Correct initial conditions and problem parameters
