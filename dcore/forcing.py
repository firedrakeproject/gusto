from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
from firedrake import Function, split, TrialFunction, TestFunction, \
    FacetNormal, inner, dx, cross, div, jump, avg, dS_v, \
    DirichletBC, LinearVariationalProblem, LinearVariationalSolver, \
    CellNormal, dot, dS, Projector


class Forcing(object):
    """
    Base class for forcing terms for dcore.

    :arg state: x :class:`.State` object.
    """
    __metaclass__ = ABCMeta

    def __init__(self, state):
        self.state = state

    @abstractmethod
    def apply(self, scale, x, x_nl, x_out):
        """
        Function takes x as input, computes F(x_nl) and returns
        x_out = x + scale*F(x_nl)
        as output.

        :arg scale: parameter to scale the output by.
        :arg x: :class:`.Function` object
        :arg x_nl: :class:`.Function` object
        :arg x_out: :class:`.Function` object
        """
        pass


class CompressibleForcing(Forcing):
    """
    Forcing class for compressible Euler equations.
    """

    def __init__(self, state):
        self.state = state

        self._build_forcing_solver()

    def _build_forcing_solver(self):
        """
        Only put forcing terms into the u equation.
        """

        state = self.state
        Vu = state.V[0]
        W = state.W

        self.x0 = Function(W)   # copy x to here

        u0,rho0,theta0 = split(self.x0)

        F = TrialFunction(Vu)
        w = TestFunction(Vu)
        self.uF = Function(Vu)

        Omega = state.Omega
        cp = state.parameters.cp
        g = state.parameters.g

        n = FacetNormal(state.mesh)

        pi = Function(state.V[1])
        self.PiProjector = Projector(exner(theta0, rho0, state),
                                     pi)

        a = inner(w,F)*dx
        L = (
            + cp*div(theta0*w)*pi*dx  # pressure gradient [volume]
            - cp*jump(w*theta0,n)*avg(pi)*dS_v  # pressure gradient [surface]
            - g*inner(w,state.k)*dx  # gravity term
        )

        if Omega is not None:
            L -= inner(w,cross(2*Omega,u0))*dx  # Coriolis term

        bcs = [DirichletBC(Vu, 0.0, "bottom"),
               DirichletBC(Vu, 0.0, "top")]

        u_forcing_problem = LinearVariationalProblem(
            a,L,self.uF, bcs=bcs
        )

        self.u_forcing_solver = LinearVariationalSolver(u_forcing_problem)

    def apply(self, scaling, x_in, x_nl, x_out):

        self.x0.assign(x_nl)
        self.PiProjector.project()
        self.u_forcing_solver.solve()  # places forcing in self.uF
        self.uF *= scaling

        u_out, _, _ = x_out.split()

        x_out.assign(x_in)
        u_out += self.uF


def exner(theta,rho,state):
    """
    Compute the exner function.
    """
    R_d = state.parameters.R_d
    p_0 = state.parameters.p_0
    kappa = state.parameters.kappa

    return (R_d/p_0)**(kappa/(1-kappa))*pow(rho*theta, kappa/(1-kappa))


def exner_rho(theta,rho,state):
    R_d = state.parameters.R_d
    p_0 = state.parameters.p_0
    kappa = state.parameters.kappa

    return (R_d/p_0)**(kappa/(1-kappa))*pow(rho*theta, kappa/(1-kappa)-1)*theta*kappa/(1-kappa)


def exner_theta(theta,rho,state):
    R_d = state.parameters.R_d
    p_0 = state.parameters.p_0
    kappa = state.parameters.kappa

    return (R_d/p_0)**(kappa/(1-kappa))*pow(rho*theta, kappa/(1-kappa)-1)*rho*kappa/(1-kappa)


class ShallowWaterForcing(Forcing):

    def __init__(self, state):
        self.state = state

        g = state.parameters.g
        f = state.f

        Vu = state.V[0]
        W = state.W

        self.x0 = Function(W)   # copy x to here

        u0, D0 = split(self.x0)
        n = FacetNormal(state.mesh)
        un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

        F = TrialFunction(Vu)
        w = TestFunction(Vu)
        self.uF = Function(Vu)

        outward_normals = CellNormal(state.mesh)
        perp = lambda u: cross(outward_normals, u)
        a = inner(w, F)*dx
        L = (
            (-f*inner(w, perp(u0)) + g*div(w)*D0)*dx
            - g*inner(jump(w, n), un('+')*D0('+') - un('-')*D0('-'))*dS)

        u_forcing_problem = LinearVariationalProblem(
            a, L, self.uF)

        self.u_forcing_solver = LinearVariationalSolver(u_forcing_problem)

    def apply(self, scaling, x_in, x_nl, x_out):

        self.x0.assign(x_nl)

        self.u_forcing_solver.solve()  # places forcing in self.uF
        self.uF *= scaling

        uF, _ = x_out.split()

        x_out.assign(x_in)
        uF += self.uF
