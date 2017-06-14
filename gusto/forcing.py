from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
from firedrake import Function, split, TrialFunction, TestFunction, \
    FacetNormal, inner, dx, cross, div, jump, avg, dS_v, \
    DirichletBC, LinearVariationalProblem, LinearVariationalSolver, \
    dot, dS, Constant, warning, Expression, as_vector


class Forcing(object):
    """
    Base class for forcing terms for Gusto.

    :arg state: x :class:`.State` object.
    :arg euler_poincare: if True then the momentum equation is in Euler
    Poincare form and we need to add 0.5*grad(u^2) to the forcing term.
    If False then this term is not added.
    :arg linear: if True then we are solving a linear equation so nonlinear
    terms (namely the Euler Poincare term) should not be added.
    """
    __metaclass__ = ABCMeta

    def __init__(self, state, euler_poincare=True, linear=False, extra_terms=None):
        self.state = state
        if linear:
            self.euler_poincare = False
            warning('Setting euler_poincare to False because you have set linear=True')
        else:
            self.euler_poincare = euler_poincare
        self.extra_terms = extra_terms
        self._build_forcing_solver()

    @abstractmethod
    def _build_forcing_solver(self):
        pass

    @abstractmethod
    def apply(self, scale, x, x_nl, x_out, **kwargs):
        """
        Function takes x as input, computes F(x_nl) and returns
        x_out = x + scale*F(x_nl)
        as output.

        :arg x: :class:`.Function` object
        :arg x_nl: :class:`.Function` object
        :arg x_out: :class:`.Function` object
        :arg mu_alpha: scale for sponge term, if present
        """
        pass


class CompressibleForcing(Forcing):
    """
    Forcing class for compressible Euler equations.
    """

    def _build_forcing_solver(self):
        """
        Only put forcing terms into the u equation.
        """

        state = self.state
        self.scaling = Constant(1.)
        Vu = state.spaces("HDiv")
        W = state.W

        self.x0 = Function(W)   # copy x to here

        u0,rho0,theta0 = split(self.x0)

        F = TrialFunction(Vu)
        w = TestFunction(Vu)
        self.uF = Function(Vu)

        Omega = state.Omega
        cp = state.parameters.cp
        mu = state.mu

        n = FacetNormal(state.mesh)

        pi = exner(theta0, rho0, state)

        a = inner(w,F)*dx
        L = self.scaling*(
            + cp*div(theta0*w)*pi*dx  # pressure gradient [volume]
            - cp*jump(w*theta0,n)*avg(pi)*dS_v  # pressure gradient [surface]
        )

        if state.geopotential_form:
            Phi = state.Phi
            L += self.scaling*div(w)*Phi*dx  # gravity term
        else:
            g = state.parameters.g
            L -= self.scaling*g*inner(w,state.k)*dx  # gravity term

        if self.euler_poincare:
            L -= self.scaling*0.5*div(w)*inner(u0, u0)*dx

        if Omega is not None:
            L -= self.scaling*inner(w,cross(2*Omega,u0))*dx  # Coriolis term

        if mu is not None:
            self.mu_scaling = Constant(1.)
            L -= self.mu_scaling*mu*inner(w,state.k)*inner(u0,state.k)*dx

        bcs = [DirichletBC(Vu, 0.0, "bottom"),
               DirichletBC(Vu, 0.0, "top")]

        if self.extra_terms is not None:
            L += self.scaling*inner(w, self.extra_terms)*dx

        u_forcing_problem = LinearVariationalProblem(
            a,L,self.uF, bcs=bcs
        )

        self.u_forcing_solver = LinearVariationalSolver(u_forcing_problem)

    def apply(self, scaling, x_in, x_nl, x_out, **kwargs):

        self.x0.assign(x_nl)
        self.scaling.assign(scaling)
        if 'mu_alpha' in kwargs and kwargs['mu_alpha'] is not None:
            self.mu_scaling.assign(kwargs['mu_alpha'])
        self.u_forcing_solver.solve()  # places forcing in self.uF

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


class IncompressibleForcing(Forcing):
    """
    Forcing class for incompressible Euler Boussinesq equations.
    """

    def _build_forcing_solver(self):
        """
        Only put forcing terms into the u equation.
        """

        state = self.state
        self.scaling = Constant(1.)
        Vu = state.spaces("HDiv")
        W = state.W

        self.x0 = Function(W)   # copy x to here

        u0,p0,b0 = split(self.x0)

        F = TrialFunction(Vu)
        w = TestFunction(Vu)
        self.uF = Function(Vu)

        Omega = state.Omega
        mu = state.mu

        a = inner(w,F)*dx
        L = (
            self.scaling*div(w)*p0*dx  # pressure gradient
            + self.scaling*b0*inner(w,state.k)*dx  # gravity term
        )

        if self.euler_poincare:
            L -= self.scaling*0.5*div(w)*inner(u0, u0)*dx

        if Omega is not None:
            L -= self.scaling*inner(w,cross(2*Omega,u0))*dx  # Coriolis term

        if mu is not None:
            self.mu_scaling = Constant(1.)
            L -= self.mu_scaling*mu*inner(w,state.k)*inner(u0,state.k)*dx

        bcs = [DirichletBC(Vu, 0.0, "bottom"),
               DirichletBC(Vu, 0.0, "top")]

        u_forcing_problem = LinearVariationalProblem(
            a,L,self.uF, bcs=bcs
        )

        self.u_forcing_solver = LinearVariationalSolver(u_forcing_problem)

        Vp = state.spaces("DG")
        p = TrialFunction(Vp)
        q = TestFunction(Vp)
        self.divu = Function(Vp)

        a = p*q*dx
        L = q*div(u0)*dx

        divergence_problem = LinearVariationalProblem(
            a, L, self.divu)

        self.divergence_solver = LinearVariationalSolver(divergence_problem)

    def apply(self, scaling, x_in, x_nl, x_out, **kwargs):

        self.x0.assign(x_nl)
        self.scaling.assign(scaling)
        if 'mu_alpha' in kwargs and kwargs['mu_alpha'] is not None:
            self.mu_scaling.assign(kwargs['mu_alpha'])
        self.u_forcing_solver.solve()  # places forcing in self.uF

        u_out, p_out, _ = x_out.split()

        x_out.assign(x_in)
        u_out += self.uF

        if 'incompressible' in kwargs and kwargs['incompressible']:
            self.divergence_solver.solve()
            p_out.assign(self.divu)


class EadyForcing(Forcing):
    """
    Forcing class for Eady Boussinesq equations.
    """

    def _build_forcing_solver(self):
        """
        Put forcing terms into the u & b equations.
        """

        state = self.state
        self.scaling = Constant(1.)
        Vu = state.spaces("HDiv")
        Vp = state.spaces("DG")
        W = state.W

        dbdy = state.parameters.dbdy
        H = state.parameters.H
        eady_exp = Function(Vp).interpolate(Expression(("x[2]-H/2"),H=H))

        self.x0 = Function(W)   # copy x to here

        u0,p0,b0 = split(self.x0)

        # u_forcing

        F = TrialFunction(Vu)
        w = TestFunction(Vu)
        self.uF = Function(Vu)

        Omega = state.Omega
        mu = state.mu

        a = inner(w,F)*dx
        L = self.scaling*(
            div(w)*p0  # pressure gradient
            + b0*inner(w,state.k)  # gravity term
            - dbdy*eady_exp*inner(w,as_vector([0.,1.,0.]))  # Eady forcing
        )*dx

        if self.euler_poincare:
            L -= self.scaling*0.5*div(w)*inner(u0, u0)*dx

        if Omega is not None:
            L -= self.scaling*inner(w,cross(2*Omega,u0))*dx  # Coriolis term

        if mu is not None:
            self.mu_scaling = Constant(1.)
            L -= self.mu_scaling*mu*inner(w,state.k)*inner(u0,state.k)*dx

        bcs = [DirichletBC(Vu, 0.0, "bottom"),
               DirichletBC(Vu, 0.0, "top")]

        u_forcing_problem = LinearVariationalProblem(
            a,L,self.uF, bcs=bcs
        )

        self.u_forcing_solver = LinearVariationalSolver(u_forcing_problem)

        # b_forcing

        Vb = state.spaces("HDiv_v")

        F = TrialFunction(Vb)
        gamma = TestFunction(Vb)
        self.bF = Function(Vb)

        a = gamma*F*dx
        L = -gamma*self.scaling*(dbdy*inner(u0,as_vector([0.,1.,0.])))*dx

        b_forcing_problem = LinearVariationalProblem(
            a,L,self.bF
        )

        self.b_forcing_solver = LinearVariationalSolver(b_forcing_problem)

        # divergence_free

        Vp = state.spaces("DG")
        p = TrialFunction(Vp)
        q = TestFunction(Vp)
        self.divu = Function(Vp)

        a = p*q*dx
        L = q*div(u0)*dx

        divergence_problem = LinearVariationalProblem(
            a, L, self.divu)

        self.divergence_solver = LinearVariationalSolver(divergence_problem)

    def apply(self, scaling, x_in, x_nl, x_out, **kwargs):

        self.x0.assign(x_nl)
        self.scaling.assign(scaling)
        if 'mu_alpha' in kwargs and kwargs['mu_alpha'] is not None:
            self.mu_scaling.assign(kwargs['mu_alpha'])
        self.u_forcing_solver.solve()  # places forcing in self.uF
        self.b_forcing_solver.solve()  # places forcing in self.bF

        u_out, p_out, b_out = x_out.split()

        x_out.assign(x_in)
        u_out += self.uF
        b_out += self.bF

        if kwargs.get("incompressible", False):
            self.divergence_solver.solve()
            p_out.assign(self.divu)


class CompressibleEadyForcing(Forcing):
    """
    Forcing class for compressible Eady equations.
    """

    def _build_forcing_solver(self):
        """
        Only put forcing terms into the u equation.
        """

        state = self.state
        self.scaling = Constant(1.)
        Vu = state.spaces("HDiv")
        W = state.W

        self.x0 = Function(W)   # copy x to here

        u0,rho0,theta0 = split(self.x0)

        # u_forcing

        F = TrialFunction(Vu)
        w = TestFunction(Vu)
        self.uF = Function(Vu)

        Omega = state.Omega
        cp = state.parameters.cp
        mu = state.mu
        dthetady = state.parameters.dthetady
        Pi0 = state.parameters.Pi0

        n = FacetNormal(state.mesh)

        Pi = exner(theta0, rho0, state)
        Pi_0 = Constant(Pi0)

        a = inner(w,F)*dx
        L = self.scaling*(
            + cp*div(theta0*w)*Pi*dx  # pressure gradient [volume]
            - cp*jump(w*theta0,n)*avg(Pi)*dS_v  # pressure gradient [surface]
            + cp*dthetady*(Pi-Pi_0)*inner(w,as_vector([0.,1.,0.]))*dx  # Eady forcing
        )

        if state.geopotential_form:
            Phi = state.Phi
            L += self.scaling*div(w)*Phi*dx  # gravity term
        else:
            g = state.parameters.g
            L -= self.scaling*g*inner(w,state.k)*dx  # gravity term

        if self.euler_poincare:
            L -= self.scaling*0.5*div(w)*inner(u0, u0)*dx

        if Omega is not None:
            L -= self.scaling*inner(w,cross(2*Omega,u0))*dx  # Coriolis term

        if mu is not None:
            self.mu_scaling = Constant(1.)
            L -= self.mu_scaling*mu*inner(w,state.k)*inner(u0,state.k)*dx

        bcs = [DirichletBC(Vu, 0.0, "bottom"),
               DirichletBC(Vu, 0.0, "top")]

        u_forcing_problem = LinearVariationalProblem(
            a,L,self.uF, bcs=bcs
        )

        self.u_forcing_solver = LinearVariationalSolver(u_forcing_problem)

        # theta_forcing

        Vt = state.spaces("HDiv_v")

        F = TrialFunction(Vt)
        gamma = TestFunction(Vt)
        self.thetaF = Function(Vt)

        a = gamma*F*dx
        L = -gamma*self.scaling*(dthetady*inner(u0,as_vector([0.,1.,0.])))*dx

        theta_forcing_problem = LinearVariationalProblem(
            a,L,self.thetaF
        )

        self.theta_forcing_solver = LinearVariationalSolver(theta_forcing_problem)

    def apply(self, scaling, x_in, x_nl, x_out, **kwargs):

        self.x0.assign(x_nl)
        self.scaling.assign(scaling)
        if 'mu_alpha' in kwargs and kwargs['mu_alpha'] is not None:
            self.mu_scaling.assign(kwargs['mu_alpha'])
        self.u_forcing_solver.solve()  # places forcing in self.uF
        self.theta_forcing_solver.solve()  # places forcing in self.thetaF

        u_out, _, theta_out = x_out.split()

        x_out.assign(x_in)
        u_out += self.uF
        theta_out += self.thetaF


class ShallowWaterForcing(Forcing):

    def _build_forcing_solver(self):

        state = self.state
        g = state.parameters.g
        f = state.fields("coriolis")

        Vu = state.spaces("HDiv")
        W = state.W

        self.x0 = Function(W)   # copy x to here

        u0, D0 = split(self.x0)
        n = FacetNormal(state.mesh)
        un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

        F = TrialFunction(Vu)
        w = TestFunction(Vu)
        self.uF = Function(Vu)

        a = inner(w, F)*dx
        L = (
            (-f*inner(w, state.perp(u0)) + g*div(w)*D0)*dx
            - g*inner(jump(w, n), un('+')*D0('+') - un('-')*D0('-'))*dS)

        if hasattr(state.fields, "topography"):
            b = state.fields("topography")
            L += g*div(w)*b*dx - g*inner(jump(w, n), un('+')*b('+') - un('-')*b('-'))*dS

        if self.euler_poincare:
            L -= 0.5*div(w)*inner(u0, u0)*dx

        u_forcing_problem = LinearVariationalProblem(
            a, L, self.uF)

        self.u_forcing_solver = LinearVariationalSolver(u_forcing_problem)

    def apply(self, scaling, x_in, x_nl, x_out, **kwargs):

        self.x0.assign(x_nl)

        self.u_forcing_solver.solve()  # places forcing in self.uF
        self.uF *= scaling

        uF, _ = x_out.split()

        x_out.assign(x_in)
        uF += self.uF


class NoForcing(Forcing):

    def _build_forcing_solver(self):
        pass

    def apply(self, scale, x_in, x_nl, x_out, **kwargs):

        x_out.assign(x_in)
