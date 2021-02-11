from abc import ABCMeta, abstractmethod
from firedrake import (Function, split, TrialFunction, TestFunction,
                       FacetNormal, inner, dx, cross, div, jump, avg, dS_v,
                       LinearVariationalProblem, LinearVariationalSolver,
                       dot, dS, Constant, as_vector, SpatialCoordinate)
from gusto.configuration import logger, DEBUG
from gusto import thermodynamics


__all__ = ["CompressibleForcing", "IncompressibleForcing", "EadyForcing", "CompressibleEadyForcing", "ShallowWaterForcing"]


class Forcing(object, metaclass=ABCMeta):
    """
    Base class for forcing terms for Gusto.

    :arg state: x :class:`.State` object.
    :arg euler_poincare: if True then the momentum equation is in Euler
    Poincare form and we need to add 0.5*grad(u^2) to the forcing term.
    If False then this term is not added.
    :arg linear: if True then we are solving a linear equation so nonlinear
    terms (namely the Euler Poincare term) should not be added.
    :arg extra_terms: extra terms to add to the u component of the forcing
    term - these will be multiplied by the appropriate test function.
    """

    def __init__(self, state, euler_poincare=False, linear=False, extra_terms=None, moisture=None):
        self.state = state
        if linear:
            self.euler_poincare = False
            logger.warning('Setting euler_poincare to False because you have set linear=True')
        else:
            self.euler_poincare = euler_poincare

        # set up functions
        self.Vu = state.spaces("HDiv")
        # this is the function that the forcing term is applied to
        self.x0 = Function(state.W)
        self.test = TestFunction(self.Vu)
        self.trial = TrialFunction(self.Vu)
        # this is the function that contains the result of solving
        # <test, trial> = <test, F(x0)>, where F is the forcing term
        self.uF = Function(self.Vu)

        # find out which terms we need
        self.extruded = self.Vu.extruded
        self.coriolis = state.Omega is not None or hasattr(state.fields, "coriolis")
        self.sponge = state.mu is not None
        self.hydrostatic = state.hydrostatic
        self.topography = hasattr(state.fields, "topography")
        self.extra_terms = extra_terms
        self.moisture = moisture

        # some constants to use for scaling terms
        self.scaling = Constant(1.)
        self.impl = Constant(1.)

        self._build_forcing_solvers()

    def mass_term(self):
        return inner(self.test, self.trial)*dx

    def coriolis_term(self):
        u0 = split(self.x0)[0]
        return -inner(self.test, cross(2*self.state.Omega, u0))*dx

    def sponge_term(self):
        u0 = split(self.x0)[0]
        return self.state.mu*inner(self.test, self.state.k)*inner(u0, self.state.k)*dx

    def euler_poincare_term(self):
        u0 = split(self.x0)[0]
        return -0.5*div(self.test)*inner(self.state.h_project(u0), u0)*dx

    def hydrostatic_term(self):
        u0 = split(self.x0)[0]
        return inner(u0, self.state.k)*inner(self.test, self.state.k)*dx

    @abstractmethod
    def pressure_gradient_term(self):
        pass

    def forcing_term(self):
        L = self.pressure_gradient_term()
        if self.extruded:
            L += self.gravity_term()
        if self.coriolis:
            L += self.coriolis_term()
        if self.euler_poincare:
            L += self.euler_poincare_term()
        if self.topography:
            L += self.topography_term()
        if self.extra_terms is not None:
            L += inner(self.test, self.extra_terms)*dx
        # scale L
        L = self.scaling * L
        # sponge term has a separate scaling factor as it is always implicit
        if self.sponge:
            L -= self.impl*self.state.timestepping.dt*self.sponge_term()
        # hydrostatic term has no scaling factor
        if self.hydrostatic:
            L += (2*self.impl-1)*self.hydrostatic_term()
        return L

    def _build_forcing_solvers(self):
        a = self.mass_term()
        L = self.forcing_term()
        bcs = None if len(self.state.bcs) == 0 else self.state.bcs

        u_forcing_problem = LinearVariationalProblem(
            a, L, self.uF, bcs=bcs
        )

        solver_parameters = {}
        if logger.isEnabledFor(DEBUG):
            solver_parameters["ksp_monitor_true_residual"] = None
        self.u_forcing_solver = LinearVariationalSolver(
            u_forcing_problem,
            solver_parameters=solver_parameters,
            options_prefix="UForcingSolver"
        )

    def apply(self, scaling, x_in, x_nl, x_out, **kwargs):
        """
        Function takes x as input, computes F(x_nl) and returns
        x_out = x + scale*F(x_nl)
        as output.

        :arg x_in: :class:`.Function` object
        :arg x_nl: :class:`.Function` object
        :arg x_out: :class:`.Function` object
        :arg implicit: forcing stage for sponge and hydrostatic terms, if present
        """
        self.scaling.assign(scaling)
        self.x0.assign(x_nl)
        implicit = kwargs.get("implicit")
        if implicit is not None:
            self.impl.assign(int(implicit))
        self.u_forcing_solver.solve()  # places forcing in self.uF

        uF = x_out.split()[0]

        x_out.assign(x_in)
        uF += self.uF


class CompressibleForcing(Forcing):
    """
    Forcing class for compressible Euler equations.
    """

    def pressure_gradient_term(self):

        u0, rho0, theta0 = split(self.x0)
        cp = self.state.parameters.cp
        n = FacetNormal(self.state.mesh)
        Vtheta = self.state.spaces("HDiv_v")

        # introduce new theta so it can be changed by moisture
        theta = theta0

        # add effect of density of water upon theta
        if self.moisture is not None:
            water_t = Function(Vtheta).assign(0.0)
            for water in self.moisture:
                water_t += self.state.fields(water)
            theta = theta / (1 + water_t)

        pi = thermodynamics.pi(self.state.parameters, rho0, theta0)

        L = (
            + cp*div(theta*self.test)*pi*dx
            - cp*jump(self.test*theta, n)*avg(pi)*dS_v
        )
        return L

    def gravity_term(self):

        g = self.state.parameters.g
        L = -g*inner(self.test, self.state.k)*dx

        return L

    def theta_forcing(self):

        cv = self.state.parameters.cv
        cp = self.state.parameters.cp
        c_vv = self.state.parameters.c_vv
        c_pv = self.state.parameters.c_pv
        c_pl = self.state.parameters.c_pl
        R_d = self.state.parameters.R_d
        R_v = self.state.parameters.R_v

        u0, _, theta0 = split(self.x0)
        water_v = self.state.fields('water_v')
        water_c = self.state.fields('water_c')

        c_vml = cv + water_v * c_vv + water_c * c_pl
        c_pml = cp + water_v * c_pv + water_c * c_pl
        R_m = R_d + water_v * R_v

        L = -theta0 * (R_m / c_vml - (R_d * c_pml) / (cp * c_vml)) * div(u0)

        return self.scaling * L

    def _build_forcing_solvers(self):

        super(CompressibleForcing, self)._build_forcing_solvers()
        # build forcing for theta equation
        if self.moisture is not None:
            _, _, theta0 = split(self.x0)
            Vt = self.state.spaces("HDiv_v")
            p = TrialFunction(Vt)
            q = TestFunction(Vt)
            self.thetaF = Function(Vt)

            a = p * q * dx
            L = self.theta_forcing()
            L = q * L * dx

            theta_problem = LinearVariationalProblem(a, L, self.thetaF)

            solver_parameters = {}
            if logger.isEnabledFor(DEBUG):
                solver_parameters["ksp_monitor_true_residual"] = None
            self.theta_solver = LinearVariationalSolver(
                theta_problem,
                solver_parameters=solver_parameters,
                options_prefix="ThetaForcingSolver"
            )

    def apply(self, scaling, x_in, x_nl, x_out, **kwargs):

        super(CompressibleForcing, self).apply(scaling, x_in, x_nl, x_out, **kwargs)
        if self.moisture is not None:
            self.theta_solver.solve()
            _, _, theta_out = x_out.split()
            theta_out += self.thetaF


class IncompressibleForcing(Forcing):
    """
    Forcing class for incompressible Euler Boussinesq equations.
    """

    def pressure_gradient_term(self):
        _, p0, _ = split(self.x0)
        L = div(self.test)*p0*dx
        return L

    def gravity_term(self):
        _, _, b0 = split(self.x0)
        L = b0*inner(self.test, self.state.k)*dx
        return L

    def _build_forcing_solvers(self):

        super(IncompressibleForcing, self)._build_forcing_solvers()
        Vp = self.state.spaces("DG")
        p = TrialFunction(Vp)
        q = TestFunction(Vp)
        self.divu = Function(Vp)

        u0, _, _ = split(self.x0)
        a = p*q*dx
        L = q*div(u0)*dx

        divergence_problem = LinearVariationalProblem(
            a, L, self.divu)

        solver_parameters = {}

        if logger.isEnabledFor(DEBUG):
            solver_parameters["ksp_monitor_true_residual"] = None
        self.divergence_solver = LinearVariationalSolver(
            divergence_problem,
            solver_parameters=solver_parameters,
            options_prefix="DivergenceSolver"
        )

    def apply(self, scaling, x_in, x_nl, x_out, **kwargs):

        super(IncompressibleForcing, self).apply(scaling, x_in, x_nl, x_out, **kwargs)
        if 'incompressible' in kwargs and kwargs['incompressible']:
            _, p_out, _ = x_out.split()
            self.divergence_solver.solve()
            p_out.assign(self.divu)


class EadyForcing(IncompressibleForcing):
    """
    Forcing class for Eady Boussinesq equations.
    """

    def forcing_term(self):

        L = Forcing.forcing_term(self)
        dbdy = self.state.parameters.dbdy
        H = self.state.parameters.H
        Vp = self.state.spaces("DG")
        _, _, z = SpatialCoordinate(self.state.mesh)
        eady_exp = Function(Vp).interpolate(z-H/2.)

        L -= self.scaling*dbdy*eady_exp*inner(self.test, as_vector([0., 1., 0.]))*dx
        return L

    def _build_forcing_solvers(self):

        super(EadyForcing, self)._build_forcing_solvers()

        # b_forcing
        dbdy = self.state.parameters.dbdy
        Vb = self.state.spaces("HDiv_v")
        F = TrialFunction(Vb)
        gamma = TestFunction(Vb)
        self.bF = Function(Vb)
        u0, _, b0 = split(self.x0)

        a = gamma*F*dx
        L = -self.scaling*gamma*(dbdy*inner(u0, as_vector([0., 1., 0.])))*dx

        b_forcing_problem = LinearVariationalProblem(
            a, L, self.bF
        )

        solver_parameters = {}
        if logger.isEnabledFor(DEBUG):
            solver_parameters["ksp_monitor_true_residual"] = None
        self.b_forcing_solver = LinearVariationalSolver(
            b_forcing_problem,
            solver_parameters=solver_parameters,
            options_prefix="BForcingSolver"
        )

    def apply(self, scaling, x_in, x_nl, x_out, **kwargs):

        super(EadyForcing, self).apply(scaling, x_in, x_nl, x_out, **kwargs)
        self.b_forcing_solver.solve()  # places forcing in self.bF
        _, _, b_out = x_out.split()
        b_out += self.bF


class CompressibleEadyForcing(CompressibleForcing):
    """
    Forcing class for compressible Eady equations.
    """

    def forcing_term(self):

        # L = super(EadyForcing, self).forcing_term()
        L = Forcing.forcing_term(self)
        dthetady = self.state.parameters.dthetady
        Pi0 = self.state.parameters.Pi0
        cp = self.state.parameters.cp

        _, rho0, theta0 = split(self.x0)
        Pi = thermodynamics.pi(self.state.parameters, rho0, theta0)
        Pi_0 = Constant(Pi0)

        L += self.scaling*cp*dthetady*(Pi-Pi_0)*inner(self.test, as_vector([0., 1., 0.]))*dx  # Eady forcing
        return L

    def _build_forcing_solvers(self):

        super(CompressibleEadyForcing, self)._build_forcing_solvers()
        # theta_forcing
        dthetady = self.state.parameters.dthetady
        Vt = self.state.spaces("HDiv_v")
        F = TrialFunction(Vt)
        gamma = TestFunction(Vt)
        self.thetaF = Function(Vt)
        u0, _, _ = split(self.x0)

        a = gamma*F*dx
        L = -self.scaling*gamma*(dthetady*inner(u0, as_vector([0., 1., 0.])))*dx

        theta_forcing_problem = LinearVariationalProblem(
            a, L, self.thetaF
        )

        solver_parameters = {}
        if logger.isEnabledFor(DEBUG):
            solver_parameters["ksp_monitor_true_residual"] = None
        self.theta_forcing_solver = LinearVariationalSolver(
            theta_forcing_problem,
            solver_parameters=solver_parameters,
            options_prefix="ThetaForcingSolver"
        )

    def apply(self, scaling, x_in, x_nl, x_out, **kwargs):

        Forcing.apply(self, scaling, x_in, x_nl, x_out, **kwargs)
        self.theta_forcing_solver.solve()  # places forcing in self.thetaF
        _, _, theta_out = x_out.split()
        theta_out += self.thetaF


class ShallowWaterForcing(Forcing):

    def coriolis_term(self):

        f = self.state.fields("coriolis")
        u0, _ = split(self.x0)
        L = -f*inner(self.test, self.state.perp(u0))*dx
        return L

    def pressure_gradient_term(self):

        g = self.state.parameters.g
        u0, D0 = split(self.x0)
        n = FacetNormal(self.state.mesh)
        un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

        L = g*(div(self.test)*D0*dx
               - inner(jump(self.test, n), un('+')*D0('+')
                       - un('-')*D0('-'))*dS)
        return L

    def topography_term(self):
        g = self.state.parameters.g
        u0, _ = split(self.x0)
        b = self.state.fields("topography")
        n = FacetNormal(self.state.mesh)
        un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

        L = g*div(self.test)*b*dx - g*inner(jump(self.test, n), un('+')*b('+') - un('-')*b('-'))*dS
        return L
