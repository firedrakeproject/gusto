from __future__ import absolute_import
from firedrake import split, LinearVariationalProblem, \
    LinearVariationalSolver, TestFunctions, TrialFunctions, \
    TestFunction, TrialFunction, lhs, rhs, DirichletBC, FacetNormal, \
    div, dx, jump, avg, dS_v, dS_h, inner, MixedFunctionSpace, dot, grad, \
    Function, Expression

from gusto.forcing import exner, exner_rho, exner_theta
from abc import ABCMeta, abstractmethod


class TimesteppingSolver(object):
    """
    Base class for timestepping linear solvers for Gusto.

    This is a dummy base class where the input is just copied to the output.

    :arg x_in: :class:`.Function` object for the input
    :arg x_out: :class:`.Function` object for the output
    """
    __metaclass__ = ABCMeta

    def __init__(self, state, params=None):

        self.state = state

        if params is None:
            self.params = {'pc_type': 'fieldsplit',
                           'pc_fieldsplit_type': 'schur',
                           'ksp_type': 'gmres',
                           'ksp_max_it': 100,
                           'ksp_gmres_restart': 50,
                           'pc_fieldsplit_schur_fact_type': 'FULL',
                           'pc_fieldsplit_schur_precondition': 'selfp',
                           'fieldsplit_0_ksp_type': 'preonly',
                           'fieldsplit_0_pc_type': 'bjacobi',
                           'fieldsplit_0_sub_pc_type': 'ilu',
                           'fieldsplit_1_ksp_type': 'preonly',
                           'fieldsplit_1_pc_type': 'gamg',
                           'fieldsplit_1_mg_levels_ksp_type': 'chebyshev',
                           'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues': True,
                           'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues_random': True,
                           'fieldsplit_1_mg_levels_ksp_max_it': 1,
                           'fieldsplit_1_mg_levels_pc_type': 'bjacobi',
                           'fieldsplit_1_mg_levels_sub_pc_type': 'ilu'}
        else:
            self.params = params

        # setup the solver
        self._setup_solver()

    @abstractmethod
    def solve(self):
        pass


class CompressibleSolver(TimesteppingSolver):
    """
    Timestepping linear solver object for the compressible equations
    in theta-pi formulation with prognostic variables u,rho,theta.

    This solver follows the following strategy:
    (1) Analytically eliminate theta (introduces error near topography)
    (2) Solve resulting system for (u,rho) using a Schur preconditioner
    (3) Reconstruct theta

    :arg state: a :class:`.State` object containing everything else.
    """

    def __init__(self, state, params=None):

        self.state = state

        if params is None:
            self.params = {'pc_type': 'fieldsplit',
                           'pc_fieldsplit_type': 'schur',
                           'ksp_type': 'gmres',
                           'ksp_max_it': 100,
                           'ksp_gmres_restart': 50,
                           'pc_fieldsplit_schur_fact_type': 'FULL',
                           'pc_fieldsplit_schur_precondition': 'selfp',
                           'fieldsplit_0_ksp_type': 'preonly',
                           'fieldsplit_0_pc_type': 'bjacobi',
                           'fieldsplit_0_sub_pc_type': 'ilu',
                           'fieldsplit_1_ksp_type': 'preonly',
                           'fieldsplit_1_pc_type': 'gamg',
                           'fieldsplit_1_mg_levels_ksp_type': 'chebyshev',
                           'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues': True,
                           'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues_random': True,
                           'fieldsplit_1_mg_levels_ksp_max_it': 1,
                           'fieldsplit_1_mg_levels_pc_type': 'bjacobi',
                           'fieldsplit_1_mg_levels_sub_pc_type': 'ilu'}
        else:
            self.params = params

        # setup the solver
        self._setup_solver()

    def _setup_solver(self):
        state = self.state      # just cutting down line length a bit
        beta = state.timestepping.dt*state.timestepping.alpha
        cp = state.parameters.cp
        mu = state.parameters.mu

        # Split up the rhs vector (symbolically)
        u_in, rho_in, theta_in = split(state.xrhs)

        # Build the reduced function space for u,rho
        M = MixedFunctionSpace((state.V[0], state.V[1]))
        w, phi = TestFunctions(M)
        u, rho = TrialFunctions(M)

        n = FacetNormal(state.mesh)

        # Get background fields
        thetabar = state.thetabar
        rhobar = state.rhobar
        pibar = exner(thetabar, rhobar, state)
        pibar_rho = exner_rho(thetabar, rhobar, state)
        pibar_theta = exner_theta(thetabar, rhobar, state)

        # Analytical (approximate) elimination of theta
        k = state.k             # Upward pointing unit vector
        theta = -dot(k,u)*dot(k,grad(thetabar))*beta + theta_in

        # Only include theta' (rather than pi') in the vertical
        # component of the gradient

        # the pi prime term (here, bars are for mean and no bars are
        # for linear perturbations)

        pi = pibar_theta*theta + pibar_rho*rho

        # vertical projection
        def V(u):
            return k*inner(u,k)

        eqn = (
            inner(w, (u - u_in))*dx
            - beta*cp*div(theta*V(w))*pibar*dx
            # following does nothing but is preserved in the comments
            # to remind us why (because V(w) is purely vertical.
            # + beta*cp*jump(theta*V(w),n)*avg(pibar)*dS_v
            - beta*cp*div(thetabar*w)*pi*dx
            + beta*cp*jump(thetabar*w,n)*avg(pi)*dS_v
            + (phi*(rho - rho_in) - beta*inner(grad(phi), u)*rhobar)*dx
            + beta*jump(phi*u, n)*avg(rhobar)*(dS_v + dS_h)
        )

        if mu is not None:
            eqn += beta*mu*inner(w,k)*inner(u,k)*dx
        aeqn = lhs(eqn)
        Leqn = rhs(eqn)

        # Place to put result of u rho solver
        self.urho = Function(M)

        # Boundary conditions (assumes extruded mesh)
        dim = M.sub(0).ufl_element().value_shape()[0]
        bc = ("0.0",)*dim
        bcs = [DirichletBC(M.sub(0), Expression(bc), "bottom"),
               DirichletBC(M.sub(0), Expression(bc), "top")]

        # Solver for u, rho
        urho_problem = LinearVariationalProblem(
            aeqn, Leqn, self.urho, bcs=bcs)

        self.urho_solver = LinearVariationalSolver(urho_problem,
                                                   solver_parameters=self.params)

        # Reconstruction of theta
        theta = TrialFunction(state.V[2])
        gamma = TestFunction(state.V[2])

        u, rho = self.urho.split()
        self.theta = Function(state.V[2])

        theta_eqn = gamma*(theta - theta_in +
                           dot(k,u)*dot(k,grad(thetabar))*beta)*dx

        theta_problem = LinearVariationalProblem(lhs(theta_eqn),
                                                 rhs(theta_eqn),
                                                 self.theta)
        self.theta_solver = LinearVariationalSolver(theta_problem)

    def solve(self):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """

        self.urho_solver.solve()

        u1, rho1 = self.urho.split()
        u, rho, theta = self.state.dy.split()
        u.assign(u1)
        rho.assign(rho1)

        self.theta_solver.solve()
        theta.assign(self.theta)


class ShallowWaterSolver(TimesteppingSolver):

    def _setup_solver(self):

        state = self.state
        H = state.parameters.H
        g = state.parameters.g
        beta = state.timestepping.dt*state.timestepping.alpha

        # Split up the rhs vector (symbolically)
        u_in, D_in = split(state.xrhs)

        W = state.W
        w, phi = TestFunctions(W)
        u, D = TrialFunctions(W)

        eqn = (
            inner(w, u) - beta*g*div(w)*D
            - inner(w, u_in)
            + phi*D + beta*H*phi*div(u)
            - phi*D_in
        )*dx

        aeqn = lhs(eqn)
        Leqn = rhs(eqn)

        # Place to put result of u rho solver
        self.uD = Function(W)

        # Solver for u, D
        uD_problem = LinearVariationalProblem(
            aeqn, Leqn, self.state.dy)

        self.uD_solver = LinearVariationalSolver(uD_problem,
                                                 solver_parameters=self.params)

    def solve(self):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """

        self.uD_solver.solve()
