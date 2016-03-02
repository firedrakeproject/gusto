from __future__ import absolute_import
from firedrake import split, LinearVariationalProblem, \
    LinearVariationalSolver, TestFunctions, TrialFunctions, \
    TestFunction, TrialFunction, lhs, rhs, DirichletBC, FacetNormal, \
    div, dx, jump, avg, dS_v, dS_h, inner, MixedFunctionSpace, dot, grad, \
    Function

from dcore.forcing import exner, exner_rho, exner_theta
from abc import ABCMeta, abstractmethod


class TimesteppingSolver(object):
    """
    Base class for timestepping linear solvers for dcore.

    This is a dummy base class where the input is just copied to the output.

    :arg x_in: :class:`.Function` object for the input
    :arg x_out: :class:`.Function` object for the output
    """
    __metaclass__ = ABCMeta

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
    :arg alpha: off-centering parameter from [0,1] (default value 0.5).
    """

    def __init__(self, state, alpha=0.5, params=None):

        self.state = state
        self.alpha = alpha

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
        beta = state.dt*self.alpha
        cp = state.cp

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

        eqn = (
            (inner(w, u) - beta*cp*div(theta*w)*pibar)*dx
            + beta*cp*jump(theta*w,n)*avg(pibar)*dS_v
            - beta*cp*div(thetabar*w)*(pibar_theta*theta + pibar_rho*rho)*dx
            + beta*cp*jump(thetabar*w,n)*avg(
                pibar_theta*theta + pibar_rho*rho)*dS_v
            - inner(w, u_in)*dx
            + (phi*rho - beta*inner(grad(phi), u)*rhobar)*dx
            + beta*jump(phi*u, n)*avg(rhobar)*(dS_v + dS_h)
            - phi*rho_in*dx
        )

        aeqn = lhs(eqn)
        Leqn = rhs(eqn)

        # Place to put result of u rho solver
        self.urho = Function(M)
        # Boundary conditions (assumes extruded mesh)
        bcs = [DirichletBC(M.sub(0), 0.0, "bottom"),
               DirichletBC(M.sub(0), 0.0, "top")]

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

        theta_eqn = gamma*(theta - dot(k,u)*dot(k,grad(thetabar))*beta + theta_in)*dx
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
