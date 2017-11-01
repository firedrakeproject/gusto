from firedrake import split, LinearVariationalProblem, \
    LinearVariationalSolver, TestFunctions, TrialFunctions, \
    TestFunction, TrialFunction, lhs, rhs, DirichletBC, FacetNormal, \
    div, dx, jump, avg, dS_v, dS_h, inner, MixedFunctionSpace, dot, grad, \
    Function, MixedVectorSpaceBasis, VectorSpaceBasis, warning
from firedrake.solving_utils import flatten_parameters

from gusto.forcing import exner, exner_rho, exner_theta
from abc import ABCMeta, abstractmethod, abstractproperty


__all__ = ["CompressibleSolver", "IncompressibleSolver", "ShallowWaterSolver"]


class TimesteppingSolver(object, metaclass=ABCMeta):
    """
    Base class for timestepping linear solvers for Gusto.

    This is a dummy base class.

    :arg state: :class:`.State` object.
    :arg solver_parameters (optional): solver parameters
    :arg overwrite_solver_parameters: boolean, if True use only the
    solver_parameters that have been passed in, if False then update
    the default solver parameters with the solver_parameters passed in.
    """

    def __init__(self, state, solver_parameters=None,
                 overwrite_solver_parameters=False):

        self.state = state
        self.xrhs = Function(state.W)

        if solver_parameters is not None:
            if not overwrite_solver_parameters:
                p = flatten_parameters(self.solver_parameters)
                p.update(flatten_parameters(solver_parameters))
                solver_parameters = p
            self.solver_parameters = solver_parameters

        # setup the solver
        self._setup_solver()

    @abstractproperty
    def solver_parameters(self):
        """Solver parameters for this solver"""
        pass

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
    :arg quadrature degree: tuple (q_h, q_v) where q_h is the required
    quadrature degree in the horizontal direction and q_v is that in
    the vertical direction
    :arg solver_parameters (optional): solver parameters
    :arg overwrite_solver_parameters: boolean, if True use only the
    solver_parameters that have been passed in, if False then update
    the default solver parameters with the solver_parameters passed in.
    :arg moisture (optional): list of names of moisture fields.
    """

    solver_parameters = {
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'schur',
        'ksp_type': 'gmres',
        'ksp_max_it': 100,
        'ksp_gmres_restart': 50,
        'pc_fieldsplit_schur_fact_type': 'FULL',
        'pc_fieldsplit_schur_precondition': 'selfp',
        'fieldsplit_0': {'ksp_type': 'preonly',
                         'pc_type': 'bjacobi',
                         'sub_pc_type': 'ilu'},
        'fieldsplit_1': {'ksp_type': 'preonly',
                         'pc_type': 'gamg',
                         'mg_levels': {'ksp_type': 'chebyshev',
                                       'ksp_chebyshev_esteig': True,
                                       'ksp_max_it': 1,
                                       'pc_type': 'bjacobi',
                                       'sub_pc_type': 'ilu'}}
    }

    def __init__(self, state, quadrature_degree=None, solver_parameters=None,
                 overwrite_solver_parameters=False, moisture=None):

        self.moisture = moisture

        if quadrature_degree is not None:
            self.quadrature_degree = quadrature_degree
        else:
            dgspace = state.spaces("DG")
            if any(deg > 2 for deg in dgspace.ufl_element().degree()):
                warning("default quadrature degree most likely not sufficient for this degree element")
            self.quadrature_degree = (5, 5)

        super().__init__(state, solver_parameters, overwrite_solver_parameters)

    def _setup_solver(self):
        state = self.state      # just cutting down line length a bit
        dt = state.timestepping.dt
        beta = dt*state.timestepping.alpha
        cp = state.parameters.cp
        mu = state.mu
        Vu = state.spaces("HDiv")
        Vtheta = state.spaces("HDiv_v")
        Vrho = state.spaces("DG")

        # Split up the rhs vector (symbolically)
        u_in, rho_in, theta_in = split(self.xrhs)

        # Build the reduced function space for u,rho
        M = MixedFunctionSpace((Vu, Vrho))
        w, phi = TestFunctions(M)
        u, rho = TrialFunctions(M)

        n = FacetNormal(state.mesh)

        # Get background fields
        thetabar = state.fields("thetabar")
        rhobar = state.fields("rhobar")
        pibar = exner(thetabar, rhobar, state)
        pibar_rho = exner_rho(thetabar, rhobar, state)
        pibar_theta = exner_theta(thetabar, rhobar, state)

        # Analytical (approximate) elimination of theta
        k = state.k             # Upward pointing unit vector
        theta = -dot(k, u)*dot(k, grad(thetabar))*beta + theta_in

        # Only include theta' (rather than pi') in the vertical
        # component of the gradient

        # the pi prime term (here, bars are for mean and no bars are
        # for linear perturbations)

        pi = pibar_theta*theta + pibar_rho*rho

        # vertical projection
        def V(u):
            return k*inner(u, k)

        # specify degree for some terms as estimated degree is too large
        dxp = dx(degree=(self.quadrature_degree))
        dS_vp = dS_v(degree=(self.quadrature_degree))

        # add effect of density of water upon theta
        if self.moisture is not None:
            water_t = Function(Vtheta).assign(0.0)
            for water in self.moisture:
                water_t += self.state.fields(water)
            theta = theta / (1 + water_t)
            thetabar = thetabar / (1 + water_t)

        eqn = (
            inner(w, (u - u_in))*dx
            - beta*cp*div(theta*V(w))*pibar*dxp
            # following does nothing but is preserved in the comments
            # to remind us why (because V(w) is purely vertical.
            # + beta*cp*jump(theta*V(w),n)*avg(pibar)*dS_v
            - beta*cp*div(thetabar*w)*pi*dxp
            + beta*cp*jump(thetabar*w, n)*avg(pi)*dS_vp
            + (phi*(rho - rho_in) - beta*inner(grad(phi), u)*rhobar)*dx
            + beta*jump(phi*u, n)*avg(rhobar)*(dS_v + dS_h)
        )

        if mu is not None:
            eqn += dt*mu*inner(w, k)*inner(u, k)*dx
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
                                                   solver_parameters=self.solver_parameters,
                                                   options_prefix='ImplicitSolver')

        # Reconstruction of theta
        theta = TrialFunction(Vtheta)
        gamma = TestFunction(Vtheta)

        u, rho = self.urho.split()
        self.theta = Function(Vtheta)

        theta_eqn = gamma*(theta - theta_in +
                           dot(k, u)*dot(k, grad(thetabar))*beta)*dx

        theta_problem = LinearVariationalProblem(lhs(theta_eqn),
                                                 rhs(theta_eqn),
                                                 self.theta)
        self.theta_solver = LinearVariationalSolver(theta_problem,
                                                    options_prefix='thetabacksubstitution')

    def solve(self, xrhs, dy):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """

        self.xrhs.assign(xrhs)
        self.urho_solver.solve()

        u1, rho1 = self.urho.split()
        u, rho, theta = dy.split()
        u.assign(u1)
        rho.assign(rho1)

        self.theta_solver.solve()
        theta.assign(self.theta)


class IncompressibleSolver(TimesteppingSolver):
    """Timestepping linear solver object for the incompressible
    Boussinesq equations with prognostic variables u, p, b.

    This solver follows the following strategy:
    (1) Analytically eliminate b (introduces error near topography)
    (2) Solve resulting system for (u,p) using a block Hdiv preconditioner
    (3) Reconstruct b

    This currently requires a (parallel) direct solver so is probably
    a bit memory-hungry, we'll improve this with a hybridised solver
    soon.

    :arg state: a :class:`.State` object containing everything else.
    :arg L: the width of the domain, used in the preconditioner.
    :arg solver_parameters: (optional) Solver parameters.
    :arg overwrite_solver_parameters: boolean, if True use only the
    solver_parameters that have been passed in, if False then update
    the default solver parameters with the solver_parameters passed in.
    """

    solver_parameters = {
        'ksp_type': 'gmres',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'additive',
        'fieldsplit_0': {'ksp_type': 'preonly',
                         'pc_type': 'lu',
                         'pc_factor_mat_solver_package': 'mumps'},
        'fieldsplit_1': {'ksp_type': 'preonly',
                         'pc_type': 'lu',
                         'pc_factor_mat_solver_package': 'mumps'}
    }

    def __init__(self, state, L, solver_parameters=None,
                 overwrite_solver_parameters=False):

        self.L = L
        super().__init__(state, solver_parameters, overwrite_solver_parameters)

    def _setup_solver(self):
        state = self.state      # just cutting down line length a bit
        dt = state.timestepping.dt
        beta = dt*state.timestepping.alpha
        mu = state.mu
        Vu = state.spaces("HDiv")
        Vb = state.spaces("HDiv_v")
        Vp = state.spaces("DG")

        # Split up the rhs vector (symbolically)
        u_in, p_in, b_in = split(self.xrhs)

        # Build the reduced function space for u,p
        M = MixedFunctionSpace((Vu, Vp))
        w, phi = TestFunctions(M)
        u, p = TrialFunctions(M)

        # Get background fields
        bbar = state.fields("bbar")

        # Analytical (approximate) elimination of theta
        k = state.k             # Upward pointing unit vector
        b = -dot(k, u)*dot(k, grad(bbar))*beta + b_in

        # vertical projection
        def V(u):
            return k*inner(u, k)

        eqn = (
            inner(w, (u - u_in))*dx
            - beta*div(w)*p*dx
            - beta*inner(w, k)*b*dx
            + phi*div(u)*dx
        )

        if mu is not None:
            eqn += dt*mu*inner(w, k)*inner(u, k)*dx
        aeqn = lhs(eqn)
        Leqn = rhs(eqn)

        # Place to put result of u p solver
        self.up = Function(M)

        # Boundary conditions (assumes extruded mesh)
        bcs = [DirichletBC(M.sub(0), 0.0, "bottom"),
               DirichletBC(M.sub(0), 0.0, "top")]

        # preconditioner equation
        L = self.L
        Ap = (
            inner(w, u) + L*L*div(w)*div(u) +
            phi*p/L/L
        )*dx

        # Solver for u, p
        up_problem = LinearVariationalProblem(
            aeqn, Leqn, self.up, bcs=bcs, aP=Ap)

        nullspace = MixedVectorSpaceBasis(M,
                                          [M.sub(0),
                                           VectorSpaceBasis(constant=True)])

        self.up_solver = LinearVariationalSolver(up_problem,
                                                 solver_parameters=self.solver_parameters,
                                                 nullspace=nullspace)

        # Reconstruction of b
        b = TrialFunction(Vb)
        gamma = TestFunction(Vb)

        u, p = self.up.split()
        self.b = Function(Vb)

        b_eqn = gamma*(b - b_in +
                       dot(k, u)*dot(k, grad(bbar))*beta)*dx

        b_problem = LinearVariationalProblem(lhs(b_eqn),
                                             rhs(b_eqn),
                                             self.b)
        self.b_solver = LinearVariationalSolver(b_problem)

    def solve(self, xrhs, dy):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """

        self.xrhs.assign(xrhs)
        self.up_solver.solve()

        u1, p1 = self.up.split()
        u, p, b = dy.split()
        u.assign(u1)
        p.assign(p1)

        self.b_solver.solve()
        b.assign(self.b)


class ShallowWaterSolver(TimesteppingSolver):

    solver_parameters = {
        'ksp_type': 'preonly',
        'mat_type': 'matfree',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.HybridizationPC',
        'hybridization': {'ksp_type': 'cg',
                          'pc_type': 'gamg',
                          'ksp_rtol': 1e-8,
                          'mg_levels': {'ksp_type': 'chebyshev',
                                        'ksp_max_it': 2,
                                        'pc_type': 'bjacobi',
                                        'sub_pc_type': 'ilu'},
                          # Broken residual construction
                          'hdiv_residual': {'ksp_type': 'cg',
                                            'pc_type': 'bjacobi',
                                            'sub_pc_type': 'ilu',
                                            'ksp_rtol': 1e-8},
                          # Projection step
                          'hdiv_projection': {'ksp_type': 'cg',
                                              'ksp_rtol': 1e-8}}
    }

    def _setup_solver(self):
        state = self.state
        H = state.parameters.H
        g = state.parameters.g
        beta = state.timestepping.dt*state.timestepping.alpha

        # Split up the rhs vector (symbolically)
        u_in, D_in = split(self.xrhs)

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
            aeqn, Leqn, self.uD)

        self.uD_solver = LinearVariationalSolver(uD_problem,
                                                 solver_parameters=self.solver_parameters,
                                                 options_prefix='SWimplicit')

    def solve(self, xrhs, dy):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """

        self.xrhs.assign(xrhs)
        self.uD_solver.solve()
        dy.assign(self.uD)
