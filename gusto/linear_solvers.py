from __future__ import absolute_import
from firedrake import split, LinearVariationalProblem, \
    LinearVariationalSolver, TestFunctions, TrialFunctions, \
    TestFunction, TrialFunction, lhs, rhs, DirichletBC, FacetNormal, \
    div, dx, jump, avg, dS_v, dS_h, inner, MixedFunctionSpace, dot, grad, \
    Function, Expression, MixedVectorSpaceBasis, VectorSpaceBasis, warning, \
    FunctionSpace, BrokenElement, ds_v, ds_t, ds_b, Tensor, assemble, \
    LinearSolver, Projector
from gusto.forcing import exner, exner_rho, exner_theta
from abc import ABCMeta, abstractmethod


class TimesteppingSolver(object):
    """
    Base class for timestepping linear solvers for Gusto.

    This is a dummy base class.

    :arg state: :class:`.State` object.
    :arg params (optional): solver parameters
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
    :arg quadrature degree: tuple (q_h, q_v) where q_h is the required
    quadrature degree in the horizontal direction and q_v is that in
    the vertical direction
    :arg params (optional): solver parameters
    """

    def __init__(self, state, quadrature_degree=None, params=None):

        self.state = state

        if quadrature_degree is not None:
            self.quadrature_degree = quadrature_degree
        else:
            dgspace = state.spaces("DG")
            if any(deg > 2 for deg in dgspace.ufl_element().degree()):
                warning("default quadrature degree most likely not sufficient for this degree element")
            self.quadrature_degree = (5, 5)

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
        dt = state.timestepping.dt
        beta = dt*state.timestepping.alpha
        cp = state.parameters.cp
        mu = state.mu
        Vu = state.spaces("HDiv")
        Vtheta = state.spaces("HDiv_v")
        Vrho = state.spaces("DG")

        # Split up the rhs vector (symbolically)
        u_in, rho_in, theta_in = split(state.xrhs)

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
        theta = -dot(k,u)*dot(k,grad(thetabar))*beta + theta_in

        # Only include theta' (rather than pi') in the vertical
        # component of the gradient

        # the pi prime term (here, bars are for mean and no bars are
        # for linear perturbations)

        pi = pibar_theta*theta + pibar_rho*rho

        # vertical projection
        def V(u):
            return k*inner(u,k)

        # specify degree for some terms as estimated degree is too large
        dxp = dx(degree=(self.quadrature_degree))
        dS_vp = dS_v(degree=(self.quadrature_degree))

        eqn = (
            inner(w, (u - u_in))*dx
            - beta*cp*div(theta*V(w))*pibar*dxp
            # following does nothing but is preserved in the comments
            # to remind us why (because V(w) is purely vertical.
            # + beta*cp*jump(theta*V(w),n)*avg(pibar)*dS_v
            - beta*cp*div(thetabar*w)*pi*dxp
            + beta*cp*jump(thetabar*w,n)*avg(pi)*dS_vp
            + (phi*(rho - rho_in) - beta*inner(grad(phi), u)*rhobar)*dx
            + beta*jump(phi*u, n)*avg(rhobar)*(dS_v + dS_h)
        )

        if mu is not None:
            eqn += dt*mu*inner(w,k)*inner(u,k)*dx
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
                                                   solver_parameters=self.params,
                                                   options_prefix='ImplicitSolver')

        # Reconstruction of theta
        theta = TrialFunction(Vtheta)
        gamma = TestFunction(Vtheta)

        u, rho = self.urho.split()
        self.theta = Function(Vtheta)

        theta_eqn = gamma*(theta - theta_in +
                           dot(k,u)*dot(k,grad(thetabar))*beta)*dx

        theta_problem = LinearVariationalProblem(lhs(theta_eqn),
                                                 rhs(theta_eqn),
                                                 self.theta)
        self.theta_solver = LinearVariationalSolver(theta_problem,
                                                    options_prefix='thetabacksubstitution')

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


class HybridisedCompressibleSolver(TimesteppingSolver):
    """
    Timestepping linear solver object for the compressible equations
    in theta-pi formulation with prognostic variables u,rho,theta.

    This solver follows the following strategy:
    (1) Analytically eliminate theta (introduces error near topography)
    (2a) Solve resulting system for (u[broken],rho,lambda) using hybridised
    solver
    (2b) reconstruct unbroken u
    (3) Reconstruct theta

    :arg state: a :class:`.State` object containing everything else.
    :arg quadrature degree: tuple (q_h, q_v) where q_h is the required
    quadrature degree in the horizontal direction and q_v is that in
    the vertical direction
    :arg params (optional): solver parameters
    """

    def __init__(self, state, quadrature_degree=None, params=None):

        self.state = state

        if quadrature_degree is not None:
            self.quadrature_degree = quadrature_degree
        else:
            dgspace = state.spaces("DG")
            if any(deg > 2 for deg in dgspace.ufl_element().degree()):
                warning("default quadrature degree most likely not sufficient for this degree element")
            self.quadrature_degree = (5, 5)

        self.params = params

        # setup the solver
        self._setup_solver()

    def _setup_solver(self):
        state = self.state      # just cutting down line length a bit
        dt = state.timestepping.dt
        beta = dt*state.timestepping.alpha
        cp = state.parameters.cp
        mu = state.mu
        Vu = state.spaces("HDiv")
        Vu_broken = FunctionSpace(state.mesh, BrokenElement(Vu.ufl_element()))
        Vtheta = state.spaces("HDiv_v")
        Vrho = state.spaces("DG")

        h_deg = state.horizontal_degree
        v_deg = state.vertical_degree
        Vtrace = FunctionSpace(state.mesh, "HDiv Trace", degree=(h_deg, v_deg))

        # Split up the rhs vector (symbolically)
        u_in, rho_in, theta_in = split(state.xrhs)

        # Build the reduced function space for u,rho
        M = MixedFunctionSpace((Vu_broken, Vrho))
        w, phi = TestFunctions(M)
        u, rho = TrialFunctions(M)

        l0 = TrialFunction(Vtrace)
        dl = TestFunction(Vtrace)

        n = FacetNormal(state.mesh)

        # Get background fields
        thetabar = state.fields("thetabar")
        rhobar = state.fields("rhobar")
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

        # specify degree for some terms as estimated degree is too large
        dxp = dx(degree=(self.quadrature_degree))
        dS_vp = dS_v(degree=(self.quadrature_degree))
        dS_hp = dS_h(degree=(self.quadrature_degree))
        ds_vp = ds_v(degree=(self.quadrature_degree))
        ds_tbp = ds_t(degree=(self.quadrature_degree)) + ds_b(degree=(self.quadrature_degree))

        rhobar_tr = Function(Vtrace)
        rbareqn = (l0('+') - avg(rhobar))*dl('+')*(dS_vp + dS_hp) + \
                  (l0 - rhobar)*dl*ds_vp + \
                  (l0 - rhobar)*dl*ds_tbp
        rhobar_prob = LinearVariationalProblem(lhs(rbareqn),rhs(rbareqn),rhobar_tr)
        self.rhobar_solver = LinearVariationalSolver(rhobar_prob,
                                                     solver_parameters={'ksp_type':'preonly',
                                                                        'pc_type':'bjacobi',
                                                                        'pc_sub_type':'lu'})

        Aeqn = (
            inner(w, (u - u_in))*dx
            - beta*cp*div(theta*V(w))*pibar*dxp
            - beta*cp*div(thetabar*w)*pi*dxp
            + (phi*(rho - rho_in) - beta*inner(grad(phi), u)*rhobar)*dx
            + beta*inner(phi*u, n)*rhobar_tr*(dS_v + dS_h)
        )
        if mu is not None:
            Aeqn += dt*mu*inner(w,k)*inner(u,k)*dx
        Aop = Tensor(lhs(Aeqn))
        Arhs = rhs(Aeqn)

        #  (A K)(U) = (U_r)
        #  (L 0)(l)   (0  )

        dl = dl('+')
        l0 = l0('+')

        K = Tensor(beta*cp*inner(thetabar*w,n)*l0*(dS_vp + dS_hp)
                   + beta*cp*inner(thetabar*w,n)*l0*ds_vp \
                   + beta*cp*inner(thetabar*w,n)*l0*ds_tbp)
        L = Tensor(dl*inner(u,n)*(dS_vp + dS_hp)\
                   + dl*inner(u,n)*ds_vp \
                   + dl*inner(u,n)*ds_tbp)

        #  U = A^{-1}(-Kl + U_r), 0=LU=-(LA^{-1}K)l + LA^{-1}U_r, so (LA^{-1}K)l = LA^{-1}U_r
        # reduced eqns for l0
        S = assemble(L * Aop.inv * K)
        self.Rexp = L * Aop.inv * Tensor(Arhs)
        # place to put the RHS (self.Rexp gets assembled in here)
        self.R = Function(M)

        # Set up the LinearSolver for the system of Lagrange multipliers
        self.lSolver = LinearSolver(S, solver_parameters=self.params)
        # a place to keep the solution
        self.lambdar = Function(Vtrace)

        # Place to put result of u rho solver
        self.urho = Function(M)

        # Reconstruction of broken u and rho
        self.ASolver = LinearSolver(assemble(Aop),
                                    solver_parameters={'ksp_type':'preonly',
                                                       'pc_type':'bjacobi',
                                                       'pc_sub_type':'lu'},
                                    options_prefix='urhoreconstruction')
        # Rhs for broken u and rho reconstruction
        self.Rurhoexp = -K*self.lambdar + Tensor(Arhs)
        self.Rurho = Function(M)
        u, rho = self.urho.split()

        self.u_hdiv = Function(Vu)
        self.u_projector = Projector(u, self.u_hdiv,
                                     solver_parameters={'ksp_type':'cg',
                                                        'pc_type':'bjacobi',
                                                        'pc_sub_type':'ilu'})

        # Reconstruction of theta
        theta = TrialFunction(Vtheta)
        gamma = TestFunction(Vtheta)

        self.theta = Function(Vtheta)

        u = self.u_hdiv
        theta_eqn = gamma*(theta - theta_in +
                           dot(k,u)*dot(k,grad(thetabar))*beta)*dx

        theta_problem = LinearVariationalProblem(lhs(theta_eqn),
                                                 rhs(theta_eqn),
                                                 self.theta)
        self.theta_solver = LinearVariationalSolver(theta_problem,
                                                    solver_parameters={'ksp_type':'cg',
                                                                       'pc_type':'bjacobi',
                                                                       'pc_sub_type':'ilu'},
                                                    options_prefix='thetabacksubstitution')

    def solve(self):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """

        # Assemble the RHS for lambda into self.R
        assemble(self.Rexp, tensor=self.R)
        # Solve for lambda
        self.lSolver.solve(self.lambdar, self.R)
        # Assemble the RHS for uhat, rho reconstruction
        assemble(self.Rurhoexp, tensor=self.Rurho)
        # Solve for uhat, rho
        self.ASolver.solve(self.urho, self.Rurho)
        # Project uhat as self.u_hdiv in H(div)
        self.u_projector.project()
        # copy back into u and rho cpts of dy
        _, rho1 = self.urho.split()
        u, rho, theta = self.state.dy.split()
        u.assign(self.u_hdiv)
        rho.assign(rho1)
        # reconstruct theta
        self.theta_solver.solve()
        # copy into theta cpt of dy
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
    :arg params: Solver parameters.
    """

    def __init__(self, state, L, params=None):

        self.state = state

        if params is None:
            self.params = {'ksp_type':'gmres',
                           'pc_type':'fieldsplit',
                           'pc_fieldsplit_type':'additive',
                           'fieldsplit_0_pc_type':'lu',
                           'fieldsplit_1_pc_type':'lu',
                           'fieldsplit_0_pc_factor_mat_solver_package': 'mumps',
                           'fieldsplit_0_pc_factor_mat_solver_package': 'mumps',
                           'fieldsplit_0_ksp_type':'preonly',
                           'fieldsplit_1_ksp_type':'preonly'}
        else:
            self.params = params

        self.L = L

        # setup the solver
        self._setup_solver()

    def _setup_solver(self):
        state = self.state      # just cutting down line length a bit
        dt = state.timestepping.dt
        beta = dt*state.timestepping.alpha
        mu = state.mu
        Vu = state.spaces("HDiv")
        Vb = state.spaces("HDiv_v")
        Vp = state.spaces("DG")

        # Split up the rhs vector (symbolically)
        u_in, p_in, b_in = split(state.xrhs)

        # Build the reduced function space for u,p
        M = MixedFunctionSpace((Vu, Vp))
        w, phi = TestFunctions(M)
        u, p = TrialFunctions(M)

        # Get background fields
        bbar = state.fields("bbar")

        # Analytical (approximate) elimination of theta
        k = state.k             # Upward pointing unit vector
        b = -dot(k,u)*dot(k,grad(bbar))*beta + b_in

        # vertical projection
        def V(u):
            return k*inner(u,k)

        eqn = (
            inner(w, (u - u_in))*dx
            - beta*div(w)*p*dx
            - beta*inner(w,k)*b*dx
            + phi*div(u)*dx
        )

        if mu is not None:
            eqn += dt*mu*inner(w,k)*inner(u,k)*dx
        aeqn = lhs(eqn)
        Leqn = rhs(eqn)

        # Place to put result of u p solver
        self.up = Function(M)

        # Boundary conditions (assumes extruded mesh)
        dim = M.sub(0).ufl_element().value_shape()[0]
        bc = ("0.0",)*dim
        bcs = [DirichletBC(M.sub(0), Expression(bc), "bottom"),
               DirichletBC(M.sub(0), Expression(bc), "top")]

        # preconditioner equation
        L = self.L
        Ap = (
            inner(w,u) + L*L*div(w)*div(u) +
            phi*p/L/L
        )*dx

        # Solver for u, p
        up_problem = LinearVariationalProblem(
            aeqn, Leqn, self.up, bcs=bcs, aP=Ap)

        nullspace = MixedVectorSpaceBasis(M,
                                          [M.sub(0),
                                           VectorSpaceBasis(constant=True)])

        self.up_solver = LinearVariationalSolver(up_problem,
                                                 solver_parameters=self.params,
                                                 nullspace=nullspace)

        # Reconstruction of b
        b = TrialFunction(Vb)
        gamma = TestFunction(Vb)

        u, p = self.up.split()
        self.b = Function(Vb)

        b_eqn = gamma*(b - b_in +
                       dot(k,u)*dot(k,grad(bbar))*beta)*dx

        b_problem = LinearVariationalProblem(lhs(b_eqn),
                                             rhs(b_eqn),
                                             self.b)
        self.b_solver = LinearVariationalSolver(b_problem)

    def solve(self):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """

        self.up_solver.solve()

        u1, p1 = self.up.split()
        u, p, b = self.state.dy.split()
        u.assign(u1)
        p.assign(p1)

        self.b_solver.solve()
        b.assign(self.b)


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
                                                 solver_parameters=self.params,
                                                 options_prefix='SWimplicit')

    def solve(self):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """

        self.uD_solver.solve()
