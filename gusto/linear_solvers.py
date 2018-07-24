from firedrake import split, LinearVariationalProblem, \
    LinearVariationalSolver, TestFunctions, TrialFunctions, \
    TestFunction, TrialFunction, lhs, rhs, DirichletBC, FacetNormal, \
    div, dx, jump, avg, dS_v, dS_h, ds_v, ds_t, ds_b, inner, dot, grad, \
    Function, VectorSpaceBasis, BrokenElement, FunctionSpace, MixedFunctionSpace, \
    assemble, LinearSolver, Tensor, AssembledVector
from firedrake.solving_utils import flatten_parameters
from firedrake.parloops import par_loop, READ, INC
from pyop2.profiling import timed_function, timed_region

from gusto.configuration import DEBUG
from gusto import thermodynamics
from abc import ABCMeta, abstractmethod, abstractproperty


__all__ = ["CompressibleSolver", "IncompressibleSolver", "ShallowWaterSolver",
           "HybridizedCompressibleSolver"]


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

        if solver_parameters is not None:
            if not overwrite_solver_parameters:
                p = flatten_parameters(self.solver_parameters)
                p.update(flatten_parameters(solver_parameters))
                solver_parameters = p
            self.solver_parameters = solver_parameters

        if state.output.log_level == DEBUG:
            self.solver_parameters["ksp_monitor_true_residual"] = True

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
                state.logger.warning("default quadrature degree most likely not sufficient for this degree element")
            self.quadrature_degree = (5, 5)

        super().__init__(state, solver_parameters, overwrite_solver_parameters)

    @timed_function("Gusto:SolverSetup")
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
        pibar = thermodynamics.pi(state.parameters, rhobar, thetabar)
        pibar_rho = thermodynamics.pi_rho(state.parameters, rhobar, thetabar)
        pibar_theta = thermodynamics.pi_theta(state.parameters, rhobar, thetabar)

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
            theta_w = theta / (1 + water_t)
            thetabar_w = thetabar / (1 + water_t)
        else:
            theta_w = theta
            thetabar_w = thetabar

        eqn = (
            inner(w, (state.h_project(u) - u_in))*dx
            - beta*cp*div(theta_w*V(w))*pibar*dxp
            # following does nothing but is preserved in the comments
            # to remind us why (because V(w) is purely vertical).
            # + beta*cp*jump(theta*V(w), n)*avg(pibar)*dS_v
            - beta*cp*div(thetabar_w*w)*pi*dxp
            + beta*cp*jump(thetabar_w*w, n)*avg(pi)*dS_vp
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

    @timed_function("Gusto:LinearSolve")
    def solve(self):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """

        with timed_region("Gusto:VelocityDensitySolve"):
            self.urho_solver.solve()

        u1, rho1 = self.urho.split()
        u, rho, theta = self.state.dy.split()
        u.assign(u1)
        rho.assign(rho1)

        with timed_region("Gusto:ThetaRecon"):
            self.theta_solver.solve()

        theta.assign(self.theta)


class HybridizedCompressibleSolver(TimesteppingSolver):
    """
    Timestepping linear solver object for the compressible equations
    in theta-pi formulation with prognostic variables u, rho, and theta.

    This solver follows the following strategy:

    (1) Analytically eliminate theta (introduces error near topography)

    (2a) Formulate the resulting mixed system for u and rho using a
         hybridized mixed method. This breaks continuity in the
         linear perturbations of u, and introduces a new unknown on the
         mesh interfaces approximating the average of the Exner pressure
         perturbations. These trace unknowns also act as Lagrange
         multipliers enforcing normal continuity of the "broken" u variable.

    (2b) Statically condense the block-sparse system into a single system
         for the Lagrange multipliers. This is the only globally coupled
         system requiring a linear solver.

    (2c) Using the computed trace variables, we locally recover the
         broken velocity and density perturbations. This is accomplished
         in two stages:
         (i): Recover rho locally using the multipliers.
         (ii): Recover "broken" u locally using rho and the multipliers.

    (2d) Project the "broken" velocity field into the HDiv-conforming
         space using local averaging.

    (3) Reconstruct theta

    :arg state: a :class:`.State` object containing everything else.
    :arg quadrature degree: tuple (q_h, q_v) where q_h is the required
         quadrature degree in the horizontal direction and q_v is that in
         the vertical direction.
    :arg solver_parameters (optional): solver parameters for the
         trace system.
    :arg overwrite_solver_parameters: boolean, if True use only the
         solver_parameters that have been passed in, if False then update.
         the default solver parameters with the solver_parameters passed in.
    :arg moisture (optional): list of names of moisture fields.
    """

    # Solver parameters for the Lagrange multiplier system
    # NOTE: The reduced operator is not symmetric
    solver_parameters = {'ksp_type': 'gmres',
                         'pc_type': 'gamg',
                         'ksp_rtol': 1.0e-8,
                         'mg_levels': {'ksp_type': 'richardson',
                                       'ksp_max_it': 2,
                                       'pc_type': 'bjacobi',
                                       'sub_pc_type': 'ilu'}}

    def __init__(self, state, quadrature_degree=None, solver_parameters=None,
                 overwrite_solver_parameters=False, moisture=None):

        self.moisture = moisture

        self.state = state

        if quadrature_degree is not None:
            self.quadrature_degree = quadrature_degree
        else:
            dgspace = state.spaces("DG")
            if any(deg > 2 for deg in dgspace.ufl_element().degree()):
                state.logger.warning("default quadrature degree most likely not sufficient for this degree element")
            self.quadrature_degree = (5, 5)

        super().__init__(state, solver_parameters, overwrite_solver_parameters)

    @timed_function("Gusto:SolverSetup")
    def _setup_solver(self):
        from firedrake.assemble import create_assembly_callable
        import numpy as np

        state = self.state
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

        # Build the function space for "broken" u and rho
        # and add the trace variable
        M = MixedFunctionSpace((Vu_broken, Vrho))
        w, phi = TestFunctions(M)
        u, rho = TrialFunctions(M)
        l0 = TrialFunction(Vtrace)
        dl = TestFunction(Vtrace)

        n = FacetNormal(state.mesh)

        # Get background fields
        thetabar = state.fields("thetabar")
        rhobar = state.fields("rhobar")
        pibar = thermodynamics.pi(state.parameters, rhobar, thetabar)
        pibar_rho = thermodynamics.pi_rho(state.parameters, rhobar, thetabar)
        pibar_theta = thermodynamics.pi_theta(state.parameters, rhobar, thetabar)

        # Analytical (approximate) elimination of theta
        k = state.k             # Upward pointing unit vector
        theta = -dot(k, u)*dot(k, grad(thetabar))*beta + theta_in

        # Only include theta' (rather than pi') in the vertical
        # component of the gradient

        # The pi prime term (here, bars are for mean and no bars are
        # for linear perturbations)
        pi = pibar_theta*theta + pibar_rho*rho

        # Vertical projection
        def V(u):
            return k*inner(u, k)

        # Specify degree for some terms as estimated degree is too large
        dxp = dx(degree=(self.quadrature_degree))
        dS_vp = dS_v(degree=(self.quadrature_degree))
        dS_hp = dS_h(degree=(self.quadrature_degree))
        ds_vp = ds_v(degree=(self.quadrature_degree))
        ds_tbp = ds_t(degree=(self.quadrature_degree)) + ds_b(degree=(self.quadrature_degree))

        # Mass matrix for the trace space
        tM = assemble(dl('+')*l0('+')*(dS_v + dS_h)
                      + dl*l0*ds_v + dl*l0*(ds_t + ds_b))

        Lrhobar = Function(Vtrace)
        Lpibar = Function(Vtrace)
        rhopi_solver = LinearSolver(tM, solver_parameters={'ksp_type': 'cg',
                                                           'pc_type': 'bjacobi',
                                                           'sub_pc_type': 'ilu'},
                                    options_prefix='rhobarpibar_solver')

        rhobar_avg = Function(Vtrace)
        pibar_avg = Function(Vtrace)

        def _traceRHS(f):
            return (dl('+')*avg(f)*(dS_v + dS_h)
                    + dl*f*ds_v + dl*f*(ds_t + ds_b))

        assemble(_traceRHS(rhobar), tensor=Lrhobar)
        assemble(_traceRHS(pibar), tensor=Lpibar)

        # Project averages of coefficients into the trace space
        with timed_region("Gusto:HybridProjectRhobar"):
            rhopi_solver.solve(rhobar_avg, Lrhobar)

        with timed_region("Gusto:HybridProjectPibar"):
            rhopi_solver.solve(pibar_avg, Lpibar)

        # Add effect of density of water upon theta
        if self.moisture is not None:
            water_t = Function(Vtheta).assign(0.0)
            for water in self.moisture:
                water_t += self.state.fields(water)
            theta_w = theta / (1 + water_t)
            thetabar_w = thetabar / (1 + water_t)
        else:
            theta_w = theta
            thetabar_w = thetabar

        # "broken" u and rho system
        Aeqn = (inner(w, (state.h_project(u) - u_in))*dx
                - beta*cp*div(theta_w*V(w))*pibar*dxp
                # following does nothing but is preserved in the comments
                # to remind us why (because V(w) is purely vertical).
                # + beta*cp*dot(theta_w*V(w), n)*pibar_avg('+')*dS_vp
                + beta*cp*dot(theta_w*V(w), n)*pibar_avg('+')*dS_hp
                + beta*cp*dot(theta_w*V(w), n)*pibar_avg*ds_tbp
                - beta*cp*div(thetabar_w*w)*pi*dxp
                + (phi*(rho - rho_in) - beta*inner(grad(phi), u)*rhobar)*dx
                + beta*dot(phi*u, n)*rhobar_avg('+')*(dS_v + dS_h))

        if mu is not None:
            Aeqn += dt*mu*inner(w, k)*inner(u, k)*dx

        # Form the mixed operators using Slate
        # (A   K)(X) = (X_r)
        # (K.T 0)(l)   (0  )
        # where X = ("broken" u, rho)
        A = Tensor(lhs(Aeqn))
        X_r = Tensor(rhs(Aeqn))

        # Off-diagonal block matrices containing the contributions
        # of the Lagrange multipliers (surface terms in the momentum equation)
        K = Tensor(beta*cp*dot(thetabar_w*w, n)*l0('+')*(dS_vp + dS_hp)
                   + beta*cp*dot(thetabar_w*w, n)*l0*ds_vp
                   + beta*cp*dot(thetabar_w*w, n)*l0*ds_tbp)

        # X = A.inv * (X_r - K * l),
        # 0 = K.T * X = -(K.T * A.inv * K) * l + K.T * A.inv * X_r,
        # so (K.T * A.inv * K) * l = K.T * A.inv * X_r
        # is the reduced equation for the Lagrange multipliers.
        # Right-hand side expression: (Forward substitution)
        Rexp = K.T * A.inv * X_r
        self.R = Function(Vtrace)

        # We need to rebuild R everytime data changes
        self._assemble_Rexp = create_assembly_callable(Rexp, tensor=self.R)

        # Schur complement operator:
        Smatexp = K.T * A.inv * K
        with timed_region("Gusto:HybridAssembleTraceOp"):
            S = assemble(Smatexp)
            S.force_evaluation()

        # Set up the Linear solver for the system of Lagrange multipliers
        self.lSolver = LinearSolver(S, solver_parameters=self.solver_parameters,
                                    options_prefix='lambda_solve')

        # Result function for the multiplier solution
        self.lambdar = Function(Vtrace)

        # Place to put result of u rho reconstruction
        self.urho = Function(M)

        # Reconstruction of broken u and rho
        u_, rho_ = self.urho.split()

        # Split operators for two-stage reconstruction
        _A = A.blocks
        _K = K.blocks
        _Xr = X_r.blocks

        A00 = _A[0, 0]
        A01 = _A[0, 1]
        A10 = _A[1, 0]
        A11 = _A[1, 1]
        K0 = _K[0, 0]
        Ru = _Xr[0]
        Rrho = _Xr[1]
        lambda_vec = AssembledVector(self.lambdar)

        # rho reconstruction
        Srho = A11 - A10 * A00.inv * A01
        rho_expr = Srho.solve(Rrho - A10 * A00.inv * (Ru - K0 * lambda_vec),
                              decomposition="PartialPivLU")
        self._assemble_rho = create_assembly_callable(rho_expr, tensor=rho_)

        # "broken" u reconstruction
        rho_vec = AssembledVector(rho_)
        u_expr = A00.solve(Ru - A01 * rho_vec - K0 * lambda_vec,
                           decomposition="PartialPivLU")
        self._assemble_u = create_assembly_callable(u_expr, tensor=u_)

        # Project broken u into the HDiv space using facet averaging.
        # Weight function counting the dofs of the HDiv element:
        shapes = (Vu.finat_element.space_dimension(), np.prod(Vu.shape))

        weight_kernel = """
        for (int i=0; i<%d; ++i) {
        for (int j=0; j<%d; ++j) {
        w[i][j] += 1.0;
        }}""" % shapes

        self._weight = Function(Vu)
        par_loop(weight_kernel, dx, {"w": (self._weight, INC)})

        # Averaging kernel
        self._average_kernel = """
        for (int i=0; i<%d; ++i) {
        for (int j=0; j<%d; ++j) {
        vec_out[i][j] += vec_in[i][j]/w[i][j];
        }}""" % shapes

        # HDiv-conforming velocity
        self.u_hdiv = Function(Vu)

        # Reconstruction of theta
        theta = TrialFunction(Vtheta)
        gamma = TestFunction(Vtheta)

        self.theta = Function(Vtheta)
        theta_eqn = gamma*(theta - theta_in +
                           dot(k, self.u_hdiv)*dot(k, grad(thetabar))*beta)*dx

        theta_problem = LinearVariationalProblem(lhs(theta_eqn), rhs(theta_eqn), self.theta)
        self.theta_solver = LinearVariationalSolver(theta_problem,
                                                    solver_parameters={'ksp_type': 'cg',
                                                                       'pc_type': 'bjacobi',
                                                                       'pc_sub_type': 'ilu'},
                                                    options_prefix='thetabacksubstitution')

        self.bcs = [DirichletBC(Vu, 0.0, "bottom"),
                    DirichletBC(Vu, 0.0, "top")]

    @timed_function("Gusto:LinearSolve")
    def solve(self):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """

        # Solve the velocity-density system
        with timed_region("Gusto:VelocityDensitySolve"):

            # Assemble the RHS for lambda into self.R
            with timed_region("Gusto:HybridRHS"):
                self._assemble_Rexp()

            # Solve for lambda
            with timed_region("Gusto:HybridTraceSolve"):
                self.lSolver.solve(self.lambdar, self.R)

            # Reconstruct broken u and rho
            with timed_region("Gusto:HybridRecon"):
                self._assemble_rho()
                self._assemble_u()

            broken_u, rho1 = self.urho.split()
            u1 = self.u_hdiv

            # Project broken_u into the HDiv space
            u1.assign(0.0)

            with timed_region("Gusto:HybridProjectHDiv"):
                par_loop(self._average_kernel, dx,
                         {"w": (self._weight, READ),
                          "vec_in": (broken_u, READ),
                          "vec_out": (u1, INC)})

            # Reapply bcs to ensure they are satisfied
            for bc in self.bcs:
                bc.apply(u1)

        # Copy back into u and rho cpts of dy
        u, rho, theta = self.state.dy.split()
        u.assign(u1)
        rho.assign(rho1)

        # Reconstruct theta
        with timed_region("Gusto:ThetaRecon"):
            self.theta_solver.solve()

        # Copy into theta cpt of dy
        theta.assign(self.theta)


class IncompressibleSolver(TimesteppingSolver):
    """Timestepping linear solver object for the incompressible
    Boussinesq equations with prognostic variables u, p, b.

    This solver follows the following strategy:
    (1) Analytically eliminate b (introduces error near topography)
    (2) Solve resulting system for (u,p) using a hybrid-mixed method
    (3) Reconstruct b

    :arg state: a :class:`.State` object containing everything else.
    :arg solver_parameters: (optional) Solver parameters.
    :arg overwrite_solver_parameters: boolean, if True use only the
         solver_parameters that have been passed in, if False then
         update the default solver parameters with the solver_parameters
         passed in.
    """

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
                                        'sub_pc_type': 'ilu'}}
    }

    def __init__(self, state, solver_parameters=None,
                 overwrite_solver_parameters=False):
        super().__init__(state, solver_parameters, overwrite_solver_parameters)

    @timed_function("Gusto:SolverSetup")
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

        # Solver for u, p
        up_problem = LinearVariationalProblem(aeqn, Leqn, self.up, bcs=bcs)

        # Provide callback for the nullspace of the trace system
        def trace_nullsp(T):
            return VectorSpaceBasis(constant=True)

        appctx = {"trace_nullspace": trace_nullsp}
        self.up_solver = LinearVariationalSolver(up_problem,
                                                 solver_parameters=self.solver_parameters,
                                                 appctx=appctx)

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

    @timed_function("Gusto:LinearSolve")
    def solve(self):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """

        with timed_region("Gusto:VelocityPressureSolve"):
            self.up_solver.solve()

        u1, p1 = self.up.split()
        u, p, b = self.state.dy.split()
        u.assign(u1)
        p.assign(p1)

        with timed_region("Gusto:BuoyancyRecon"):
            self.b_solver.solve()

        b.assign(self.b)


class ShallowWaterSolver(TimesteppingSolver):
    """
    Timestepping linear solver object for the nonlinear shallow water
    equations with prognostic variables u and D. The linearized system
    is solved using a hybridized-mixed method.
    """

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
                                        'sub_pc_type': 'ilu'}}
    }

    @timed_function("Gusto:SolverSetup")
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
                                                 solver_parameters=self.solver_parameters,
                                                 options_prefix='SWimplicit')

    @timed_function("Gusto:LinearSolve")
    def solve(self):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """

        self.uD_solver.solve()
