"""
This module provides abstract linear solver objects.

The linear solvers provided here are used for solving linear problems on mixed
finite element spaces.
"""

from firedrake import (split, LinearVariationalProblem, Constant,
                       LinearVariationalSolver, TestFunctions, TrialFunctions,
                       TestFunction, TrialFunction, lhs, rhs, FacetNormal,
                       div, dx, jump, avg, dS_v, dS_h, ds_v, ds_t, ds_b, ds_tb, inner, action,
                       dot, grad, Function, VectorSpaceBasis, BrokenElement,
                       FunctionSpace, MixedFunctionSpace, DirichletBC)
from firedrake.petsc import flatten_parameters
from pyop2.profiling import timed_function, timed_region

from gusto.active_tracers import TracerVariableType
from gusto.logging import logger, DEBUG, logging_ksp_monitor_true_residual
from gusto.labels import linearisation, time_derivative, hydrostatic
from gusto import thermodynamics
from gusto.fml.form_manipulation_language import Term, drop
from gusto.recovery.recovery_kernels import AverageWeightings, AverageKernel
from abc import ABCMeta, abstractmethod, abstractproperty


__all__ = ["IncompressibleSolver", "LinearTimesteppingSolver", "CompressibleSolver"]


class TimesteppingSolver(object, metaclass=ABCMeta):
    """Base class for timestepping linear solvers for Gusto."""

    def __init__(self, equations, alpha=0.5, solver_parameters=None,
                 overwrite_solver_parameters=False):
        """
        Args:
            equations (:class:`PrognosticEquation`): the model's equation.
            alpha (float, optional): the semi-implicit off-centring factor.
                Defaults to 0.5. A value of 1 is fully-implicit.
            solver_parameters (dict, optional): contains the options to be
                passed to the underlying :class:`LinearVariationalSolver`.
                Defaults to None.
            overwrite_solver_parameters (bool, optional): if True use only the
                `solver_parameters` that have been passed in. If False then
                update the default parameters with the `solver_parameters`
                passed in. Defaults to False.
        """
        self.equations = equations
        self.dt = equations.domain.dt
        self.alpha = alpha

        if solver_parameters is not None:
            if not overwrite_solver_parameters:
                p = flatten_parameters(self.solver_parameters)
                p.update(flatten_parameters(solver_parameters))
                solver_parameters = p
            self.solver_parameters = solver_parameters

        # ~ if logger.isEnabledFor(DEBUG):
            # ~ self.solver_parameters["ksp_monitor_true_residual"] = None

        # setup the solver
        self._setup_solver()

    @staticmethod
    def log_ksp_residuals(ksp):
        if logger.isEnabledFor(DEBUG):
            ksp.setMonitor(logging_ksp_monitor_true_residual)

    @abstractproperty
    def solver_parameters(self):
        """Solver parameters for this solver"""
        pass

    @abstractmethod
    def _setup_solver(self):
        pass

    @abstractmethod
    def solve(self):
        pass


class CompressibleSolver(TimesteppingSolver):
    """
    Timestepping linear solver object for the compressible Euler equations.

    This solves a linear problem for the compressible Euler equations in
    theta-exner formulation with prognostic variables u (velocity), rho
    (density) and theta (potential temperature). It follows the following
    strategy:

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
    """

    solver_parameters = {'mat_type': 'matfree',
                         'ksp_type': 'preonly',
                         'pc_type': 'python',
                         'pc_python_type': 'firedrake.SCPC',
                         'pc_sc_eliminate_fields': '0, 1',
                         # The reduced operator is not symmetric
                         'condensed_field': {'ksp_type': 'fgmres',
                                             'ksp_rtol': 1.0e-8,
                                             'ksp_atol': 1.0e-8,
                                             'ksp_max_it': 100,
                                             'pc_type': 'gamg',
                                             'pc_gamg_sym_graph': None,
                                             'mg_levels': {'ksp_type': 'gmres',
                                                           'ksp_max_it': 5,
                                                           'pc_type': 'bjacobi',
                                                           'sub_pc_type': 'ilu'}}}

    def __init__(self, equations, alpha=0.5,
                 quadrature_degree=None, solver_parameters=None,
                 overwrite_solver_parameters=False):
        """
        Args:
            equations (:class:`PrognosticEquation`): the model's equation.
            alpha (float, optional): the semi-implicit off-centring factor.
                Defaults to 0.5. A value of 1 is fully-implicit.
            quadrature_degree (tuple, optional): a tuple (q_h, q_v) where q_h is
                the required quadrature degree in the horizontal direction and
                q_v is that in the vertical direction. Defaults to None.
            solver_parameters (dict, optional): contains the options to be
                passed to the underlying :class:`LinearVariationalSolver`.
                Defaults to None.
            overwrite_solver_parameters (bool, optional): if True use only the
                `solver_parameters` that have been passed in. If False then
                update the default parameters with the `solver_parameters`
                passed in. Defaults to False.
        """
        self.equations = equations

        if quadrature_degree is not None:
            self.quadrature_degree = quadrature_degree
        else:
            dgspace = equations.domain.spaces("DG")
            if any(deg > 2 for deg in dgspace.ufl_element().degree()):
                logger.warning("default quadrature degree most likely not sufficient for this degree element")
            self.quadrature_degree = (5, 5)

        super().__init__(equations, alpha, solver_parameters,
                         overwrite_solver_parameters)

    @timed_function("Gusto:SolverSetup")
    def _setup_solver(self):

        equations = self.equations
        dt = self.dt
        beta_ = dt*self.alpha
        cp = equations.parameters.cp
        Vu = equations.domain.spaces("HDiv")
        Vu_broken = FunctionSpace(equations.domain.mesh, BrokenElement(Vu.ufl_element()))
        Vtheta = equations.domain.spaces("theta")
        Vrho = equations.domain.spaces("DG")

        # Store time-stepping coefficients as UFL Constants
        beta = Constant(beta_)
        beta_cp = Constant(beta_ * cp)

        h_deg = Vrho.ufl_element().degree()[0]
        v_deg = Vrho.ufl_element().degree()[1]
        Vtrace = FunctionSpace(equations.domain.mesh, "HDiv Trace", degree=(h_deg, v_deg))

        # Split up the rhs vector (symbolically)
        self.xrhs = Function(self.equations.function_space)
        u_in, rho_in, theta_in = split(self.xrhs)[0:3]

        # Build the function space for "broken" u, rho, and pressure trace
        M = MixedFunctionSpace((Vu_broken, Vrho, Vtrace))
        w, phi, dl = TestFunctions(M)
        u, rho, l0 = TrialFunctions(M)

        n = FacetNormal(equations.domain.mesh)

        # Get background fields
        _, rhobar, thetabar = split(equations.X_ref)[0:3]
        exnerbar = thermodynamics.exner_pressure(equations.parameters, rhobar, thetabar)
        exnerbar_rho = thermodynamics.dexner_drho(equations.parameters, rhobar, thetabar)
        exnerbar_theta = thermodynamics.dexner_dtheta(equations.parameters, rhobar, thetabar)

        # Analytical (approximate) elimination of theta
        k = equations.domain.k             # Upward pointing unit vector
        theta = -dot(k, u)*dot(k, grad(thetabar))*beta + theta_in

        # Only include theta' (rather than exner') in the vertical
        # component of the gradient

        # The exner prime term (here, bars are for mean and no bars are
        # for linear perturbations)
        exner = exnerbar_theta*theta + exnerbar_rho*rho

        # Vertical projection
        def V(u):
            return k*inner(u, k)

        # hydrostatic projection
        h_project = lambda u: u - k*inner(u, k)

        # Specify degree for some terms as estimated degree is too large
        dxp = dx(degree=(self.quadrature_degree))
        dS_vp = dS_v(degree=(self.quadrature_degree))
        dS_hp = dS_h(degree=(self.quadrature_degree))
        ds_vp = ds_v(degree=(self.quadrature_degree))
        ds_tbp = (ds_t(degree=(self.quadrature_degree))
                  + ds_b(degree=(self.quadrature_degree)))

        # Add effect of density of water upon theta, using moisture reference profiles
        # TODO: Explore if this is the right thing to do for the linear problem
        if equations.active_tracers is not None:
            mr_t = Constant(0.0)*thetabar
            for tracer in equations.active_tracers:
                if tracer.chemical == 'H2O':
                    if tracer.variable_type == TracerVariableType.mixing_ratio:
                        idx = equations.field_names.index(tracer.name)
                        mr_bar = split(equations.X_ref)[idx]
                        mr_t += mr_bar
                    else:
                        raise NotImplementedError('Only mixing ratio tracers are implemented')

            theta_w = theta / (1 + mr_t)
            thetabar_w = thetabar / (1 + mr_t)
        else:
            theta_w = theta
            thetabar_w = thetabar

        _l0 = TrialFunction(Vtrace)
        _dl = TestFunction(Vtrace)
        a_tr = _dl('+')*_l0('+')*(dS_vp + dS_hp) + _dl*_l0*ds_vp + _dl*_l0*ds_tbp

        def L_tr(f):
            return _dl('+')*avg(f)*(dS_vp + dS_hp) + _dl*f*ds_vp + _dl*f*ds_tbp

        cg_ilu_parameters = {'ksp_type': 'cg',
                             'pc_type': 'bjacobi',
                             'sub_pc_type': 'ilu'}

        # Project field averages into functions on the trace space
        rhobar_avg = Function(Vtrace)
        exnerbar_avg = Function(Vtrace)

        rho_avg_prb = LinearVariationalProblem(a_tr, L_tr(rhobar), rhobar_avg)
        exner_avg_prb = LinearVariationalProblem(a_tr, L_tr(exnerbar), exnerbar_avg)

        self.rho_avg_solver = LinearVariationalSolver(rho_avg_prb,
                                                      solver_parameters=cg_ilu_parameters,
                                                      options_prefix='rhobar_avg_solver')
        self.exner_avg_solver = LinearVariationalSolver(exner_avg_prb,
                                                        solver_parameters=cg_ilu_parameters,
                                                        options_prefix='exnerbar_avg_solver')

        # "broken" u, rho, and trace system
        # NOTE: no ds_v integrals since equations are defined on
        # a periodic (or sphere) base mesh.
        if any([t.has_label(hydrostatic) for t in self.equations.residual]):
            u_mass = inner(w, (h_project(u) - u_in))*dx
        else:
            u_mass = inner(w, (u - u_in))*dx

        eqn = (
            # momentum equation
            u_mass
            - beta_cp*div(theta_w*V(w))*exnerbar*dxp
            # following does nothing but is preserved in the comments
            # to remind us why (because V(w) is purely vertical).
            # + beta_cp*jump(theta_w*V(w), n=n)*exnerbar_avg('+')*dS_vp
            + beta_cp*jump(theta_w*V(w), n=n)*exnerbar_avg('+')*dS_hp
            + beta_cp*dot(theta_w*V(w), n)*exnerbar_avg*ds_tbp
            - beta_cp*div(thetabar_w*w)*exner*dxp
            # trace terms appearing after integrating momentum equation
            + beta_cp*jump(thetabar_w*w, n=n)*l0('+')*(dS_vp + dS_hp)
            + beta_cp*dot(thetabar_w*w, n)*l0*(ds_tbp + ds_vp)
            # mass continuity equation
            + (phi*(rho - rho_in) - beta*inner(grad(phi), u)*rhobar)*dx
            + beta*jump(phi*u, n=n)*rhobar_avg('+')*(dS_v + dS_h)
            # term added because u.n=0 is enforced weakly via the traces
            + beta*phi*dot(u, n)*rhobar_avg*(ds_tb + ds_v)
            # constraint equation to enforce continuity of the velocity
            # through the interior facets and weakly impose the no-slip
            # condition
            + dl('+')*jump(u, n=n)*(dS_vp + dS_hp)
            + dl*dot(u, n)*(ds_tbp + ds_vp)
        )

        # TODO: can we get this term using FML?
        # contribution of the sponge term
        if hasattr(self.equations, "mu"):
            eqn += dt*self.equations.mu*inner(w, k)*inner(u, k)*dx

        aeqn = lhs(eqn)
        Leqn = rhs(eqn)

        # Function for the hybridized solutions
        self.urhol0 = Function(M)

        hybridized_prb = LinearVariationalProblem(aeqn, Leqn, self.urhol0)
        hybridized_solver = LinearVariationalSolver(hybridized_prb,
                                                    solver_parameters=self.solver_parameters,
                                                    options_prefix='ImplicitSolver')
        self.hybridized_solver = hybridized_solver

        # Project broken u into the HDiv space using facet averaging.
        # Weight function counting the dofs of the HDiv element:
        self._weight = Function(Vu)
        weight_kernel = AverageWeightings(Vu)
        weight_kernel.apply(self._weight)

        # Averaging kernel
        self._average_kernel = AverageKernel(Vu)

        # HDiv-conforming velocity
        self.u_hdiv = Function(Vu)

        # Reconstruction of theta
        theta = TrialFunction(Vtheta)
        gamma = TestFunction(Vtheta)

        self.theta = Function(Vtheta)
        theta_eqn = gamma*(theta - theta_in
                           + dot(k, self.u_hdiv)*dot(k, grad(thetabar))*beta)*dx

        theta_problem = LinearVariationalProblem(lhs(theta_eqn), rhs(theta_eqn), self.theta)
        self.theta_solver = LinearVariationalSolver(theta_problem,
                                                    solver_parameters=cg_ilu_parameters,
                                                    options_prefix='thetabacksubstitution')

        # Store boundary conditions for the div-conforming velocity to apply
        # post-solve
        self.bcs = self.equations.bcs['u']

        # Log residuals on hybridized solver
        self.log_ksp_residuals(self.hybridized_solver.snes.ksp)
        # Log residuals on the trace system too
        from gusto.logging import attach_custom_monitor
        python_context = self.hybridized_solver.snes.ksp.pc.getPythonContext()
        attach_custom_monitor(python_context, logging_ksp_monitor_true_residual)

    @timed_function("Gusto:LinearSolve")
    def solve(self, xrhs, dy):
        """
        Solve the linear problem.

        Args:
            xrhs (:class:`Function`): the right-hand side field in the
                appropriate :class:`MixedFunctionSpace`.
            dy (:class:`Function`): the resulting field in the appropriate
                :class:`MixedFunctionSpace`.
        """
        self.xrhs.assign(xrhs)

        # TODO: can we avoid computing these each time the solver is called?
        with timed_region("Gusto:HybridProjectRhobar"):
            self.rho_avg_solver.solve()

        with timed_region("Gusto:HybridProjectExnerbar"):
            self.exner_avg_solver.solve()

        # Solve the hybridized system
        self.hybridized_solver.solve()

        broken_u, rho1, _ = self.urhol0.subfunctions
        u1 = self.u_hdiv

        # Project broken_u into the HDiv space
        u1.assign(0.0)

        with timed_region("Gusto:HybridProjectHDiv"):
            self._average_kernel.apply(u1, self._weight, broken_u)

        # Reapply bcs to ensure they are satisfied
        for bc in self.bcs:
            bc.apply(u1)

        # Copy back into u and rho cpts of dy
        u, rho, theta = dy.subfunctions[0:3]
        u.assign(u1)
        rho.assign(rho1)

        # Reconstruct theta
        with timed_region("Gusto:ThetaRecon"):
            self.theta_solver.solve()

        # Copy into theta cpt of dy
        theta.assign(self.theta)


class IncompressibleSolver(TimesteppingSolver):
    """
    Linear solver object for the incompressible Boussinesq equations.

    This solves a linear problem for the incompressible Boussinesq equations
    with prognostic variables u (velocity), p (pressure) and b (buoyancy). It
    follows the following strategy:

    This solver follows the following strategy:
    (1) Analytically eliminate b (introduces error near topography)
    (2) Solve resulting system for (u,p) using a hybrid-mixed method
    (3) Reconstruct b
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
        equation = self.equations      # just cutting down line length a bit
        dt = self.dt
        beta_ = dt*self.alpha
        Vu = equation.domain.spaces("HDiv")
        Vb = equation.domain.spaces("theta")
        Vp = equation.domain.spaces("DG")

        # Store time-stepping coefficients as UFL Constants
        beta = Constant(beta_)

        # Split up the rhs vector (symbolically)
        self.xrhs = Function(self.equations.function_space)
        u_in, p_in, b_in = split(self.xrhs)

        # Build the reduced function space for u,p
        M = MixedFunctionSpace((Vu, Vp))
        w, phi = TestFunctions(M)
        u, p = TrialFunctions(M)

        # Get background fields
        bbar = split(equation.X_ref)[2]

        # Analytical (approximate) elimination of theta
        k = equation.domain.k             # Upward pointing unit vector
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

        if hasattr(self.equations, "mu"):
            eqn += dt*self.equations.mu*inner(w, k)*inner(u, k)*dx
        aeqn = lhs(eqn)
        Leqn = rhs(eqn)

        # Place to put result of u p solver
        self.up = Function(M)

        # Boundary conditions (assumes extruded mesh)
        # BCs are declared for the plain velocity space. As we need them in
        # a mixed problem, we replicate the BCs but for subspace of M
        bcs = [DirichletBC(M.sub(0), bc.function_arg, bc.sub_domain) for bc in self.equations.bcs['u']]

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

        u, p = self.up.subfunctions
        self.b = Function(Vb)

        b_eqn = gamma*(b - b_in
                       + dot(k, u)*dot(k, grad(bbar))*beta)*dx

        b_problem = LinearVariationalProblem(lhs(b_eqn),
                                             rhs(b_eqn),
                                             self.b)
        self.b_solver = LinearVariationalSolver(b_problem)

        # Log residuals on hybridized solver
        self.log_ksp_residuals(self.up_solver.snes.ksp)

    @timed_function("Gusto:LinearSolve")
    def solve(self, xrhs, dy):
        """
        Solve the linear problem.

        Args:
            xrhs (:class:`Function`): the right-hand side field in the
                appropriate :class:`MixedFunctionSpace`.
            dy (:class:`Function`): the resulting field in the appropriate
                :class:`MixedFunctionSpace`.
        """
        self.xrhs.assign(xrhs)

        with timed_region("Gusto:VelocityPressureSolve"):
            self.up_solver.solve()

        u1, p1 = self.up.subfunctions
        u, p, b = dy.subfunctions
        u.assign(u1)
        p.assign(p1)

        with timed_region("Gusto:BuoyancyRecon"):
            self.b_solver.solve()

        b.assign(self.b)


class LinearTimesteppingSolver(object):
    """
    A general object for solving mixed finite element linear problems.

    This linear solver object is general and is designed for use with different
    equation sets, including with the non-linear shallow-water equations. It
    forms the linear problem from the equations using FML. The linear system is
    solved using a hybridised-mixed method.
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

    def __init__(self, equation, alpha):
        """
        Args:
            equation (:class:`PrognosticEquation`): the model's equation object.
            alpha (float): the semi-implicit off-centring factor. A value of 1
                is fully-implicit.
        """
        residual = equation.residual.label_map(
            lambda t: t.has_label(linearisation),
            lambda t: Term(t.get(linearisation).form, t.labels),
            drop)

        dt = equation.domain.dt
        W = equation.function_space
        beta = dt*alpha

        # Split up the rhs vector (symbolically)
        self.xrhs = Function(W)

        aeqn = residual.label_map(
            lambda t: (t.has_label(time_derivative) and t.has_label(linearisation)),
            map_if_false=lambda t: beta*t)
        Leqn = residual.label_map(
            lambda t: (t.has_label(time_derivative) and t.has_label(linearisation)),
            map_if_false=drop)

        # Place to put result of solver
        self.dy = Function(W)

        # Solver
        bcs = [DirichletBC(W.sub(0), bc.function_arg, bc.sub_domain) for bc in equation.bcs['u']]
        problem = LinearVariationalProblem(aeqn.form,
                                           action(Leqn.form, self.xrhs),
                                           self.dy, bcs=bcs)

        self.solver = LinearVariationalSolver(problem,
                                              solver_parameters=self.solver_parameters,
                                              options_prefix='linear_solver')

    @timed_function("Gusto:LinearSolve")
    def solve(self, xrhs, dy):
        """
        Solve the linear problem.

        Args:
            xrhs (:class:`Function`): the right-hand side field in the
                appropriate :class:`MixedFunctionSpace`.
            dy (:class:`Function`): the resulting field in the appropriate
                :class:`MixedFunctionSpace`.
        """
        self.xrhs.assign(xrhs)
        self.solver.solve()
        dy.assign(self.dy)
