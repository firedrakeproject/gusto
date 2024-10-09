"""
This module provides abstract linear solver objects.

The linear solvers provided here are used for solving linear problems on mixed
finite element spaces.
"""

from firedrake import (
    split, LinearVariationalProblem, Constant, LinearVariationalSolver,
    TestFunctions, TrialFunctions, TestFunction, TrialFunction, lhs,
    rhs, FacetNormal, div, dx, jump, avg, dS, dS_v, dS_h, ds_v, ds_t, ds_b,
    ds_tb, inner, action, dot, grad, Function, VectorSpaceBasis,
    BrokenElement, FunctionSpace, MixedFunctionSpace, DirichletBC,
    conditional, Interpolator, max_value, min_value
)
from firedrake.fml import Term, drop
from firedrake.petsc import flatten_parameters
from pyop2.profiling import timed_function, timed_region

from gusto.active_tracers import TracerVariableType
from gusto.logging import logger, DEBUG, logging_ksp_monitor_true_residual
from gusto.labels import linearisation, time_derivative, hydrostatic
from gusto import thermodynamics
from gusto.recovery.recovery_kernels import AverageWeightings, AverageKernel
from gusto.saturation_function import compute_saturation
from abc import ABCMeta, abstractmethod, abstractproperty


__all__ = ["BoussinesqSolver", "LinearTimesteppingSolver", "CompressibleSolver", "ThermalSWSolver", "MoistConvectiveSWSolver", "MoistDynamicsSWSolver", "MoistThermalSWSolver"]


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
    def solve(self, xrhs, dy, xn):
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
            logger.info('Compressible linear solver: rho average solve')
            self.rho_avg_solver.solve()

        with timed_region("Gusto:HybridProjectExnerbar"):
            logger.info('Compressible linear solver: Exner average solve')
            self.exner_avg_solver.solve()

        # Solve the hybridized system
        logger.info('Compressible linear solver: hybridized solve')
        self.hybridized_solver.solve()

        broken_u, rho1, _ = self.urhol0.subfunctions
        u1 = self.u_hdiv

        # Project broken_u into the HDiv space
        u1.assign(0.0)

        with timed_region("Gusto:HybridProjectHDiv"):
            logger.info('Compressible linear solver: restore continuity')
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
            logger.info('Compressible linear solver: theta solve')
            self.theta_solver.solve()

        # Copy into theta cpt of dy
        theta.assign(self.theta)


class BoussinesqSolver(TimesteppingSolver):
    """
    Linear solver object for the Boussinesq equations.

    This solves a linear problem for the Boussinesq equations
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
        )

        if equation.compressible:
            cs = equation.parameters.cs
            eqn += phi * (p - p_in) * dx + beta * phi * cs**2 * div(u) * dx
        else:
            eqn += phi * div(u) * dx

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
    def solve(self, xrhs, dy, xn):
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
            logger.info('Boussinesq linear solver: mixed solve')
            self.up_solver.solve()

        u1, p1 = self.up.subfunctions
        u, p, b = dy.subfunctions
        u.assign(u1)
        p.assign(p1)

        with timed_region("Gusto:BuoyancyRecon"):
            logger.info('Boussinesq linear solver: buoyancy reconstruction')
            self.b_solver.solve()

        b.assign(self.b)


class ThermalSWSolver(TimesteppingSolver):
    """
    Linear solver object for the thermal shallow water equations.

    This solves a linear problem for the thermal shallow water equations with
    prognostic variables u (velocity), D (depth) and b (buoyancy). It follows
    the following strategy:

    (1) Eliminate b
    (2) Solve the resulting system for (u, D) using a hybrid-mixed method
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
        VD = equation.domain.spaces("DG")
        Vb = equation.domain.spaces("DG")

        # Check that the third field is buoyancy
        if not equation.field_names[2] == 'b':
            raise NotImplementedError("Field 'b' must exist to use the thermal linear solver in the SIQN scheme")

        # Store time-stepping coefficients as UFL Constants
        beta = Constant(beta_)

        # Split up the rhs vector
        self.xrhs = Function(self.equations.function_space)
        u_in = split(self.xrhs)[0]
        D_in = split(self.xrhs)[1]
        b_in = split(self.xrhs)[2]

        # Build the reduced function space for u, D
        M = MixedFunctionSpace((Vu, VD))
        w, phi = TestFunctions(M)
        u, D = TrialFunctions(M)

        # Get background buoyancy and depth
        Dbar = split(equation.X_ref)[1]
        bbar = split(equation.X_ref)[2]

        # Approximate elimination of b
        b = -dot(u, grad(bbar))*beta + b_in

        n = FacetNormal(equation.domain.mesh)

        eqn = (
            inner(w, (u - u_in)) * dx
            - beta * (D - Dbar) * div(w*bbar) * dx
            + beta * jump(w*bbar, n) * avg(D-Dbar) * dS
            - beta * 0.5 * Dbar * bbar * div(w) * dx
            - beta * 0.5 * Dbar * b * div(w) * dx
            - beta * 0.5 * bbar * div(w*(D-Dbar)) * dx
            + beta * 0.5 * jump((D-Dbar)*w, n) * avg(bbar) * dS
            + inner(phi, (D - D_in)) * dx
            + beta * phi * Dbar * div(u) * dx
        )

        if 'coriolis' in equation.prescribed_fields._field_names:
            f = equation.prescribed_fields('coriolis')
            eqn += beta * f * inner(w, equation.domain.perp(u)) * dx

        aeqn = lhs(eqn)
        Leqn = rhs(eqn)

        # Place to put results of (u,D) solver
        self.uD = Function(M)

        # Boundary conditions
        bcs = [DirichletBC(M.sub(0), bc.function_arg, bc.sub_domain) for bc in self.equations.bcs['u']]

        # Solver for u, D
        uD_problem = LinearVariationalProblem(aeqn, Leqn, self.uD, bcs=bcs)

        # Provide callback for the nullspace of the trace system
        def trace_nullsp(T):
            return VectorSpaceBasis(constant=True)

        appctx = {"trace_nullspace": trace_nullsp}
        self.uD_solver = LinearVariationalSolver(uD_problem,
                                                 solver_parameters=self.solver_parameters,
                                                 appctx=appctx)
        # Reconstruction of b
        b = TrialFunction(Vb)
        gamma = TestFunction(Vb)

        u, D = self.uD.subfunctions
        self.b = Function(Vb)

        b_eqn = gamma*(b - b_in + inner(u, grad(bbar))*beta) * dx

        b_problem = LinearVariationalProblem(lhs(b_eqn),
                                             rhs(b_eqn),
                                             self.b)
        self.b_solver = LinearVariationalSolver(b_problem)

        # Log residuals on hybridized solver
        self.log_ksp_residuals(self.uD_solver.snes.ksp)

    @timed_function("Gusto:LinearSolve")
    def solve(self, xrhs, dy, xn):
        """
        Solve the linear problem.

        Args:
            xrhs (:class:`Function`): the right-hand side field in the
                appropriate :class:`MixedFunctionSpace`.
            dy (:class:`Function`): the resulting field in the appropriate
                :class:`MixedFunctionSpace`.
        """
        self.xrhs.assign(xrhs)

        # Check that the b reference profile has been set
        bbar = split(self.equations.X_ref)[2]
        b = dy.subfunctions[2]
        bbar_func = Function(b.function_space()).interpolate(bbar)
        if bbar_func.dat.data.max() == 0 and bbar_func.dat.data.min() == 0:
            logger.warning("The reference profile for b in the linear solver is zero. To set a non-zero profile add b to the set_reference_profiles argument.")

        with timed_region("Gusto:VelocityDepthSolve"):
            logger.info('Thermal linear solver: mixed solve')
            self.uD_solver.solve()

        u1, D1 = self.uD.subfunctions
        u = dy.subfunctions[0]
        D = dy.subfunctions[1]
        b = dy.subfunctions[2]
        u.assign(u1)
        D.assign(D1)

        with timed_region("Gusto:BuoyancyRecon"):
            logger.info('Thermal linear solver: buoyancy reconstruction')
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
    def solve(self, xrhs, dy, xn):
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


class MoistConvectiveSWSolver(TimesteppingSolver):
    """
    Linear solver for the moist convective shallow water equations.

    This solves a linear problem for the shallow water equations with prognostic
    variables u (velocity) and D (depth). The linear system is solved using a
    hybridised-mixed method.
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
        VD = equation.domain.spaces("DG")

        # Store time-stepping coefficients as UFL Constants
        beta = Constant(beta_)

        # Split up the rhs vector
        self.xrhs = Function(self.equations.function_space)
        u_in = split(self.xrhs)[0]
        D_in = split(self.xrhs)[1]

        # Build the reduced function space for u, D
        M = MixedFunctionSpace((Vu, VD))
        w, phi = TestFunctions(M)
        u, D = TrialFunctions(M)

        # Get background depth
        Dbar = split(equation.X_ref)[1]

        g = equation.parameters.g

        eqn = (
            inner(w, (u - u_in)) * dx
            - beta * (D - Dbar) * div(w*g) * dx
            + inner(phi, (D - D_in)) * dx
            + beta * phi * Dbar * div(u) * dx
        )

        if 'coriolis' in equation.prescribed_fields._field_names:
            f = equation.prescribed_fields('coriolis')
            eqn += beta * f * inner(w, equation.domain.perp(u)) * dx

        aeqn = lhs(eqn)
        Leqn = rhs(eqn)

        # Place to put results of (u,D) solver
        self.uD = Function(M)

        # Boundary conditions
        bcs = [DirichletBC(M.sub(0), bc.function_arg, bc.sub_domain) for bc in self.equations.bcs['u']]

        # Solver for u, D
        uD_problem = LinearVariationalProblem(aeqn, Leqn, self.uD, bcs=bcs)

        # Provide callback for the nullspace of the trace system
        def trace_nullsp(T):
            return VectorSpaceBasis(constant=True)

        appctx = {"trace_nullspace": trace_nullsp}
        self.uD_solver = LinearVariationalSolver(uD_problem,
                                                 solver_parameters=self.solver_parameters,
                                                 appctx=appctx)

        # Log residuals on hybridized solver
        self.log_ksp_residuals(self.uD_solver.snes.ksp)

    @timed_function("Gusto:LinearSolve")
    def solve(self, xrhs, dy, xn):
        """
        Solve the linear problem.

        Args:
            xrhs (:class:`Function`): the right-hand side field in the
                appropriate :class:`MixedFunctionSpace`.
            dy (:class:`Function`): the resulting field in the appropriate
                :class:`MixedFunctionSpace`.
        """
        self.xrhs.assign(xrhs)

        with timed_region("Gusto:VelocityDepthSolve"):
            logger.info('Moist convective linear solver: mixed solve')
            self.uD_solver.solve()

        u1, D1 = self.uD.subfunctions
        u = dy.subfunctions[0]
        D = dy.subfunctions[1]
        u.assign(u1)
        D.assign(D1)


class MoistDynamicsSWSolver(TimesteppingSolver):
    """
    Linear solver object for the moist shallow water equations, written with
    all moisture in the dynamics.

    This solves a linear problem for the moist shallow water equations with
    prognostic variables u (velocity), D (depth) and b_e (equivalent buoyancy).
    It follows the following strategy:

    (1) Eliminate b_e
    (2) Solve the resulting system for (u, D) using a hybrid-mixed method
    (3) Reconstruct b_e
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
        VD = equation.domain.spaces("DG")
        Vb = equation.domain.spaces("DG")

        # Check that the third field is buoyancy
        if not equation.field_names[2] == 'b_e':
            raise NotImplementedError("Field 'b_e' must exist to use the moist dynamics linear solver in the SIQN scheme")
        if not equation.field_names[3] == 'q_t':
            raise NotImplementedError("Field 'q_t' must exist to use the moist dynamics linear solver in the SIQN scheme")

        # Store time-stepping coefficients as UFL Constants
        beta = Constant(beta_)

        # Split up the rhs vector
        self.xrhs = Function(self.equations.function_space)
        u_in = split(self.xrhs)[0]
        D_in = split(self.xrhs)[1]
        b_e_in = split(self.xrhs)[2]

        # Split up the xn vector
        self.xn = Function(self.equations.function_space)
        D_xn = split(self.xn)[1]
        b_e_xn = split(self.xn)[2]
        q_t_xn = split(self.xn)[3]

        # Build the reduced function space for u, D
        M = MixedFunctionSpace((Vu, VD))
        w, phi = TestFunctions(M)
        u, D = TrialFunctions(M)

        # Get background buoyancy and depth
        Dbar = split(equation.X_ref)[1]
        b_ebar = split(equation.X_ref)[2]

        # Approximate elimination of b_e
        b_e = -dot(u, grad(b_ebar))*beta + b_e_in

        n = FacetNormal(equation.domain.mesh)

        # compute q_v using q_sat to partition q_t into q_v and q_c
        g = equation.parameters.g
        H = equation.parameters.H
        q0 = equation.q0
        beta2 = equation.beta2
        nu = equation.nu

        # check for topography
        if hasattr(equation.prescribed_fields, "topography"):
            B = equation.prescribed_fields("topography")
        else:
            B = None

        self.q_sat_func = Function(VD)
        self.q_v_bar = Function(VD)

        # set up interpolators that use the xn values for D and b_e
        self.q_sat_expr_interpolator = Interpolator(compute_saturation(q0, nu, H, g, D_xn, b_e_xn, B), VD)
        self.q_v_interpolator = Interpolator(conditional(q_t_xn < self.q_sat_func, q_t_xn, self.q_sat_func), VD)

        q_v_bar = self.q_v_bar  # to make line length shorter
        eqn = (
            inner(w, (u - u_in)) * dx
            - beta * (D-Dbar) * div(w * (b_ebar + beta2*q_v_bar)) * dx
            + beta * jump(w*(b_ebar + beta2*q_v_bar), n) * avg((D-Dbar)) * dS
            - beta * 0.5 * H * (b_ebar + b_e + beta2*q_v_bar) * div(w) * dx
            - beta * 0.5 * (b_ebar + beta2*q_v_bar) * div(w*(D-Dbar)) * dx
            + beta * 0.5 * jump((D-Dbar)*w, n) * avg(b_ebar + beta2*q_v_bar) * dS
            + inner(phi, (D - D_in)) * dx
            + beta * phi * H * div(u) * dx
        )

        if 'coriolis' in equation.prescribed_fields._field_names:
            f = equation.prescribed_fields('coriolis')
            eqn += beta * f * inner(w, equation.domain.perp(u)) * dx

        aeqn = lhs(eqn)
        Leqn = rhs(eqn)

        # Place to put results of (u,D) solver
        self.uD = Function(M)

        # Boundary conditions
        bcs = [DirichletBC(M.sub(0), bc.function_arg, bc.sub_domain) for bc in self.equations.bcs['u']]

        # Solver for u, D
        uD_problem = LinearVariationalProblem(aeqn, Leqn, self.uD, bcs=bcs)

        # Provide callback for the nullspace of the trace system
        def trace_nullsp(T):
            return VectorSpaceBasis(constant=True)

        appctx = {"trace_nullspace": trace_nullsp}
        self.uD_solver = LinearVariationalSolver(uD_problem,
                                                 solver_parameters=self.solver_parameters,
                                                 appctx=appctx)
        # Reconstruction of b_e
        b_e = TrialFunction(Vb)
        gamma = TestFunction(Vb)

        u, D = self.uD.subfunctions
        self.b_e = Function(Vb)

        b_e_eqn = gamma*(b_e - b_e_in + inner(u, grad(b_ebar))*beta) * dx

        b_e_problem = LinearVariationalProblem(lhs(b_e_eqn),
                                               rhs(b_e_eqn),
                                               self.b_e)
        self.b_e_solver = LinearVariationalSolver(b_e_problem)

        # Log residuals on hybridized solver
        self.log_ksp_residuals(self.uD_solver.snes.ksp)

    @timed_function("Gusto:LinearSolve")
    def solve(self, xrhs, dy, xn):
        """
        Solve the linear problem.

        Args:
            xrhs (:class:`Function`): the right-hand side field in the
                appropriate :class:`MixedFunctionSpace`.
            dy (:class:`Function`): the resulting field in the appropriate
                :class:`MixedFunctionSpace`.
        """
        self.xrhs.assign(xrhs)
        self.xn.assign(xn)

        # Check that the b reference profile has been set
        b_ebar = split(self.equations.X_ref)[2]
        b_e = dy.subfunctions[2]

        b_ebar_func = Function(b_e.function_space()).interpolate(b_ebar)
        if b_ebar_func.dat.data.max() == 0 and b_ebar_func.dat.data.min() == 0:
            logger.warning("The reference profile for b_e in the linear solver is zero. To set a non-zero profile add b_e to the set_reference_profiles argument.")

        with timed_region("Gusto:VelocityDepthSolve"):
            logger.info('Moist dynamics linear solver: mixed solve')

            self.q_sat_func.assign(self.q_sat_expr_interpolator.interpolate())

            self.q_v_bar.assign(self.q_v_interpolator.interpolate())

            self.uD_solver.solve()

        u1, D1 = self.uD.subfunctions
        u = dy.subfunctions[0]
        D = dy.subfunctions[1]
        b_e = dy.subfunctions[2]
        u.assign(u1)
        D.assign(D1)

        with timed_region("Gusto:BuoyancyRecon"):
            logger.info('Moist dynamics linear solver: buoyancy reconstruction')
            self.b_e_solver.solve()

        b_e.assign(self.b_e)


class MoistThermalSWSolver(TimesteppingSolver):
    """
    Linear solver object for the moist thermal shallow water equations, where
    moisture is included in the linear solve.

    This solves a linear problem for the moist thermal shallow water equations
    with prognostic variables u (velocity), D (depth), b (buoyancy), qv (water
    vapour) and qc (cloud water).
     """

    solver_parameters = {
    "snes_type": "ksponly",
    "mat_type": "matfree",
    "ksp_type": "gmres",
    "ksp_converged_reason": None,
    "ksp_atol": 1e-5,
    "ksp_rtol": 1e-5,
    "ksp_max_it": 400,
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_star_sub_sub_pc_type": "lu",
    "assembled_pc_type": "python",
    "assembled_pc_python_type": "firedrake.ASMStarPC",
    "assembled_pc_star_construct_dim": 0,
    "assembled_pc_star_sub_sub_pc_factor_mat_ordering_type": "rcm",
    "assembled_pc_star_sub_sub_pc_factor_reuse_ordering": None,
    "assembled_pc_star_sub_sub_pc_factor_reuse_fill": None,
    "assembled_pc_star_sub_sub_pc_factor_fill": 1.2
    }

    def __init__(self, equations, q0, nu, beta2, tau, alpha=0.5,
                 solver_parameters=None, overwrite_solver_parameters=False):
        """
        Args:
            equations (:class:`PrognosticEquation`): the model's equation.
            alpha (float, optional): the semi-implicit off-centring factor.
                Defaults to 0.5. A value of 1 is fully-implicit.
            q0 (float): scaling factor for the saturation function.
            nu (float): scaling factor for the saturation function.
            beta2 (float): feedback proportionality on the buoyancy equation due
                to phase changes.
            tau (float): the timescale for the moist process.
            solver_parameters (dict, optional): contains the options to be
                passed to the underlying :class:`LinearVariationalSolver`.
                Defaults to None.
            overwrite_solver_parameters (bool, optional): if True use only the
                `solver_parameters` that have been passed in. If False then
                update the default parameters with the `solver_parameters`
                passed in. Defaults to False.
        """
        self.q0 = q0
        self.nu = nu
        self.beta2 = beta2
        self.tau = tau

        super().__init__(equations, alpha, solver_parameters,
                         overwrite_solver_parameters)

    @timed_function("Gusto:SolverSetup")
    def _setup_solver(self):
        equation = self.equations      # just cutting down line length a bit
        dt = self.dt
        beta_ = dt*self.alpha
        Vu = equation.domain.spaces("HDiv")
        VD = equation.domain.spaces("DG")
        Vb = equation.domain.spaces("DG")
        Vv = equation.domain.spaces("DG")
        Vc = equation.domain.spaces("DG")

        # Check that we have the correct fields
        if not equation.field_names[2] == 'b':
            raise NotImplementedError("Field 'b' must exist to use the moist thermal linear solver in the SIQN scheme")
        if not equation.field_names[3] == 'water_vapour':
            raise NotImplementedError("Field 'water_vapour' must exist to use the moist thermal linear solver in the SIQN scheme")
        if not equation.field_names[4] == 'cloud_water':
            raise NotImplementedError("Field 'cloud_water' must exist to use the moist thermal linear solver in the SIQN scheme")

        # Store time-stepping coefficients as UFL Constants
        beta = Constant(beta_)

        # Split up the rhs vector
        self.xrhs = Function(self.equations.function_space)
        u_in = split(self.xrhs)[0]
        D_in = split(self.xrhs)[1]
        b_in = split(self.xrhs)[2]
        qv_in = split(self.xrhs)[3]
        qc_in = split(self.xrhs)[4]

        # Split up the xn vector
        self.xn = Function(self.equations.function_space)
        D_xn = split(self.xn)[1]
        b_xn = split(self.xn)[2]
        qv_xn = split(self.xn)[3]
        qc_xn = split(self.xn)[4]

        # Function space and functions for the linear problem
        W = equation.function_space
        w, phi, lamda, tau1, tau2 = TestFunctions(W)
        u, D, b, qv, qc = TrialFunctions(W)

        # Background depth, buoyancy, vapour and cloud
        Dbar = split(equation.X_ref)[1]
        bbar = split(equation.X_ref)[2]
        qvbar = split(equation.X_ref)[3]
        qcbar = split(equation.X_ref)[4]

        n = FacetNormal(equation.domain.mesh)

        # get saturation function
        g = equation.parameters.g
        H = equation.parameters.H
        # just to shorten line length
        q0 = self.q0
        beta2 = self.beta2
        nu = self.nu

        # check for topography
        if hasattr(equation.prescribed_fields, "topography"):
            B = equation.prescribed_fields("topography")
        else:
            B = None

        # functions for sat function and source term P
        self.sat_func = Function(VD)
        self.P_func = Function(VD)

        # expressions for saturation and P (to be interpolated)
        sat_expr = compute_saturation(q0, nu, H, g, D_xn, b_xn, B)
        P_expr = (qv_xn - self.sat_func)/self.tau
        P_expr = conditional(sat_expr < 0,
                             max_value(sat_expr,
                                       -qc_xn/self.tau),
                             min_value(sat_expr,
                                       qv_xn/self.tau))

        # interpolators for sat function and P function
        self.q_sat_interpolator = Interpolator(sat_expr, self.sat_func)
        self.P_interpolator = Interpolator(P_expr, self.P_func)

        # write the weak form of the system to solve
        eqn = (
            # u equation
            inner(w, (u - u_in)) * dx
            - beta * (D - Dbar) * div(w*bbar) * dx
            + beta * jump(w*bbar, n) * avg(D-Dbar) * dS
            - beta * 0.5 * Dbar * bbar * div(w) * dx
            - beta * 0.5 * Dbar * b * div(w) * dx
            - beta * 0.5 * bbar * div(w*(D-Dbar)) * dx
            + beta * 0.5 * jump((D-Dbar)*w, n) * avg(bbar) * dS
            # D equation
            + inner(phi, (D - D_in)) * dx
            + beta * phi * Dbar * div(u) * dx
            # b equation
            + inner(lamda, (b - b_in)) * dx
            - beta * bbar * div(lamda*u) * dx
            + beta * jump(lamda*u, n) * avg(bbar) * dS
            + beta * lamda * beta2 * self.P_func * dx
            # qv equation
            + inner(tau1, (qv - qv_in)) * dx
            - beta * qvbar * div(tau1*u) * dx
            + beta * jump(tau1*u, n) * avg(qvbar) * dS
            + beta * tau1 * self.P_func * dx
            # qc equation
            + inner(tau2, (qc - qc_in)) * dx
            - beta * qcbar * div(tau2*u) * dx
            + beta * jump(tau2*u, n) * avg(qcbar) * dS
            - beta * tau2 * self.P_func * dx
        )

        if 'coriolis' in equation.prescribed_fields._field_names:
            f = equation.prescribed_fields('coriolis')
            eqn += beta * f * inner(w, equation.domain.perp(u)) * dx

        aeqn = lhs(eqn)
        Leqn = rhs(eqn)

        # Place to put results of solver
        self.dy = Function(W)

        # Boundary conditions
        bcs = [DirichletBC(M.sub(0), bc.function_arg, bc.sub_domain) for bc in self.equations.bcs['u']]

        # Solver
        problem = LinearVariationalProblem(aeqn, Leqn, self.dy, bcs=bcs)

        self.solver = LinearVariationalSolver(problem,
                                              solver_parameters=self.solver_parameters)

    @timed_function("Gusto:LinearSolve")
    def solve(self, xrhs, dy, xn):
        """
        Solve the linear problem.

        Args:
            xrhs (:class:`Function`): the right-hand side field in the
                appropriate :class:`MixedFunctionSpace`.
            dy (:class:`Function`): the resulting field in the appropriate
                :class:`MixedFunctionSpace`.
        """
        self.xrhs.assign(xrhs)

        # Check that the reference profiles have been set
        bbar = split(self.equations.X_ref)[2]
        b = dy.subfunctions[2]
        bbar_func = Function(b.function_space()).interpolate(bbar)
        if bbar_func.dat.data.max() == 0 and bbar_func.dat.data.min() == 0:
            logger.warning("The reference profile for b in the linear solver is zero. To set a non-zero profile add b to the set_reference_profiles argument.")

        qvbar = split(self.equations.X_ref)[3]
        qv = dy.subfunctions[3]
        qvbar_func = Function(qv.function_space()).interpolate(qvbar)
        if qvbar_func.dat.data.max() == 0 and qvbar_func.dat.data.min() == 0:
            logger.warning("The reference profile for vapour in the linear solver is zero. To set a non-zero profile add water_vapour to the set_reference_profiles argument.")

        qcbar = split(self.equations.X_ref)[4]
        qc = dy.subfunctions[3]
        qcbar_func = Function(qc.function_space()).interpolate(qcbar)
        if qcbar_func.dat.data.max() == 0 and qcbar_func.dat.data.min() == 0:
            logger.warning("The reference profile for cloud in the linear solver is zero. To set a non-zero profile add cloud_water to the set_reference_profiles argument.")

        # Call the saturation function and source function interpolators
        self.q_sat_interpolator.interpolate()
        self.P_interpolator.interpolate()

        # Solve
        self.solver.solve()
        dy.assign(self.dy)
