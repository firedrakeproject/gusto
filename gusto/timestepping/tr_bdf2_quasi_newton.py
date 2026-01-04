"""
The TR-BDF2 Quasi-Newton timestepper.
"""

from firedrake import (
    Function, FunctionSpace, sqrt
)


from pyop2.profiling import timed_stage
from gusto.core import TimeLevelFields, StateFields
from gusto.core.labels import (transport, diffusion, sponge, incompressible)
from gusto.core.logging import logger
from gusto.time_discretisation.time_discretisation import ExplicitTimeDiscretisation
from gusto.timestepping.timestepper import BaseTimestepper
from gusto.timestepping.semi_implicit_quasi_newton import Forcing, QuasiNewtonLinearSolver
from gusto.solvers.solver_presets import hybridised_solver_parameters


__all__ = ["TRBDF2QuasiNewton"]


class TRBDF2QuasiNewton(BaseTimestepper):
    """
    Implements a TR-BDF2 quasi-Newton discretisation,
    with Strang splitting and auxiliary semi-Lagrangian transport.

    The timestep consists of a two stage timestep in which an initial SIQN
    step is followed by a BDF2 stage.

    """

    def __init__(self, equation_set, io, transport_schemes, spatial_methods,
                 tr_solver=None,
                 bdf_solver=None,
                 diffusion_schemes=None,
                 physics_schemes=None,
                 slow_physics_schemes=None,
                 fast_physics_schemes=None,
                 gamma=(1-sqrt(2)/2),
                 tau_values_tr=None,
                 tau_values_bdf=None,
                 num_outer_tr=2, num_inner_tr=2,
                 num_outer_bdf=2, num_inner_bdf=2,
                 reference_update_freq=None,
                 solver_prognostics=None,
                 tr_solver_parameters=None,
                 tr_appctx=None,
                 bdf_solver_parameters=None,
                 bdf_appctx=None
                 ):
        """
        Args:
            equation_set (:class:`PrognosticEquationSet`): the prognostic
                equation set to be solved
            io (:class:`IO`): the model's object for controlling input/output.
            transport_schemes: iterable of ``(field_name, scheme)`` pairs
                indicating the name of the field (str) to transport, and the
                :class:`TimeDiscretisation` to use
            spatial_methods (iter): a list of objects describing the spatial
                discretisations of transport or diffusion terms to be used.
            tr_solver (:class:`TimesteppingSolver`, optional): the object
                to use for the linear solve in the TR step. Defaults to None.
            bdf_solver (:class:`TimesteppingSolver`, optional): the object
                to use for the linear solve in the BDF step. Defaults to None.
            diffusion_schemes (iter, optional): iterable of pairs of the form
                ``(field_name, scheme)`` indicating the fields to diffuse, and
                the :class:`~.TimeDiscretisation` to use. Defaults to None.
            physics_schemes: (list, optional): a list of tuples of the form
                (:class:`PhysicsParametrisation`, :class:`TimeDiscretisation`),
                pairing physics parametrisations and timestepping schemes to use
                for each. Timestepping schemes for physics must be explicit.
                These schemes are all evaluated at the end of the time step.
                Defaults to None.
            slow_physics_schemes: (list, optional): a list of tuples of the form
                (:class:`PhysicsParametrisation`, :class:`TimeDiscretisation`).
                These schemes are all evaluated at the start of the time step.
                Defaults to None.
            fast_physics_schemes: (list, optional): a list of tuples of the form
                (:class:`PhysicsParametrisation`, :class:`TimeDiscretisation`).
                These schemes are evaluated within the outer loop. Defaults to
                None. -- Not yet implemented--
            gamma (`float, optional): the off-centering parameter for the
                timestep between 0 and 0.5. A value of 0.5 corresponds to a SIQN
                scheme with alpha = 0.5. Defaults to 1 - sqrt(2)/2, which makes
                the scheme L-stable.
            tau_values_tr (dict, optional): a dictionary with keys the names of
                prognostic variables and values the tau values to use for each
                variable. Defaults to None, in which case the value of gamma
                is used.
            tau_values_bdf (dict, optional): a dictionary with keys the names of
                prognostic variables and values the tau values to use for each
                variable. Defaults to None, in which case the value of gamma2
                is used.
            num_outer_tr (int, optional): number of outer iterations in the
                trapeziodal step. The outer loop includes transport and any
                fast physics schemes. Defaults to 2.
            num_inner_tr (int, optional): number of inner iterations in the
                trapeziodal step. The inner loop includes the evaluation of
                implicit forcing (pressure gradient and Coriolis) terms, and the
                linear solve. Defaults to 2.
            num_outer_bdf (int, optional): number of outer iterations in the
                BDF step. The outer loop includes transport and any
                fast physics schemes. Defaults to 2.
            num_inner_bdf (int, optional): number of inner iterations in the
                BDF step. The inner loop includes the evaluation of
                implicit forcing (pressure gradient and Coriolis) terms, and the
                linear solve. Defaults to 2.
            reference_update_freq (float, optional): frequency with which to
                update the reference profile with the n-th time level state
                fields. This variable corresponds to time in seconds, and
                setting this to zero will update the reference profiles every
                time step. Setting it to None turns off the update, and
                reference profiles will remain at their initial values.
                Defaults to None.
            solver_prognostics (list, optional): a list of the names of
                prognostic variables to include in the solver. Defaults to None,
                in which case all prognostics that aren't active tracers are
                included in the solver.
            tr_solver_parameters (dict, optional): contains the options to
                be passed to the underlying :class:`LinearVariationalSolver` for the
                trapezoidal step.
                Defaults to None.
            tr_appctx (dict, optional): a dictionary of application context for the
                underlying :class:`LinearVariationalSolver` for the trapezoidal step.
                Defaults to None.
            bdf_solver_parameters (dict, optional): contains the options to
                be passed to the underlying :class:`LinearVariationalSolver` for the
                BDF2 step.
                Defaults to None.
            bdf_appctx (dict, optional): a dictionary of application context for the
                underlying :class:`LinearVariationalSolver` for the BDF2 step.
                Defaults to None.
        """

        self.num_outer_tr = num_outer_tr
        self.num_inner_tr = num_inner_tr

        self.num_outer_bdf = num_outer_bdf
        self.num_inner_bdf = num_inner_bdf

        mesh = equation_set.domain.mesh
        R = FunctionSpace(mesh, "R", 0)
        self.gamma = Function(R, val=float(gamma))
        self.gamma2 = Function(R, val=((1 - 2*float(gamma))/(2 - 2*float(gamma))))
        self.gamma3 = Function(R, val=((1-float(self.gamma2))/(2*float(gamma))))
        self.implicit_terms = [incompressible, sponge]

        # Options relating to reference profiles
        self.reference_update_freq = reference_update_freq
        self.to_update_ref_profile = False

        # Set transporting velocity to be average
        self.alpha_u = Function(R, val=0.5)
        self.implicit_terms = [incompressible, sponge]
        self.spatial_methods = spatial_methods

        # Determine prognostics for solver -------------------------------------
        self.non_solver_prognostics = []
        if solver_prognostics is not None:
            assert type(solver_prognostics) is list, (
                "solver_prognostics should be a list of prognostic variable "
                + f"names, and not {type(solver_prognostics)}"
            )
            for prognostic_name in solver_prognostics:
                assert prognostic_name in equation_set.field_names, (
                    f"Prognostic variable {prognostic_name} not found in "
                    + "equation set field names"
                )
            for field_name in equation_set.field_names:
                if field_name not in solver_prognostics:
                    self.non_solver_prognostics.append(field_name)
        else:
            # Remove any active tracers by default
            solver_prognostics = []
            if equation_set.active_tracers is not None:
                self.non_solver_prognostics = [
                    tracer.name for tracer in equation_set.active_tracers
                ]
                for field_name in equation_set.field_names:
                    if field_name not in self.non_solver_prognostics:
                        solver_prognostics.append(field_name)
            else:
                # No active tracers so prognostics are field names
                solver_prognostics = equation_set.field_names.copy()

        logger.info(
            'TR-BDF2 Quasi-Newton: Using solver prognostics '
            + f'{solver_prognostics}'
        )

        if physics_schemes is not None:
            self.final_physics_schemes = physics_schemes
        else:
            self.final_physics_schemes = []
        if slow_physics_schemes is not None:
            self.slow_physics_schemes = slow_physics_schemes
        else:
            self.slow_physics_schemes = []
        if fast_physics_schemes is not None:
            self.fast_physics_schemes = fast_physics_schemes
            raise NotImplementedError(
                "Fast physics schemes are not yet implemented for the TR-BDF "
                "timestepper. This is because the time discretisations used by the "
                "physics schemes need to take into account the appropriate gamma "
                "factors for their timestep."
            )
        else:
            self.fast_physics_schemes = []
        self.all_physics_schemes = (self.slow_physics_schemes
                                    + self.fast_physics_schemes
                                    + self.final_physics_schemes)

        for parametrisation, scheme in self.all_physics_schemes:
            assert scheme.nlevels == 1, "multilevel schemes not supported as part of this timestepping loop"
            if hasattr(parametrisation, "explicit_only") and parametrisation.explicit_only:
                assert isinstance(scheme, ExplicitTimeDiscretisation), \
                    ("Only explicit time discretisations can be used with "
                     + f"physics scheme {parametrisation.label.label}")

        self.active_transport = []
        for scheme in transport_schemes:
            assert scheme.nlevels == 1, "multilevel schemes not supported as part of this timestepping loop"
            assert scheme.field_name in equation_set.field_names
            # Check that there is a corresponding transport method
            method_found = False
            for method in spatial_methods:
                if scheme.field_name == method.variable and method.term_label == transport:
                    method_found = True
                    self.active_transport.append((scheme.field_name, scheme))
            assert method_found, f'No transport method found for variable {scheme.field_name}'

        self.diffusion_schemes = []
        if diffusion_schemes is not None:
            for scheme in diffusion_schemes:
                assert scheme.nlevels == 1, "multilevel schemes not supported as part of this timestepping loop"
                assert scheme.field_name in equation_set.field_names
                self.diffusion_schemes.append((scheme.field_name, scheme))
                # Check that there is a corresponding transport method
                method_found = False
                for method in spatial_methods:
                    if scheme.field_name == method.variable and method.term_label == diffusion:
                        method_found = True
                assert method_found, f'No diffusion method found for variable {scheme.field_name}'

        super().__init__(equation_set, io)

        self.tracers_to_copy = []
        if equation_set.active_tracers is not None:
            for active_tracer in equation_set.active_tracers:
                self.tracers_to_copy.append(active_tracer.name)

        self.field_name = equation_set.field_name
        W = equation_set.function_space
        self.xrhs = Function(W)
        self.xrhs_phys = Function(W)
        self.dy = Function(W)

        if tr_solver is None:
            if tr_solver_parameters is None:
                self.tr_solver_parameters, self.tr_appctx = \
                    hybridised_solver_parameters(equation_set, alpha=self.gamma, tau_values=tau_values_tr)
            else:
                self.tr_solver_parameters = tr_solver_parameters
                self.tr_appctx = tr_appctx
            self.tr_solver = QuasiNewtonLinearSolver(
                equation_set, solver_prognostics, self.implicit_terms,
                self.gamma, tau_values=tau_values_tr,
                solver_parameters=self.tr_solver_parameters,
                appctx=self.tr_appctx
            )
        else:
            self.tr_solver = tr_solver

        if bdf_solver is None:
            if bdf_solver_parameters is None:
                self.bdf_solver_parameters, self.bdf_appctx = \
                    hybridised_solver_parameters(equation_set, alpha=self.gamma2, tau_values=tau_values_bdf)
            else:
                self.bdf_solver_parameters = bdf_solver_parameters
                self.bdf_appctx = bdf_appctx
            self.bdf_solver = QuasiNewtonLinearSolver(
                equation_set, solver_prognostics, self.implicit_terms,
                self.gamma2, tau_values=tau_values_bdf,
                solver_parameters=self.bdf_solver_parameters,
                appctx=self.bdf_appctx
            )
        else:
            self.bdf_solver = bdf_solver

        dt = self.equation.domain.dt
        self.tr_forcing = Forcing(equation_set, self.implicit_terms, alpha=0.5, dt=2.0*self.gamma*dt)
        self.bdf_forcing = Forcing(equation_set, self.implicit_terms, alpha=1.0, dt=self.gamma2*dt)
        self.bcs = equation_set.bcs

    def _apply_bcs(self, X):
        """
        Set the zero boundary conditions in the velocity.
        """
        for bc in self.bcs['u']:
            bc.apply(X('u'))

    @property
    def transporting_velocity(self):
        return self.ubar

    def update_transporting_velocity(self, uk, ukp1, factor):
        """Update transporting velocity by combining uk and ukp1"""
        # note: factor takes into account different timestep scalings
        self.ubar.assign(factor*0.5*(uk + ukp1))

    def setup_fields(self):
        """Sets up time levels"""
        self.x = TimeLevelFields(self.equation, 1)
        self.x.add_fields(self.equation, levels=("star", "p", "after_slow", "after_fast", "m", "pm"))
        # Only the prescribed fields of the main equation need passing to StateFields
        self.fields = StateFields(self.x, self.equation.prescribed_fields,
                                  *self.io.output.dumplist)
        self.ubar = Function(self.x.n('u').function_space())

    def setup_scheme(self):
        """Sets up transport, diffusion and physics schemes"""
        # TODO: apply_bcs should be False for advection but this means
        # tests with KGOs fail
        apply_bcs = True
        self.setup_equation(self.equation)
        for _, scheme in self.active_transport:
            scheme.setup(self.equation, apply_bcs, transport)
            self.setup_transporting_velocity(scheme)
            if self.io.output.log_courant:
                scheme.courant_max = self.io.courant_max

        apply_bcs = True
        for _, scheme in self.diffusion_schemes:
            scheme.setup(self.equation, apply_bcs, diffusion)
        for parametrisation, scheme in self.all_physics_schemes:
            apply_bcs = True
            scheme.setup(self.equation, apply_bcs, parametrisation.label)

    def copy_active_tracers(self, x_in, x_out):
        """
        Copies active tracers from one set of fields to another, if those fields
        are not included in the linear solver. This is determined by whether the
        time derivative term for that tracer has a linearisation.

        Args:
           x_in:  The input set of fields
           x_out: The output set of fields
        """

        for name in self.tracers_to_copy:
            x_out(name).assign(x_in(name))

    def transport_fields(self, outer, x_in, x_out):
        """
        Transports all fields in x_in with a transport scheme
        and places the result in x_out.

        Args:
            outer (int): the outer loop iteration number
            x_in (:class:`Fields`): the collection of state fields to be
                  transported.
            x_out (:class:`Fields`): the collection of state fields resulting from
                  the transport.
        """
        for name, scheme in self.active_transport:

            logger.info(f'TR-BDF2 Quasi Newton: Transport {outer}: {name}')
            # transports a single field from x_in and puts the result in x_out
            scheme.apply(x_out(name), x_in(name))

    def update_reference_profiles(self):
        """
        Updates the reference profiles and if required also updates them in the
        linear solver.
        """

        if self.reference_update_freq is not None:
            if float(self.t) + self.reference_update_freq > self.last_ref_update_time:
                self.equation.X_ref.assign(self.x.n(self.field_name))
                self.last_ref_update_time = float(self.t)
                self.tr_solver.update_reference_profiles()
                self.bdf_solver.update_reference_profiles()

        elif self.to_update_ref_profile:
            self.tr_solver.update_reference_profiles()
            self.bdf_solver.update_reference_profiles()
            self.to_update_ref_profile = False

    def timestep(self):
        """Defines the timestep"""
        xn = self.x.n
        xnp1 = self.x.np1
        xstar = self.x.star
        xp = self.x.p
        xpm = self.x.pm
        xm = self.x.m
        x_after_slow = self.x.after_slow
        x_after_fast = self.x.after_fast
        xrhs = self.xrhs
        xrhs_phys = self.xrhs_phys
        dy = self.dy
        fname = self.equation.field_name

        # Make first guess of xm
        xm(fname).assign(xn(fname))

        # Update reference profiles --------------------------------------------
        self.update_reference_profiles()

        # Slow physics ---------------------------------------------------------
        x_after_slow(self.field_name).assign(xn(self.field_name))
        if len(self.slow_physics_schemes) > 0:
            with timed_stage("Slow physics"):
                logger.info('TR-BDF2 Quasi Newton: TR Slow physics')
                for _, scheme in self.slow_physics_schemes:
                    scheme.apply(x_after_slow(scheme.field_name), x_after_slow(scheme.field_name))

        # TR step ==============================================================

        # Explicit forcing -----------------------------------------------------
        with timed_stage("Apply forcing terms"):
            logger.info('TR-BDF2 Quasi Newton: TR Explicit forcing')
            # Put explicit forcing into xstar
            self.tr_forcing.apply(x_after_slow, xn, xstar(self.field_name), "explicit")

        # set xp here so that variables that are not transported have
        # the correct values
        xp(self.field_name).assign(xstar(self.field_name))

        # OUTER ----------------------------------------------------------------
        for outer in range(self.num_outer_tr):

            # Transport --------------------------------------------------------
            with timed_stage("Transport"):
                # Transport by 0.5*(u^n + u^m) for 2*gamma*dt
                self.update_transporting_velocity(xn('u'), xm('u'), 2*self.gamma)
                self.io.log_courant(self.fields, 'transporting_velocity',
                                    message=f'TR: transporting velocity, outer iteration {outer}')
                self.transport_fields(f'TR: {outer}', xstar, xp)

            # Fast physics -----------------------------------------------------
            x_after_fast(self.field_name).assign(xp(self.field_name))
            if len(self.fast_physics_schemes) > 0:
                with timed_stage("Fast physics"):
                    logger.info(f'TR-BDF2 Quasi Newton: TR Fast physics {outer}')
                    for _, scheme in self.fast_physics_schemes:
                        scheme.apply(x_after_fast(scheme.field_name), x_after_fast(scheme.field_name))

            xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve
            xrhs_phys.assign(x_after_fast(self.field_name) - xp(self.field_name))

            for inner in range(self.num_inner_tr):

                # Implicit forcing ---------------------------------------------
                with timed_stage("Apply forcing terms"):
                    logger.info(f'TR-BDF2 Quasi Newton: TR Implicit forcing {(outer, inner)}')
                    self.tr_forcing.apply(xp, xm, xrhs, "implicit")
                    xrhs += xrhs_phys
                    if inner > 0:
                        # Zero implicit forcing to accelerate solver convergence
                        self.tr_forcing.zero_non_wind_terms(self.equation, xm, xrhs, self.equation.field_names)

                xrhs -= xm(self.field_name)

                # Linear solve -------------------------------------------------

                with timed_stage("Implicit solve"):
                    logger.info(f'TR-BDF2 Quasi Newton: TR Mixed solve {(outer, inner)}')
                    self.tr_solver.solve(xrhs, dy)  # solves linear system and places result in dy

                xmX = xm(self.field_name)
                xmX += dy

            # Update xnp1 values for active tracers not included in the linear solve
            self.copy_active_tracers(x_after_fast, xm)

            self._apply_bcs(xm)

        # BDF step =============================================================

        # set xp here so that variables that are not transported have
        # the correct values
        xp(self.field_name).assign(xn(fname))
        xpm(self.field_name).assign(xm(fname))
        xnp1(self.field_name).assign(xn(fname))  # First guess doesn't seem to make much difference

        # OUTER ----------------------------------------------------------------
        for outer in range(self.num_outer_bdf):

            # Transport --------------------------------------------------------
            with timed_stage("Transport"):
                # Transport by u^np1 for (1-2*gamma)*dt, so scale u by (1-2*gamma)
                self.update_transporting_velocity(xnp1('u'), xnp1('u'), 1-2*self.gamma)
                self.io.log_courant(self.fields, 'transporting_velocity',
                                    message=f'BDF m: transporting velocity, outer iteration {outer}')
                self.transport_fields(f'BDF m: {outer}', xm, xpm)

                # Transport by u^np1 for dt, so scale u by 1
                self.update_transporting_velocity(xnp1('u'), xnp1('u'), 1)
                self.io.log_courant(self.fields, 'transporting_velocity',
                                    message=f'BDF n: transporting velocity, outer iteration {outer}')
                self.transport_fields(f'BDF n:{outer}', xn, xp)

            # Combine transported fields into a single variable
            xp(fname).assign((1 - self.gamma3)*xp(fname) + self.gamma3*xpm(fname))

            # Fast physics -----------------------------------------------------
            x_after_fast(self.field_name).assign(xp(self.field_name))
            if len(self.fast_physics_schemes) > 0:
                with timed_stage("Fast physics"):
                    logger.info(f'TR-BDF2 Quasi Newton: BDF Fast physics {outer}')
                    for _, scheme in self.fast_physics_schemes:
                        scheme.apply(x_after_fast(scheme.field_name), x_after_fast(scheme.field_name))

            xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve
            xrhs_phys.assign(x_after_fast(self.field_name) - xp(self.field_name))
            for inner in range(self.num_inner_bdf):
                # Implicit forcing ---------------------------------------------
                with timed_stage("Apply forcing terms"):
                    logger.info(f'TR-BDF2 Quasi Newton: BDF Implicit forcing {(outer, inner)}')
                    self.bdf_forcing.apply(xp, xnp1, xrhs, "implicit")
                    xrhs += xrhs_phys
                    if inner > 0:
                        # Zero implicit forcing to accelerate solver convergence
                        self.bdf_forcing.zero_non_wind_terms(self.equation, xnp1, xrhs, self.equation.field_names)

                xrhs -= xnp1(self.field_name)

                # Linear solve -------------------------------------------------
                with timed_stage("Implicit solve"):
                    logger.info(f'TR-BDF2 Quasi Newton: BDF Mixed solve {(outer, inner)}')
                    self.bdf_solver.solve(xrhs, dy)  # solves linear system and places result in dy

                xnp1X = xnp1(self.field_name)
                xnp1X += dy

            # Update xnp1 values for active tracers not included in the linear solve
            self.copy_active_tracers(x_after_fast, xnp1)
            self._apply_bcs(xnp1)

        with timed_stage("Diffusion"):
            for name, scheme in self.diffusion_schemes:
                logger.debug(f"TR-BDF2 Quasi-Newton diffusing {name}")
                scheme.apply(xnp1(name), xnp1(name))

        if len(self.final_physics_schemes) > 0:
            with timed_stage("Final Physics"):
                for _, scheme in self.final_physics_schemes:
                    scheme.apply(xnp1(scheme.field_name), xnp1(scheme.field_name))

        logger.debug("Leaving TR-BDF2 Quasi-Newton timestep method")

    def run(self, t, tmax, pick_up=False):
        """
        Runs the model for the specified time, from t to tmax.

        Args:
            t (float): the start time of the run
            tmax (float): the end time of the run
            pick_up: (bool): specify whether to pick_up from a previous run
        """

        if not pick_up and self.reference_update_freq is None:
            assert self.reference_profiles_initialised, \
                'Reference profiles for must be initialised to use TR-BDF2 Timestepper'

        if not pick_up and self.reference_update_freq is not None:
            # Force reference profiles to be updated on first time step
            self.last_ref_update_time = float(t) - float(self.dt)

        elif not pick_up or (pick_up and self.reference_update_freq is None):
            # Indicate that linear solver profile needs updating
            self.to_update_ref_profile = True

        super().run(t, tmax, pick_up=pick_up)
