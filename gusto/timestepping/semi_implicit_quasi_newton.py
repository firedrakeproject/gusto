"""
The Semi-Implicit Quasi-Newton timestepper used by the Met Office's ENDGame
and GungHo dynamical cores.
"""

from firedrake import (
    Function, Constant, TrialFunctions, DirichletBC, div, assemble,
    LinearVariationalProblem, LinearVariationalSolver, FunctionSpace
)
from firedrake.fml import drop, replace_subject
from firedrake.__future__ import interpolate
from pyop2.profiling import timed_stage
from gusto.core import TimeLevelFields, StateFields
from gusto.core.labels import (transport, diffusion, time_derivative,
                               hydrostatic, physics_label, sponge,
                               incompressible)
from gusto.solvers import LinearTimesteppingSolver, mass_parameters
from gusto.core.logging import logger, DEBUG, logging_ksp_monitor_true_residual
from gusto.time_discretisation.time_discretisation import ExplicitTimeDiscretisation
from gusto.timestepping.timestepper import BaseTimestepper


__all__ = ["SemiImplicitQuasiNewton"]


class SemiImplicitQuasiNewton(BaseTimestepper):
    """
    Implements a semi-implicit quasi-Newton discretisation,
    with Strang splitting and auxiliary semi-Lagrangian transport.

    The timestep consists of an outer loop applying the transport and an
    inner loop to perform the quasi-Newton interations for the fast-wave
    terms.
    """

    def __init__(self, equation_set, io, transport_schemes, spatial_methods,
                 auxiliary_equations_and_schemes=None, linear_solver=None,
                 diffusion_schemes=None, physics_schemes=None,
                 slow_physics_schemes=None, fast_physics_schemes=None,
                 alpha=Constant(0.5), off_centred_u=False,
                 num_outer=2, num_inner=2, accelerator=False,
                 predictor=None, reference_update_freq=None,
                 spinup_steps=0):
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
            auxiliary_equations_and_schemes: iterable of ``(equation, scheme)``
                pairs indicating any additional equations to be solved and the
                scheme to use to solve them. Defaults to None.
            linear_solver (:class:`TimesteppingSolver`, optional): the object
                to use for the linear solve. Defaults to None.
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
                None.
            alpha (`ufl.Constant`, optional): the semi-implicit off-centering
                parameter. A value of 1 corresponds to fully implicit, while 0
                corresponds to fully explicit. Defaults to Constant(0.5).
            off_centred_u (bool, optional): option to offcentre the transporting
                velocity. Defaults to False, in which case transporting velocity
                is centred. If True offcentring uses value of alpha.
            num_outer (int, optional): number of outer iterations in the semi-
                implicit algorithm. The outer loop includes transport and any
                fast physics schemes. Defaults to 2. Note that default used by
                the Met Office's ENDGame and GungHo models is 2.
            num_inner (int, optional): number of inner iterations in the semi-
                implicit algorithm. The inner loop includes the evaluation of
                implicit forcing (pressure gradient and Coriolis) terms, and the
                linear solve. Defaults to 2. Note that default used by the Met
                Office's ENDGame and GungHo models is 2.
            accelerator (bool, optional): Whether to zero non-wind implicit
                forcings for transport terms in order to speed up solver
                convergence. Defaults to False.
            predictor (str, optional): a single string corresponding to the name
                of a variable to transport using the divergence predictor. This
                pre-multiplies that variable by (1 - beta*dt*div(u)) before the
                transport step, and calculates its transport increment from the
                transport of this variable. This can improve the stability of
                the time stepper at large time steps, when not using an
                advective-then-flux formulation. This is only suitable for the
                use on the conservative variable (e.g. depth or density).
                Defaults to None, in which case no predictor is used.
            reference_update_freq (float, optional): frequency with which to
                update the reference profile with the n-th time level state
                fields. This variable corresponds to time in seconds, and
                setting this to zero will update the reference profiles every
                time step. Setting it to None turns off the update, and
                reference profiles will remain at their initial values.
                Defaults to None.
            spinup_steps (int, optional): the number of steps to run the model
                in "spin-up" mode, where the alpha parameter is set to 1.0.
                Defaults to 0, which corresponds to no spin-up.
        """

        self.num_outer = num_outer
        self.num_inner = num_inner
        mesh = equation_set.domain.mesh
        R = FunctionSpace(mesh, "R", 0)
        self.alpha = Function(R, val=float(alpha))
        self.predictor = predictor
        self.accelerator = accelerator

        # Options relating to reference profiles and spin-up
        self._alpha_original = Function(R, val=float(alpha))
        self.reference_update_freq = reference_update_freq
        self.to_update_ref_profile = False
        self.spinup_steps = spinup_steps
        self.spinup_begun = False
        self.spinup_done = False

        # Flag for if we have simultaneous transport
        self.simult = False

        # default is to not offcentre transporting velocity but if it
        # is offcentred then use the same value as alpha
        self.alpha_u = Function(R, val=float(alpha)) if off_centred_u else Function(R, val=0.5)

        self.spatial_methods = spatial_methods

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
        self.transported_fields = []
        for scheme in transport_schemes:
            assert scheme.nlevels == 1, "multilevel schemes not supported as part of this timestepping loop"
            if isinstance(scheme.field_name, list):
                # This means that multiple fields are being transported simultaneously
                self.simult = True
                for subfield in scheme.field_name:
                    assert subfield in equation_set.field_names

                    # Check that there is a corresponding transport method for
                    # each field in the list
                    method_found = False
                    for method in spatial_methods:
                        if subfield == method.variable and method.term_label == transport:
                            method_found = True
                    assert method_found, f'No transport method found for variable {scheme.field_name}'
                self.active_transport.append((scheme.field_name, scheme))
            else:
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

        if auxiliary_equations_and_schemes is not None:
            for eqn, scheme in auxiliary_equations_and_schemes:
                assert not hasattr(eqn, "field_names"), 'Cannot use auxiliary schemes with multiple fields'
            self.auxiliary_schemes = [
                (eqn.field_name, scheme)
                for eqn, scheme in auxiliary_equations_and_schemes]

        else:
            auxiliary_equations_and_schemes = []
            self.auxiliary_schemes = []
        self.auxiliary_equations_and_schemes = auxiliary_equations_and_schemes

        super().__init__(equation_set, io)

        for aux_eqn, aux_scheme in self.auxiliary_equations_and_schemes:
            self.setup_equation(aux_eqn)
            aux_scheme.setup(aux_eqn)
            self.setup_transporting_velocity(aux_scheme)

        self.tracers_to_copy = []
        if equation_set.active_tracers is not None:
            for active_tracer in equation_set.active_tracers:
                self.tracers_to_copy.append(active_tracer.name)

        self.field_name = equation_set.field_name
        W = equation_set.function_space
        self.xrhs = Function(W)
        self.xrhs_phys = Function(W)
        self.dy = Function(W)
        if linear_solver is None:
            self.linear_solver = LinearTimesteppingSolver(equation_set, self.alpha)
        else:
            self.linear_solver = linear_solver
        self.forcing = Forcing(equation_set, self.alpha)
        self.bcs = equation_set.bcs

        if self.predictor is not None:
            V_DG = equation_set.domain.spaces('DG')
            self.predictor_field_in = Function(V_DG)
            div_factor = Constant(1.0) - (Constant(1.0) - self.alpha)*self.dt*div(self.x.n('u'))
            self.predictor_interpolate = interpolate(
                self.x.star(predictor)*div_factor, V_DG)

    def _apply_bcs(self):
        """
        Set the zero boundary conditions in the velocity.
        """
        unp1 = self.x.np1("u")

        for bc in self.bcs['u']:
            bc.apply(unp1)

    @property
    def transporting_velocity(self):
        """Computes ubar=(1-alpha)*un + alpha*unp1"""
        xn = self.x.n
        xnp1 = self.x.np1
        # computes ubar from un and unp1
        return xn('u') + self.alpha_u*(xnp1('u')-xn('u'))

    def setup_fields(self):
        """Sets up time levels n, star, p and np1"""
        self.x = TimeLevelFields(self.equation, 1)
        if self.simult is True:
            # If there is any simultaneous transport, add an extra 'simult' field:
            self.x.add_fields(self.equation, levels=("star", "p", "simult", "after_slow", "after_fast"))
        else:
            self.x.add_fields(self.equation, levels=("star", "p", "after_slow", "after_fast"))
        for aux_eqn, _ in self.auxiliary_equations_and_schemes:
            self.x.add_fields(aux_eqn)
        # Prescribed fields for auxiliary eqns should come from prognostics of
        # other equations, so only the prescribed fields of the main equation
        # need passing to StateFields
        self.fields = StateFields(self.x, self.equation.prescribed_fields,
                                  *self.io.output.dumplist)

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

    def transport_fields(self, outer, xstar, xp):
        """
        Transports all fields in xstar with a transport scheme
        and places the result in xp.

        Args:
            outer (int): the outer loop iteration number
            xstar (:class:`Fields`): the collection of state fields to be
                transported.
            xp (:class:`Fields`): the collection of state fields resulting from
                the transport.
        """
        for name, scheme in self.active_transport:
            if isinstance(name, list):
                # Transport multiple fields from xstar simultaneously.
                # We transport the mixed function space from xstar to xsimult, then
                # extract the updated fields and pass them to xp; this avoids overwriting
                # any previously transported fields.
                logger.info(f'Semi-implicit Quasi Newton: Transport {outer}: '
                            + f'Simultaneous transport of {name}')
                scheme.apply(self.x.simult(self.field_name), xstar(self.field_name))
                for field_name in name:
                    xp(field_name).assign(self.x.simult(field_name))
            else:
                logger.info(f'Semi-implicit Quasi Newton: Transport {outer}: {name}')
                # transports a single field from xstar and puts the result in xp
                if name == self.predictor:
                    # Pre-multiply this variable by (1 - dt*beta*div(u))
                    V = xstar(name).function_space()
                    field_out = Function(V)
                    self.predictor_field_in.assign(assemble(self.predictor_interpolate))
                    scheme.apply(field_out, self.predictor_field_in)

                    # xp is xstar plus the increment from the transported predictor
                    xp(name).assign(xstar(name) + field_out - self.predictor_field_in)
                else:
                    # Standard transport
                    scheme.apply(xp(name), xstar(name))

    def update_reference_profiles(self):
        """
        Updates the reference profiles and if required also updates them in the
        linear solver.
        """

        if self.reference_update_freq is not None:
            if float(self.t) + self.reference_update_freq > self.last_ref_update_time:
                self.equation.X_ref.assign(self.x.n(self.field_name))
                self.last_ref_update_time = float(self.t)
                self.linear_solver.update_reference_profiles()

        elif self.to_update_ref_profile:
            self.linear_solver.update_reference_profiles()
            self.to_update_ref_profile = False

    def start_spinup(self):
        """
        Initialises the spin-up period, so that the scheme is implicit by
        setting the off-centering parameter alpha to be 1.
        """
        logger.debug('Starting spin-up period')
        # Update alpha
        self.alpha.assign(1.0)
        self.linear_solver.alpha.assign(1.0)
        # We need to tell solvers that they may need rebuilding
        self.linear_solver.update_reference_profiles()
        self.forcing.solvers['explicit'].invalidate_jacobian()
        self.forcing.solvers['implicit'].invalidate_jacobian()
        # This only needs doing once, so update the flag
        self.spinup_begun = True

    def finish_spinup(self):
        """
        Finishes the spin-up period, returning the off-centering parameter
        to its original value.
        """
        logger.debug('Finishing spin-up period')
        # Update alpha
        self.alpha.assign(self._alpha_original)
        self.linear_solver.alpha.assign(self._alpha_original)
        # We need to tell solvers that they may need rebuilding
        self.linear_solver.update_reference_profiles()
        self.forcing.solvers['explicit'].invalidate_jacobian()
        self.forcing.solvers['implicit'].invalidate_jacobian()
        # This only needs doing once, so update the flag
        self.spinup_done = True

    def timestep(self):
        """Defines the timestep"""
        xn = self.x.n
        xnp1 = self.x.np1
        xstar = self.x.star
        xp = self.x.p
        x_after_slow = self.x.after_slow
        x_after_fast = self.x.after_fast
        xrhs = self.xrhs
        xrhs_phys = self.xrhs_phys
        dy = self.dy

        # Update reference profiles --------------------------------------------
        self.update_reference_profiles()

        # Are we in spin-up period? --------------------------------------------
        # Note: steps numbered from 1 onwards
        if self.step < self.spinup_steps + 1 and not self.spinup_begun:
            self.start_spinup()
        elif self.step >= self.spinup_steps + 1 and not self.spinup_done:
            self.finish_spinup()

        # Slow physics ---------------------------------------------------------
        x_after_slow(self.field_name).assign(xn(self.field_name))
        if len(self.slow_physics_schemes) > 0:
            with timed_stage("Slow physics"):
                logger.info('Semi-implicit Quasi Newton: Slow physics')
                for _, scheme in self.slow_physics_schemes:
                    scheme.apply(x_after_slow(scheme.field_name), x_after_slow(scheme.field_name))

        # Explict forcing ------------------------------------------------------
        with timed_stage("Apply forcing terms"):
            logger.info('Semi-implicit Quasi Newton: Explicit forcing')
            # Put explicit forcing into xstar
            self.forcing.apply(x_after_slow, xn, xstar(self.field_name), "explicit")

        # set xp here so that variables that are not transported have
        # the correct values
        xp(self.field_name).assign(xstar(self.field_name))

        # OUTER ----------------------------------------------------------------
        for outer in range(self.num_outer):

            # Transport --------------------------------------------------------
            with timed_stage("Transport"):
                self.io.log_courant(self.fields, 'transporting_velocity',
                                    message=f'transporting velocity, outer iteration {outer}')
                self.transport_fields(outer, xstar, xp)

            # Fast physics -----------------------------------------------------
            x_after_fast(self.field_name).assign(xp(self.field_name))
            if len(self.fast_physics_schemes) > 0:
                with timed_stage("Fast physics"):
                    logger.info(f'Semi-implicit Quasi Newton: Fast physics {outer}')
                    for _, scheme in self.fast_physics_schemes:
                        scheme.apply(x_after_fast(scheme.field_name), x_after_fast(scheme.field_name))

            xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve
            xrhs_phys.assign(x_after_fast(self.field_name) - xp(self.field_name))

            for inner in range(self.num_inner):

                # Implicit forcing ---------------------------------------------
                with timed_stage("Apply forcing terms"):
                    logger.info(f'Semi-implicit Quasi Newton: Implicit forcing {(outer, inner)}')
                    self.forcing.apply(xp, xnp1, xrhs, "implicit")
                    if (inner > 0 and self.accelerator):
                        # Zero implicit forcing to accelerate solver convergence
                        self.forcing.zero_forcing_terms(self.equation, xp, xrhs, self.transported_fields)

                xrhs -= xnp1(self.field_name)
                xrhs += xrhs_phys

                # Linear solve -------------------------------------------------
                with timed_stage("Implicit solve"):
                    logger.info(f'Semi-implicit Quasi Newton: Mixed solve {(outer, inner)}')
                    self.linear_solver.solve(xrhs, dy)  # solves linear system and places result in dy

                xnp1X = xnp1(self.field_name)
                xnp1X += dy

            # Update xnp1 values for active tracers not included in the linear solve
            self.copy_active_tracers(x_after_fast, xnp1)

            self._apply_bcs()

        for name, scheme in self.auxiliary_schemes:
            # transports a field from xn and puts result in xnp1
            logger.debug(f"Semi-implicit Quasi-Newton auxiliary scheme for {name}")
            scheme.apply(xnp1(name), xn(name))

        with timed_stage("Diffusion"):
            for name, scheme in self.diffusion_schemes:
                logger.debug(f"Semi-implicit Quasi-Newton diffusing {name}")
                scheme.apply(xnp1(name), xnp1(name))

        if len(self.final_physics_schemes) > 0:
            with timed_stage("Final Physics"):
                for _, scheme in self.final_physics_schemes:
                    scheme.apply(xnp1(scheme.field_name), xnp1(scheme.field_name))

        logger.debug("Leaving Semi-implicit Quasi-Newton timestep method")

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
                'Reference profiles for must be initialised to use Semi-Implicit Timestepper'

        if not pick_up and self.reference_update_freq is not None:
            # Force reference profiles to be updated on first time step
            self.last_ref_update_time = float(t) - float(self.dt)

        elif not pick_up or (pick_up and self.reference_update_freq is None):
            # Indicate that linear solver profile needs updating
            self.to_update_ref_profile = True

        super().run(t, tmax, pick_up=pick_up)


class Forcing(object):
    """
    Discretises forcing terms for the Semi-Implicit Quasi-Newton timestepper.

    This class describes the evaluation of forcing terms, for instance the
    gravitational force, the Coriolis force or the pressure gradient force.
    These are terms that can simply be evaluated, generally as part of some
    semi-implicit time discretisation.
    """

    def __init__(self, equation, alpha):
        """
        Args:
            equation (:class:`PrognosticEquationSet`): the prognostic equations
                containing the forcing terms.
            alpha (:class:`Constant`): semi-implicit off-centering factor. An
                alpha of 0 corresponds to fully explicit, while a factor of 1
                corresponds to fully implicit.
        """

        self.field_name = equation.field_name
        implicit_terms = [incompressible, sponge]
        dt = equation.domain.dt

        W = equation.function_space
        self.x0 = Function(W)
        self.xF = Function(W)

        # set up boundary conditions on the u subspace of W
        bcs = [DirichletBC(W.sub(0), bc.function_arg, bc.sub_domain) for bc in equation.bcs['u']]

        # drop terms relating to transport, diffusion and physics
        residual = equation.residual.label_map(
            lambda t: any(t.has_label(transport, diffusion, physics_label,
                                      return_tuple=True)), drop)

        # the lhs of both of the explicit and implicit solvers is just
        # the time derivative form
        trials = TrialFunctions(W)
        a = residual.label_map(lambda t: t.has_label(time_derivative),
                               replace_subject(trials),
                               map_if_false=drop)

        # the explicit forms are multiplied by (1-alpha) and moved to the rhs
        one_minus_alpha = Function(alpha.function_space(), val=1-alpha)
        L_explicit = -one_minus_alpha*dt*residual.label_map(
            lambda t:
                any(t.has_label(time_derivative, hydrostatic, *implicit_terms,
                                return_tuple=True)),
            drop,
            replace_subject(self.x0))

        # the implicit forms are multiplied by alpha and moved to the rhs
        L_implicit = -alpha*dt*residual.label_map(
            lambda t:
                any(t.has_label(
                    time_derivative, hydrostatic, *implicit_terms,
                    return_tuple=True)),
            drop,
            replace_subject(self.x0))

        # now add the terms that are always fully implicit
        L_implicit -= dt*residual.label_map(
            lambda t: any(t.has_label(*implicit_terms, return_tuple=True)),
            replace_subject(self.x0),
            drop)

        # the hydrostatic equations require some additional forms:
        if any([t.has_label(hydrostatic) for t in residual]):
            L_explicit += residual.label_map(
                lambda t: t.has_label(hydrostatic),
                replace_subject(self.x0),
                drop)

            L_implicit -= residual.label_map(
                lambda t: t.has_label(hydrostatic),
                replace_subject(self.x0),
                drop)

        # now we can set up the explicit and implicit problems
        explicit_forcing_problem = LinearVariationalProblem(
            a.form, L_explicit.form, self.xF, bcs=bcs,
            constant_jacobian=True
        )

        implicit_forcing_problem = LinearVariationalProblem(
            a.form, L_implicit.form, self.xF, bcs=bcs,
            constant_jacobian=True
        )

        self.solver_parameters = mass_parameters(W, equation.domain.spaces)

        self.solvers = {}
        self.solvers["explicit"] = LinearVariationalSolver(
            explicit_forcing_problem,
            solver_parameters=self.solver_parameters,
            options_prefix="ExplicitForcingSolver"
        )
        self.solvers["implicit"] = LinearVariationalSolver(
            implicit_forcing_problem,
            solver_parameters=self.solver_parameters,
            options_prefix="ImplicitForcingSolver"
        )

        if logger.isEnabledFor(DEBUG):
            self.solvers["explicit"].snes.ksp.setMonitor(logging_ksp_monitor_true_residual)
            self.solvers["implicit"].snes.ksp.setMonitor(logging_ksp_monitor_true_residual)

    def apply(self, x_in, x_nl, x_out, label):
        """
        Applies the discretisation for a forcing term F(x).

        This takes x_in and x_nl and computes F(x_nl), and updates x_out to   \n
        x_out = x_in + scale*F(x_nl)                                          \n
        where 'scale' is the appropriate semi-implicit factor.

        Args:
            x_in (:class:`FieldCreator`): the field to be incremented.
            x_nl (:class:`FieldCreator`): the field which the forcing term is
                evaluated on.
            x_out (:class:`FieldCreator`): the output field to be updated.
            label (str): denotes which forcing to apply. Should be 'explicit' or
                'implicit'. TODO: there should be a check on this. Or this
                should be an actual label.
        """

        self.x0.assign(x_nl(self.field_name))

        self.solvers[label].solve()  # places forcing in self.xF

        x_out.assign(x_in(self.field_name))
        x_out += self.xF

    def zero_forcing_terms(self, equation, x_in, x_out, transported_field_names):
        """
        Zero forcing term F(x) for non-wind transport.

        This takes x_in and x_out, where                                      \n
        x_out = x_in + scale*F(x_nl)                                          \n
        for some field x_nl and sets x_out = x_in for all non-wind transport terms

        Args:
            equation (:class:`PrognosticEquationSet`): the prognostic
                equation set to be solved
            x_in (:class:`FieldCreator`): the field to be incremented.
            x_out (:class:`FieldCreator`): the output field to be updated.
            transported_field_names (str): list of fields names for transported fields
        """
        for field_name in transported_field_names:
            if field_name != 'u':
                logger.info(f'Semi-Implicit Quasi Newton: Zeroing implicit forcing for {field_name}')
                field_index = equation.field_names.index(field_name)
                x_out.subfunctions[field_index].assign(x_in(field_name))
