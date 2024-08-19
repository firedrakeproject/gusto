"""
The Semi-Implicit Quasi-Newton timestepper used by the Met Office's ENDGame
and GungHo dynamical cores.
"""

from firedrake import (Function, Constant, TrialFunctions, DirichletBC,
                       LinearVariationalProblem, LinearVariationalSolver)
from firedrake.fml import drop, replace_subject
from pyop2.profiling import timed_stage
from gusto.core import TimeLevelFields, StateFields
from gusto.core.labels import (transport, diffusion, time_derivative,
                               linearisation, prognostic, hydrostatic,
                               physics_label, sponge, incompressible)
from gusto.solvers import LinearTimesteppingSolver
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
                 num_outer=2, num_inner=2, accelerator=False):

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
            accelerator (bool, optional): Whether to zero non-wind implicit forcings
                for transport terms in order to speed up solver convergence
        """

        self.num_outer = num_outer
        self.num_inner = num_inner
        self.alpha = alpha

        # default is to not offcentre transporting velocity but if it
        # is offcentred then use the same value as alpha
        self.alpha_u = Constant(alpha) if off_centred_u else Constant(0.5)

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
            assert scheme.field_name in equation_set.field_names
            self.active_transport.append((scheme.field_name, scheme))
            self.transported_fields.append(scheme.field_name)
            # Check that there is a corresponding transport method
            method_found = False
            for method in spatial_methods:
                if scheme.field_name == method.variable and method.term_label == transport:
                    method_found = True
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
        for name in equation_set.field_names:
            # Extract time derivative for that prognostic
            mass_form = equation_set.residual.label_map(
                lambda t: (t.has_label(time_derivative) and t.get(prognostic) == name),
                map_if_false=drop)
            # Copy over field if the time derivative term has no linearisation
            if not mass_form.terms[0].has_label(linearisation):
                self.tracers_to_copy.append(name)

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
        self.accelerator = accelerator

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

        x_after_slow(self.field_name).assign(xn(self.field_name))
        if len(self.slow_physics_schemes) > 0:
            with timed_stage("Slow physics"):
                logger.info('Semi-implicit Quasi Newton: Slow physics')
                for _, scheme in self.slow_physics_schemes:
                    scheme.apply(x_after_slow(scheme.field_name), x_after_slow(scheme.field_name))

        with timed_stage("Apply forcing terms"):
            logger.info('Semi-implicit Quasi Newton: Explicit forcing')
            # Put explicit forcing into xstar
            self.forcing.apply(x_after_slow, xn, xstar(self.field_name), "explicit")

        # set xp here so that variables that are not transported have
        # the correct values
        xp(self.field_name).assign(xstar(self.field_name))

        for outer in range(self.num_outer):

            with timed_stage("Transport"):
                self.io.log_courant(self.fields, 'transporting_velocity',
                                    message=f'transporting velocity, outer iteration {outer}')
                for name, scheme in self.active_transport:
                    logger.info(f'Semi-implicit Quasi Newton: Transport {outer}: {name}')
                    # transports a field from xstar and puts result in xp
                    scheme.apply(xp(name), xstar(name))

            x_after_fast(self.field_name).assign(xp(self.field_name))
            if len(self.fast_physics_schemes) > 0:
                with timed_stage("Fast physics"):
                    logger.info(f'Semi-implicit Quasi Newton: Fast physics {outer}')
                    for _, scheme in self.fast_physics_schemes:
                        scheme.apply(x_after_fast(scheme.field_name), x_after_fast(scheme.field_name))

            xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve
            xrhs_phys.assign(x_after_fast(self.field_name) - xp(self.field_name))

            for inner in range(self.num_inner):

                # TODO: this is where to update the reference state

                with timed_stage("Apply forcing terms"):
                    logger.info(f'Semi-implicit Quasi Newton: Implicit forcing {(outer, inner)}')
                    self.forcing.apply(xp, xnp1, xrhs, "implicit")
                    if (inner > 0 and self.accelerator):
                        # Zero implicit forcing to accelerate solver convergence
                        self.forcing.zero_forcing_terms(self.equation, xp, xrhs, self.transported_fields)

                xrhs -= xnp1(self.field_name)
                xrhs += xrhs_phys

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

        if not pick_up:
            assert self.reference_profiles_initialised, \
                'Reference profiles for must be initialised to use Semi-Implicit Timestepper'

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
        L_explicit = -(1-alpha)*dt*residual.label_map(
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
            a.form, L_explicit.form, self.xF, bcs=bcs
        )

        implicit_forcing_problem = LinearVariationalProblem(
            a.form, L_implicit.form, self.xF, bcs=bcs
        )

        self.solvers = {}
        self.solvers["explicit"] = LinearVariationalSolver(
            explicit_forcing_problem,
            options_prefix="ExplicitForcingSolver"
        )
        self.solvers["implicit"] = LinearVariationalSolver(
            implicit_forcing_problem,
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
