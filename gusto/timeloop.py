"""Classes for controlling the timestepping loop."""

from abc import ABCMeta, abstractmethod, abstractproperty
from firedrake import Function, Projector, Constant, split
from pyop2.profiling import timed_stage
from gusto.configuration import logger
from gusto.equations import PrognosticEquationSet
from gusto.forcing import Forcing
from gusto.fml.form_manipulation_labelling import drop, Label, Term
from gusto.labels import (transport, diffusion, time_derivative, linearisation,
                          prognostic, physics, transporting_velocity)
from gusto.linear_solvers import LinearTimesteppingSolver
from gusto.fields import TimeLevelFields, StateFields
from gusto.time_discretisation import ExplicitTimeDiscretisation
from gusto.transport_methods import TransportMethod
import ufl

__all__ = ["Timestepper", "SplitPhysicsTimestepper", "SemiImplicitQuasiNewton",
           "PrescribedTransport"]


class BaseTimestepper(object, metaclass=ABCMeta):
    """Base class for timesteppers."""

    def __init__(self, equation, io):
        """
        Args:
            equation (:class:`PrognosticEquation`): the prognostic equation.
            io (:class:`IO`): the model's object for controlling input/output.
        """

        self.equation = equation
        self.io = io
        self.dt = self.equation.domain.dt
        self.t = Constant(0.0)
        self.reference_profiles_initialised = False

        self.setup_fields()
        self.setup_scheme()

        self.io.log_parameters(equation)

    @abstractproperty
    def transporting_velocity(self):
        return NotImplementedError

    @abstractmethod
    def setup_fields(self):
        """Set up required fields. Must be implemented in child classes"""
        pass

    @abstractmethod
    def setup_scheme(self):
        """Set up required scheme(s). Must be implemented in child classes"""
        pass

    @abstractmethod
    def timestep(self):
        """Defines the timestep. Must be implemented in child classes"""
        return NotImplementedError

    def set_initial_timesteps(self, num_steps):
        """Sets the number of initial time steps for a multi-level scheme."""
        can_set = (hasattr(self, 'scheme')
                   and hasattr(self.scheme, 'initial_timesteps')
                   and num_steps is not None)
        if can_set:
            self.scheme.initial_timesteps = num_steps

    def get_initial_timesteps(self):
        """Gets the number of initial time steps from a multi-level scheme."""
        can_get = (hasattr(self, 'scheme')
                   and hasattr(self.scheme, 'initial_timesteps'))
        # Return None if this is not applicable
        return self.scheme.initial_timesteps if can_get else None

    def setup_equation(self, equation):
        """
        Sets up the spatial methods for an equation, by the setting the
        forms used for transport/diffusion in the equation.

        Args:
            equation (:class:`PrognosticEquation`): the equation that the
                transport method is to be applied to.
        """

        # For now, we only have methods for transport and diffusion
        for term_label in [transport, diffusion]:
            # ---------------------------------------------------------------- #
            # Check that appropriate methods have been provided
            # ---------------------------------------------------------------- #
            # Extract all terms corresponding to this type of term
            residual = equation.residual.label_map(
                lambda t: t.has_label(term_label), map_if_false=drop
            )
            active_variables = [t.get(prognostic) for t in residual.terms]
            active_methods = list(filter(lambda t: t.term_label == term_label,
                                         self.spatial_methods))
            method_variables = [method.variable for method in active_methods]
            for variable in active_variables:
                if variable not in method_variables:
                    message = f'Variable {variable} has a {term_label.label} ' \
                        + 'but no method for this has been specified. Using ' \
                        + 'default form for this term'
                    logger.warning(message)

        # -------------------------------------------------------------------- #
        # Check that appropriate methods have been provided
        # -------------------------------------------------------------------- #
        # Replace forms in equation
        if self.spatial_methods is not None:
            for method in self.spatial_methods:
                method.replace_form(equation)

    def setup_transporting_velocity(self, scheme):
        """
        Set up the time discretisation by replacing the transporting velocity
        used by the appropriate one for this time loop.

        Args:
            scheme (:class:`TimeDiscretisation`): the time discretisation whose
                transport term should be replaced with the transport term of
                this discretisation.
        """

        if self.transporting_velocity == "prognostic" and "u" in self.fields._field_names:
            # Use the prognostic wind variable as the transporting velocity
            u_idx = self.equation.field_names.index('u')
            uadv = split(self.equation.X)[u_idx]
        else:
            uadv = self.transporting_velocity

        scheme.residual = scheme.residual.label_map(
            lambda t: t.has_label(transporting_velocity),
            map_if_true=lambda t:
            Term(ufl.replace(t.form, {t.get(transporting_velocity): uadv}), t.labels)
        )

        scheme.residual = transporting_velocity.update_value(scheme.residual, uadv)

    def run(self, t, tmax, pick_up=False):
        """
        Runs the model for the specified time, from t to tmax

        Args:
            t (float): the start time of the run
            tmax (float): the end time of the run
            pick_up: (bool): specify whether to pick_up from a previous run
        """

        # Set up diagnostics, which may set up some fields necessary to pick up
        self.io.setup_diagnostics(self.fields)

        if pick_up:
            # Pick up fields, and return other info to be picked up
            t, reference_profiles, initial_timesteps = self.io.pick_up_from_checkpoint(self.fields)
            self.set_reference_profiles(reference_profiles)
            self.set_initial_timesteps(initial_timesteps)

        # Set up dump, which may also include an initial dump
        with timed_stage("Dump output"):
            self.io.setup_dump(self.fields, t, pick_up)

        self.t.assign(t)

        # Time loop
        while float(self.t) < tmax - 0.5*float(self.dt):
            logger.info(f'at start of timestep, t={float(self.t)}, dt={float(self.dt)}')

            self.x.update()

            self.timestep()

            self.t.assign(self.t + self.dt)

            with timed_stage("Dump output"):
                self.io.dump(self.fields, float(self.t), self.get_initial_timesteps())

        if self.io.output.checkpoint and self.io.output.checkpoint_method == 'dumbcheckpoint':
            self.io.chkpt.close()

        logger.info(f'TIMELOOP complete. t={float(self.t)}, tmax={tmax}')

    def set_reference_profiles(self, reference_profiles):
        """
        Initialise the model's reference profiles.

        reference_profiles (list): an iterable of pairs: (field_name, expr),
            where 'field_name' is the string giving the name of the reference
            profile field expr is the :class:`ufl.Expr` whose value is used to
            set the reference field.
        """
        for field_name, profile in reference_profiles:
            if field_name+'_bar' in self.fields:
                # For reference profiles already added to state, allow
                # interpolation from expressions
                ref = self.fields(field_name+'_bar')
            elif isinstance(profile, Function):
                # Need to add reference profile to state so profile must be
                # a Function
                ref = self.fields(field_name+'_bar', space=profile.function_space(),
                                  pick_up=True, dump=False, field_type='reference')
            else:
                raise ValueError(f'When initialising reference profile {field_name}'
                                 + ' the passed profile must be a Function')
            ref.interpolate(profile)

            # Assign profile to X_ref belonging to equation
            if isinstance(self.equation, PrognosticEquationSet):
                assert field_name in self.equation.field_names, \
                    f'Cannot set reference profile as field {field_name} not found'
                idx = self.equation.field_names.index(field_name)
                X_ref = self.equation.X_ref.split()[idx]
                X_ref.assign(ref)

        self.reference_profiles_initialised = True


class Timestepper(BaseTimestepper):
    """
    Implements a timeloop by applying a scheme to a prognostic equation.
    """

    def __init__(self, equation, scheme, io, spatial_methods=None):
        """
        Args:
            equation (:class:`PrognosticEquation`): the prognostic equation
            scheme (:class:`TimeDiscretisation`): the scheme to use to timestep
                the prognostic equation
            io (:class:`IO`): the model's object for controlling input/output.
            spatial_methods (iter,optional): a list of objects describing the
                methods to use for discretising transport or diffusion terms
                for each transported/diffused variable. Defaults to None,
                in which case the terms follow the original discretisation in
                the equation.
        """
        self.scheme = scheme
        if spatial_methods is not None:
            self.spatial_methods = spatial_methods
        else:
            self.spatial_methods = []

        super().__init__(equation=equation, io=io)

    @property
    def transporting_velocity(self):
        return "prognostic"

    def setup_fields(self):
        self.x = TimeLevelFields(self.equation, self.scheme.nlevels)
        self.fields = StateFields(self.x, self.equation.prescribed_fields,
                                  *self.io.output.dumplist)

    def setup_scheme(self):
        self.setup_equation(self.equation)
        self.scheme.setup(self.equation)
        self.setup_transporting_velocity(self.scheme)

    def timestep(self):
        """
        Implement the timestep
        """
        xnp1 = self.x.np1
        name = self.equation.field_name
        x_in = [x(name) for x in self.x.previous[-self.scheme.nlevels:]]

        self.scheme.apply(xnp1(name), *x_in)


class SplitPhysicsTimestepper(Timestepper):
    """
    Implements a timeloop by applying schemes separately to the physics and
    dynamics. This 'splits' the physics from the dynamics and allows a different
    scheme to be applied to the physics terms than the prognostic equation.
    """

    def __init__(self, equation, scheme, io, spatial_methods=None,
                 physics_schemes=None):
        """
        Args:
            equation (:class:`PrognosticEquation`): the prognostic equation
            scheme (:class:`TimeDiscretisation`): the scheme to use to timestep
                the prognostic equation
            io (:class:`IO`): the model's object for controlling input/output.
            spatial_methods (iter,optional): a list of objects describing the
                methods to use for discretising transport or diffusion terms
                for each transported/diffused variable. Defaults to None,
                in which case the terms follow the original discretisation in
                the equation.
            physics_schemes: (list, optional): a list of :class:`Physics` and
                :class:`TimeDiscretisation` options describing physical
                parametrisations and timestepping schemes to use for each.
                Timestepping schemes for physics must be explicit. Defaults to
                None.
        """

        super().__init__(equation, scheme, io, spatial_methods=spatial_methods)

        if physics_schemes is not None:
            self.physics_schemes = physics_schemes
        else:
            self.physics_schemes = []

        for _, phys_scheme in self.physics_schemes:
            # check that the supplied schemes for physics are explicit
            assert isinstance(phys_scheme, ExplicitTimeDiscretisation), "Only explicit schemes can be used for physics"
            apply_bcs = False
            phys_scheme.setup(equation, apply_bcs, physics)

    @property
    def transporting_velocity(self):
        return "prognostic"

    def setup_scheme(self):
        self.setup_equation(self.equation)
        # Go through and label all non-physics terms with a "dynamics" label
        dynamics = Label('dynamics')
        self.equation.label_terms(lambda t: not any(t.has_label(time_derivative, physics)), dynamics)
        apply_bcs = True
        self.scheme.setup(self.equation, apply_bcs, dynamics)
        self.setup_transporting_velocity(self.scheme)

    def timestep(self):

        super().timestep()

        with timed_stage("Physics"):
            for _, scheme in self.physics_schemes:
                scheme.apply(self.x.np1(scheme.field_name), self.x.np1(scheme.field_name))


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
                 diffusion_schemes=None, physics_schemes=None, **kwargs):

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
                ``(field_name, scheme)`` indicating the fields to diffuse, and the
                the :class:`~.TimeDiscretisation` to use. Defaults to None.
            physics_schemes: (list, optional): a list of :class:`Physics` and
                :class:`TimeDiscretisation` options describing physical
                parametrisations and timestepping schemes to use for each.
                Timestepping schemes for physics must be explicit. Defaults to
                None.

        :kwargs: maxk is the number of outer iterations, maxi is the number
            of inner iterations and alpha is the offcentering parameter
    """

        self.maxk = kwargs.pop("maxk", 4)
        self.maxi = kwargs.pop("maxi", 1)
        self.alpha = kwargs.pop("alpha", 0.5)
        if kwargs:
            raise ValueError("unexpected kwargs: %s" % list(kwargs.keys()))

        self.spatial_methods = spatial_methods

        if physics_schemes is not None:
            self.physics_schemes = physics_schemes
        else:
            self.physics_schemes = []
        for _, scheme in self.physics_schemes:
            assert scheme.nlevels == 1, "multilevel schemes not supported as part of this timestepping loop"
            assert isinstance(scheme, ExplicitTimeDiscretisation), "Only explicit schemes can be used for physics"

        self.active_transport = []
        for scheme in transport_schemes:
            assert scheme.nlevels == 1, "multilevel schemes not supported as part of this timestepping loop"
            assert scheme.field_name in equation_set.field_names
            self.active_transport.append((scheme.field_name, scheme))
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
        self.dy = Function(W)
        if linear_solver is None:
            self.linear_solver = LinearTimesteppingSolver(equation_set, self.alpha)
        else:
            self.linear_solver = linear_solver
        self.forcing = Forcing(equation_set, self.alpha)
        self.bcs = equation_set.bcs

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
        return xn('u') + self.alpha*(xnp1('u')-xn('u'))

    def setup_fields(self):
        """Sets up time levels n, star, p and np1"""
        self.x = TimeLevelFields(self.equation, 1)
        self.x.add_fields(self.equation, levels=("star", "p"))
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

        apply_bcs = True
        for _, scheme in self.diffusion_schemes:
            scheme.setup(self.equation, apply_bcs, diffusion)
        for _, scheme in self.physics_schemes:
            apply_bcs = True
            scheme.setup(self.equation, apply_bcs, physics)

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
        xrhs = self.xrhs
        dy = self.dy

        with timed_stage("Apply forcing terms"):
            self.forcing.apply(xn, xn, xstar(self.field_name), "explicit")

        xp(self.field_name).assign(xstar(self.field_name))

        for k in range(self.maxk):

            with timed_stage("Transport"):
                for name, scheme in self.active_transport:
                    # transports a field from xstar and puts result in xp
                    scheme.apply(xp(name), xstar(name))

            xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve

            for i in range(self.maxi):

                with timed_stage("Apply forcing terms"):
                    self.forcing.apply(xp, xnp1, xrhs, "implicit")

                xrhs -= xnp1(self.field_name)

                with timed_stage("Implicit solve"):
                    self.linear_solver.solve(xrhs, dy)  # solves linear system and places result in dy

                xnp1X = xnp1(self.field_name)
                xnp1X += dy

            # Update xnp1 values for active tracers not included in the linear solve
            self.copy_active_tracers(xp, xnp1)

            self._apply_bcs()

        for name, scheme in self.auxiliary_schemes:
            # transports a field from xn and puts result in xnp1
            scheme.apply(xnp1(name), xn(name))

        with timed_stage("Diffusion"):
            for name, scheme in self.diffusion_schemes:
                scheme.apply(xnp1(name), xnp1(name))

        with timed_stage("Physics"):
            for _, scheme in self.physics_schemes:
                scheme.apply(xnp1(scheme.field_name), xnp1(scheme.field_name))

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


class PrescribedTransport(Timestepper):
    """
    Implements a timeloop with a prescibed transporting velocity.
    """
    def __init__(self, equation, scheme, io, transport_method,
                 physics_schemes=None, prescribed_transporting_velocity=None):
        """
        Args:
            equation (:class:`PrognosticEquation`): the prognostic equation
            scheme (:class:`TimeDiscretisation`): the scheme to use to timestep
                the prognostic equation.
            transport_method (:class:`TransportMethod`): describes the method
                used for discretising the transport term.
            io (:class:`IO`): the model's object for controlling input/output.
            physics_schemes: (list, optional): a list of :class:`Physics` and
                :class:`TimeDiscretisation` options describing physical
                parametrisations and timestepping schemes to use for each.
                Timestepping schemes for physics must be explicit. Defaults to
                None.
            prescribed_transporting_velocity (func, optional): a function,
                with a single argument representing the time, that returns a
                :class:`ufl.Expr` for the transporting velocity. This allows
                the transporting velocity field to be updated with time. If
                `None` is provided then the equation's velocity field is not
                updated. Defaults to None.
        """

        if isinstance(transport_method, TransportMethod):
            transport_methods = [transport_method]
        else:
            # Assume an iterable has been provided
            transport_methods = transport_method

        super().__init__(equation, scheme, io, spatial_methods=transport_methods)

        if physics_schemes is not None:
            self.physics_schemes = physics_schemes
        else:
            self.physics_schemes = []

        for _, scheme in self.physics_schemes:
            # check that the supplied schemes for physics are explicit
            assert isinstance(scheme, ExplicitTimeDiscretisation), "Only explicit schemes can be used for physics"
            apply_bcs = False
            scheme.setup(equation, apply_bcs, physics)

        if prescribed_transporting_velocity is not None:
            self.velocity_projection = Projector(
                prescribed_transporting_velocity(self.t),
                self.fields('u'))
        else:
            self.velocity_projection = None

    @property
    def transporting_velocity(self):
        return self.fields('u')

    def setup_fields(self):
        self.x = TimeLevelFields(self.equation, self.scheme.nlevels)
        self.fields = StateFields(self.x, self.equation.prescribed_fields,
                                  *self.io.output.dumplist)

    def timestep(self):
        if self.velocity_projection is not None:
            self.velocity_projection.project()

        super().timestep()

        with timed_stage("Physics"):
            for _, scheme in self.physics_schemes:
                scheme.apply(self.x.np1(scheme.field_name), self.x.np1(scheme.field_name))
