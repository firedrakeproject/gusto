"""Defines the basic timestepper objects."""

from abc import ABCMeta, abstractmethod, abstractproperty
from firedrake import Function, Projector, split
from firedrake.fml import drop, Term, all_terms
from pyop2.profiling import timed_stage
from gusto.equations import PrognosticEquationSet
from gusto.core import TimeLevelFields, StateFields
from gusto.core.labels import transport, diffusion, prognostic, transporting_velocity
from gusto.core.logging import logger
from gusto.time_discretisation.time_discretisation import ExplicitTimeDiscretisation
from gusto.spatial_methods.transport_methods import TransportMethod
import ufl


__all__ = ["BaseTimestepper", "Timestepper", "PrescribedTransport"]


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
        self.t = self.equation.domain.t
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
            variables = [t.get(prognostic) for t in residual.terms]
            methods = list(filter(lambda t: t.term_label == term_label,
                                  self.spatial_methods))
            method_variables = [method.variable for method in methods]
            for variable in variables:
                if variable not in method_variables:
                    message = f'Variable {variable} has a {term_label.label} ' \
                        + 'term but no method for this has been specified. ' \
                        + 'Using default form for this term'
                    logger.warning(message)

        # -------------------------------------------------------------------- #
        # Check that appropriate methods have been provided
        # -------------------------------------------------------------------- #
        # Replace forms in equation
        if self.spatial_methods is not None:
            for method in self.spatial_methods:
                method.replace_form(equation)

        # -------------------------------------------------------------------- #
        # Ensure the quadrature degree is not excessive for any integrals
        # -------------------------------------------------------------------- #
        def update_quadrature_degree(t):
            # This function takes in a term and returns a new term
            # with the same form and labels, the only difference being
            # that any integrals in the form with no quadrature_degree
            # set will have their quadrature degree set to max_quad_deg

            # This list will hold the updated integrals
            new_itgs = []

            # Loop over integrals in this term's form
            for itg in t.form._integrals:
                # check if the quadrature degree is not set
                if 'quadrature_degree' not in itg._metadata.keys():
                    # create new integral with updated quadrature degree
                    new_itg = itg.reconstruct(
                        metadata={'quadrature_degree': equation.max_quad_deg})
                    new_itgs.append(new_itg)
                else:
                    # if quadrature degree was already set, just keep
                    # this integral
                    new_itgs.append(itg)

            new_form = ufl.form.Form(new_itgs)

            return Term(new_form, t.labels)

        equation.residual = equation.residual.label_map(
            all_terms,
            lambda t: update_quadrature_degree(t))

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

    def log_timestep(self):
        """
        Logs the start of a time step.
        """
        logger.info('')
        logger.info('='*40)
        logger.info(f'at start of timestep {self.step}, t={float(self.t)}, dt={float(self.dt)}')

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
        self.io.setup_log_courant(self.fields)
        if self.equation.domain.mesh.extruded:
            self.io.setup_log_courant(self.fields, component='horizontal')
            self.io.setup_log_courant(self.fields, component='vertical')
        if self.transporting_velocity != "prognostic":
            self.io.setup_log_courant(self.fields, name='transporting_velocity',
                                      expression=self.transporting_velocity)

        if pick_up:
            # Pick up fields, and return other info to be picked up
            t, reference_profiles, self.step, initial_timesteps = self.io.pick_up_from_checkpoint(self.fields)
            self.set_reference_profiles(reference_profiles)
            self.set_initial_timesteps(initial_timesteps)
        else:
            self.step = 1

        # Set up dump, which may also include an initial dump
        with timed_stage("Dump output"):
            self.io.setup_dump(self.fields, t, pick_up)

        self.t.assign(t)

        # Time loop
        while float(self.t) < tmax - 0.5*float(self.dt):
            self.log_timestep()

            self.x.update()

            self.io.log_courant(self.fields)
            if self.equation.domain.mesh.extruded:
                self.io.log_courant(self.fields, component='horizontal', message='horizontal')
                self.io.log_courant(self.fields, component='vertical', message='vertical')

            self.timestep()

            self.t.assign(self.t + self.dt)
            self.step += 1

            with timed_stage("Dump output"):
                self.io.dump(self.fields, float(self.t), self.step, self.get_initial_timesteps())

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
            # if field name is not prognostic we need to add it
            ref.interpolate(profile)
            # Assign profile to X_ref belonging to equation
            if isinstance(self.equation, PrognosticEquationSet):
                if field_name in self.equation.field_names:
                    idx = self.equation.field_names.index(field_name)
                    X_ref = self.equation.X_ref.subfunctions[idx]
                    X_ref.assign(ref)
                else:
                    # reference profile of a diagnostic
                    # warn user in case they made a typo
                    logger.warning(f'Setting reference profile for diagnostic {field_name}')
                    # Don't need to do anything else as value in field container has already been set
        self.reference_profiles_initialised = True


class Timestepper(BaseTimestepper):
    """
    Implements a timeloop by applying a scheme to a prognostic equation.
    """

    def __init__(self, equation, scheme, io, spatial_methods=None,
                 physics_parametrisations=None):
        """
        Args:
            equation (:class:`PrognosticEquation`): the prognostic equation
            scheme (:class:`TimeDiscretisation`): the scheme to use to timestep
                the prognostic equation
            io (:class:`IO`): the model's object for controlling input/output.
            spatial_methods (iter, optional): a list of objects describing the
                methods to use for discretising transport or diffusion terms
                for each transported/diffused variable. Defaults to None,
                in which case the terms follow the original discretisation in
                the equation.
            physics_parametrisations: (iter, optional): an iterable of
                :class:`PhysicsParametrisation` objects that describe physical
                parametrisations to be included to add to the equation. They can
                only be used when the time discretisation `scheme` is explicit.
                Defaults to None.
        """
        self.scheme = scheme
        if spatial_methods is not None:
            self.spatial_methods = spatial_methods
        else:
            self.spatial_methods = []

        if physics_parametrisations is not None:
            self.physics_parametrisations = physics_parametrisations
            if len(self.physics_parametrisations) > 1:
                assert isinstance(scheme, ExplicitTimeDiscretisation), \
                    ('Physics parametrisations can only be used with the '
                     + 'basic TimeStepper when the time discretisation is '
                     + 'explicit. If you want to use an implicit scheme, the '
                     + 'SplitPhysicsTimestepper is more appropriate.')
        else:
            self.physics_parametrisations = []

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
        if self.io.output.log_courant:
            self.scheme.courant_max = self.io.courant_max

    def timestep(self):
        """
        Implement the timestep
        """
        xnp1 = self.x.np1
        name = self.equation.field_name
        x_in = [x(name) for x in self.x.previous[-self.scheme.nlevels:]]

        self.scheme.apply(xnp1(name), *x_in)


class PrescribedTransport(Timestepper):
    """
    Implements a timeloop with a prescibed transporting velocity.
    """
    def __init__(self, equation, scheme, io, transport_method,
                 physics_parametrisations=None,
                 prescribed_transporting_velocity=None):
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
            physics_parametrisations: (iter, optional): an iterable of
                :class:`PhysicsParametrisation` objects that describe physical
                parametrisations to be included to add to the equation. They can
                only be used when the time discretisation `scheme` is explicit.
                Defaults to None.
        """

        if isinstance(transport_method, TransportMethod):
            transport_methods = [transport_method]
        else:
            # Assume an iterable has been provided
            transport_methods = transport_method

        super().__init__(equation, scheme, io, spatial_methods=transport_methods,
                         physics_parametrisations=physics_parametrisations)

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

    def run(self, t, tmax, pick_up=False):
        """
        Runs the model for the specified time, from t to tmax
        Args:
            t (float): the start time of the run
            tmax (float): the end time of the run
            pick_up: (bool): specify whether to pick_up from a previous run
        """
        # It's best to have evaluated the velocity before we start
        if self.velocity_projection is not None:
            self.velocity_projection.project()

        super().run(t, tmax, pick_up=pick_up)

    def timestep(self):
        if self.velocity_projection is not None:
            self.velocity_projection.project()

        super().timestep()
