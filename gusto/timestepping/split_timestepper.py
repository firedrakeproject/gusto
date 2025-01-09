"""Split timestepping methods for generically solving terms separately."""

from firedrake import Projector
from firedrake.fml import Label, drop
from pyop2.profiling import timed_stage
from gusto.core import TimeLevelFields, StateFields
from gusto.core.labels import time_derivative, physics_label, dynamics_label
from gusto.time_discretisation.time_discretisation import ExplicitTimeDiscretisation
from gusto.timestepping.timestepper import BaseTimestepper, Timestepper
from numpy import ones

__all__ = ["SplitTimestepper", "SplitPhysicsTimestepper", "SplitPrescribedTransport"]


class SplitTimestepper(BaseTimestepper):
    """
    Implements a timeloop by applying separate schemes to different terms, e.g, physics
    and individual dynamics components in a user-defined order. This allows a different
    time discretisation to be applied to each defined component. Different terms can be
    substepped by specifying weights; this enables Strang-Splitting to be applied.
    """

    def __init__(self, equation, term_splitting, dynamics_schemes, io,
                 weights=None, spatial_methods=None, physics_schemes=None):
        """
        Args:
            equation (:class:`PrognosticEquation`): the prognostic equation
            term_splitting (list): a list of labels specifying the terms that should
                be solved separately and the order to do so.
            dynamics_schemes: (:class:`TimeDiscretisation`) A list of time
                discretisations for the dynamics schemes. A scheme must be
                provided for each non-physics label that is provided in the
                term_splitting list.
            io (:class:`IO`): the model's object for controlling input/output.
                weights (array, optional): An array of weights for substepping
                of any dynamics or physics scheme. The sum of weights for
                each distinct label in term_splitting must be 1.
            spatial_methods (iter,optional): a list of objects describing the
                methods to use for discretising transport or diffusion terms
                for each transported/diffused variable. Defaults to None,
                in which case the terms follow the original discretisation in
                the equation.
            physics_schemes: (list, optional): a list of tuples of the form
                (:class:`PhysicsParametrisation`, :class:`TimeDiscretisation`),
                pairing physics parametrisations and timestepping schemes to use
                for each. Timestepping schemes for physics must be explicit.
                Defaults to None.
        """

        if spatial_methods is not None:
            self.spatial_methods = spatial_methods
        else:
            self.spatial_methods = []

        # If we have physics schemes, extract these.
        if 'physics' in term_splitting:
            if physics_schemes is None:
                raise ValueError('Physics schemes need to be specified when applying '
                                 + 'a physics splitting in the SplitTimestepper')
            else:
                # Check that the weights are correct for physics:
                count = 0
                if weights is not None:
                    for idx, term in enumerate(term_splitting):
                        if term == 'physics':
                            count += weights[idx]
                    if count != 1:
                        raise ValueError('Incorrect weights are specified for the '
                                         + 'physics schemes in the split timestepper.')
                self.physics_schemes = physics_schemes
        else:
            self.physics_schemes = []

        for parametrisation, phys_scheme in self.physics_schemes:
            # Check that the supplied schemes for physics are valid
            if hasattr(parametrisation, "explicit_only") and parametrisation.explicit_only:
                assert isinstance(phys_scheme, ExplicitTimeDiscretisation), \
                    ("Only explicit time discretisations can be used with "
                     + f"physics scheme {parametrisation.label.label}")

        self.term_splitting = term_splitting
        self.dynamics_schemes = dynamics_schemes

        if weights is not None:
            self.weights = weights
        else:
            self.weights = ones(len(self.term_splitting))

        # Check that each dynamics label in term_splitting has a corresponding
        # dynamics scheme
        for term in self.term_splitting:
            if term != 'physics':
                assert term in self.dynamics_schemes, \
                    f'The {term} terms do not have a specified scheme in the split timestepper'

        # Multilevel schemes are currently not supported for the dynamics terms.
        for label, scheme in self.dynamics_schemes.items():
            assert scheme.nlevels == 1, \
                "Multilevel schemes are not currently implemented in the split timestepper"

        # As we handle physics in separate parametrisations, these are not
        # passed to the super __init__
        super().__init__(equation, io)

        # Check that each dynamics term is specified by a label
        # in the term_splitting list, but also that there are not
        # multiple labels, i.e. there is a single specified time discretisation.
        # When using weights, these should add to 1 for each term.
        terms = self.equation.residual.label_map(
            lambda t: any(t.has_label(time_derivative, physics_label)), map_if_true=drop
        )
        for term in terms:
            count = 0
            for idx, label in enumerate(self.term_splitting):
                if term.has_label(Label(label)):
                    count += self.weights[idx]
            if count != 1:
                raise ValueError('The term_splitting list does not correctly cover '
                                 + 'the dynamics terms in the equation(s).')

        # Timesteps for each scheme in the term_splitting list
        self.split_dts = [self.equation.domain.dt*weight for weight in self.weights]

    @property
    def transporting_velocity(self):
        return self.fields('u')

    def setup_fields(self):
        self.x = TimeLevelFields(self.equation, 1)
        self.fields = StateFields(self.x, self.equation.prescribed_fields,
                                  *self.io.output.dumplist)

    def setup_scheme(self):
        """Sets up transport, diffusion and physics schemes"""
        # TODO: apply_bcs should be False for advection but this means
        # tests with KGOs fail
        self.setup_equation(self.equation)

        apply_bcs = True
        for label, scheme in self.dynamics_schemes.items():
            scheme.setup(self.equation, apply_bcs, Label(label))
            self.setup_transporting_velocity(scheme)
            if self.io.output.log_courant and label == 'transport':
                scheme.courant_max = self.io.courant_max

        apply_bcs = False
        for parametrisation, scheme in self.physics_schemes:
            scheme.setup(self.equation, apply_bcs, parametrisation.label)

    def timestep(self):

        for idx, term in enumerate(self.term_splitting):
            split_dt = self.split_dts[idx]
            if term == 'physics':
                with timed_stage("Physics"):
                    for _, scheme in self.physics_schemes:
                        scheme.dt = split_dt
                        scheme.apply(self.x.np1(scheme.field_name), self.x.np1(scheme.field_name))
            else:
                scheme = self.dynamics_schemes[term]
                scheme.dt = split_dt
                scheme.apply(self.x.np1(scheme.field_name), self.x.np1(scheme.field_name))


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
            physics_schemes: (list, optional): a list of tuples of the form
                (:class:`PhysicsParametrisation`, :class:`TimeDiscretisation`),
                pairing physics parametrisations and timestepping schemes to use
                for each. Timestepping schemes for physics must be explicit.
                Defaults to None.
        """

        # As we handle physics differently to the Timestepper, these are not
        # passed to the super __init__
        super().__init__(equation, scheme, io, spatial_methods=spatial_methods)

        if physics_schemes is not None:
            self.physics_schemes = physics_schemes
        else:
            self.physics_schemes = []

        for parametrisation, phys_scheme in self.physics_schemes:
            # check that the supplied schemes for physics are valid
            if hasattr(parametrisation, "explicit_only") and parametrisation.explicit_only:
                assert isinstance(phys_scheme, ExplicitTimeDiscretisation), \
                    ("Only explicit time discretisations can be used with "
                     + f"physics scheme {parametrisation.label.label}")
            apply_bcs = False
            phys_scheme.setup(equation, apply_bcs, parametrisation.label)

    @property
    def transporting_velocity(self):
        return "prognostic"

    def setup_scheme(self):
        self.setup_equation(self.equation)
        # Go through and label all non-physics terms with a "dynamics" label
        dynamics = Label('dynamics')
        self.equation.label_terms(lambda t: not any(t.has_label(time_derivative, physics_label)), dynamics)
        apply_bcs = True
        self.scheme.setup(self.equation, apply_bcs, dynamics)
        self.setup_transporting_velocity(self.scheme)
        if self.io.output.log_courant:
            self.scheme.courant_max = self.io.courant_max

    def timestep(self):

        super().timestep()

        with timed_stage("Physics"):
            for _, scheme in self.physics_schemes:
                scheme.apply(self.x.np1(scheme.field_name), self.x.np1(scheme.field_name))


class SplitPrescribedTransport(Timestepper):
    """
    Implements a timeloop where the physics terms are solved separately from
    the dynamics, like with SplitPhysicsTimestepper, but here we define
    a prescribed transporting velocity, as opposed to having the
    velocity as a prognostic variable.
    """

    def __init__(self, equation, scheme, io, prescribed_transporting_velocity,
                 spatial_methods=None, physics_schemes=None):
        """
        Args:
            equation (:class:`PrognosticEquation`): the prognostic equation
            scheme (:class:`TimeDiscretisation`): the scheme to use to timestep
                the prognostic equation
            io (:class:`IO`): the model's object for controlling input/output.
            prescribed_transporting_velocity: (bool): Whether a time-varying
                transporting velocity will be defined. If True, this will
                require the transporting velocity to be setup by calling either
                the `setup_prescribed_expr` or `setup_prescribed_apply` methods.
            spatial_methods (iter,optional): a list of objects describing the
                methods to use for discretising transport or diffusion terms
                for each transported/diffused variable. Defaults to None,
                in which case the terms follow the original discretisation in
                the equation.
            physics_schemes: (list, optional): a list of tuples of the form
                (:class:`PhysicsParametrisation`, :class:`TimeDiscretisation`),
                pairing physics parametrisations and timestepping schemes to use
                for each. Timestepping schemes for physics can be explicit
                or implicit. Defaults to None.
            prescribed_transporting_velocity: (field, optional): A known
                velocity field that is used for the transport of tracers.
                This can be made time-varying by defining a python function
                that uses time as an argument.
                Defaults to None.
        """

        # As we handle physics differently to the Timestepper, these are not
        # passed to the super __init__
        super().__init__(equation, scheme, io, spatial_methods=spatial_methods)

        if physics_schemes is not None:
            self.physics_schemes = physics_schemes
        else:
            self.physics_schemes = []

        for parametrisation, phys_scheme in self.physics_schemes:
            # check that the supplied schemes for physics are valid
            if hasattr(parametrisation, "explicit_only") and parametrisation.explicit_only:
                assert isinstance(phys_scheme, ExplicitTimeDiscretisation), \
                    ("Only explicit time discretisations can be used with "
                     + f"physics scheme {parametrisation.label.label}")
            apply_bcs = False
            phys_scheme.setup(equation, apply_bcs, parametrisation.label)

        self.prescribed_transport_velocity = prescribed_transporting_velocity
        self.is_velocity_setup = not self.prescribed_transport_velocity
        self.velocity_projection = None
        self.velocity_apply = None

    @property
    def transporting_velocity(self):
        return self.fields('u')

    def setup_scheme(self):
        print('Setting up base equation')
        self.setup_equation(self.equation)

         # If there is an augmentation, set up the residual now
        if self.scheme.augmentation is not None:
            if self.scheme.augmentation.name == 'mean_mixing_ratio':
                self.scheme.augmentation.setup_residual(self.spatial_methods, self.equation)
                print('Setting up augmented equation')
                # Go through and label all non-physics terms with a "dynamics" label
                dynamics = Label('dynamics')
                self.scheme.augmentation.residual = self.scheme.augmentation.residual.label_map(
                    lambda t: not any(t.has_label(time_derivative, physics_label)),
                    map_if_true=lambda t: dynamics(t)
                )
                print(len(self.scheme.augmentation.residual.label_map(
                    lambda t: t.has_label(dynamics),
                    map_if_false=drop
                )))

        # Go through and label all non-physics terms with a "dynamics" label
        dynamics = Label('dynamics')
        self.equation.label_terms(lambda t: not any(t.has_label(time_derivative, physics_label)), dynamics)
        apply_bcs = True
        self.scheme.setup(self.equation, apply_bcs, dynamics)
        self.setup_transporting_velocity(self.scheme)
        if self.io.output.log_courant:
            self.scheme.courant_max = self.io.courant_max


    def setup_prescribed_expr(self, expr_func):
        """
        Sets up the prescribed transporting velocity, through a python function
        which has time as an argument, and returns a `ufl.Expr`. This allows the
        velocity to be updated with time.

        Args:
            expr_func (func): a python function with a single argument that
                represents the model time, and returns a `ufl.Expr`.
        """

        if self.is_velocity_setup:
            raise RuntimeError('Prescribed velocity already set up!')

        self.velocity_projection = Projector(
            expr_func(self.t), self.fields('u')
        )

        self.is_velocity_setup = True

    def setup_prescribed_apply(self, apply_method):
        """
        Sets up the prescribed transporting velocity, through a python function
        which has time as an argument. This function will perform the evaluation
        of the transporting velocity.

        Args:
            expr_func (func): a python function with a single argument that
                represents the model time, and performs the evaluation of the
                transporting velocity.
        """

        if self.is_velocity_setup:
            raise RuntimeError('Prescribed velocity already set up!')
        self.velocity_apply = apply_method
        self.is_velocity_setup = True

    def run(self, t, tmax, pick_up=False):
        """
        Runs the model for the specified time, from t to tmax
        Args:
            t (float): the start time of the run
            tmax (float): the end time of the run
            pick_up: (bool): specify whether to pick_up from a previous run
        """

        # Throw an error if no transporting velocity has been set up
        if self.prescribed_transport_velocity and not self.is_velocity_setup:
            raise RuntimeError(
                'A time-varying prescribed velocity is required. This must be '
                + 'set up through calling either the setup_prescribed_expr or '
                + 'setup_prescribed_apply routines.')

        # It's best to have evaluated the velocity before we start
        if self.velocity_projection is not None:
            self.velocity_projection.project()
        if self.velocity_apply is not None:
            self.velocity_apply(self.t)

        super().run(t, tmax, pick_up=pick_up)

    def timestep(self):

        if self.velocity_projection is not None:
            self.velocity_projection.project()
        if self.velocity_apply is not None:
            self.velocity_apply(self.t)

        super().timestep()

        with timed_stage("Physics"):
            for _, scheme in self.physics_schemes:
                scheme.apply(self.x.np1(scheme.field_name), self.x.np1(scheme.field_name))
