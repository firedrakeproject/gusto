"""Split timestepping methods for generically solving terms separately."""

from firedrake import Projector
from firedrake.fml import Label
from pyop2.profiling import timed_stage
from gusto.core.labels import time_derivative, physics_label
from gusto.time_discretisation.time_discretisation import ExplicitTimeDiscretisation
from gusto.timestepping.timestepper import Timestepper

__all__ = ["SplitTimestepper", "SplitPhysicsTimestepper", "SplitPrescribedTransport"]


class SplitTimestepper(Timestepper):
    """
    Implements a timeloop by applying separate schemes to different terms, e.g
    physics and dynamics. This splits these terms and allows a different 
    time discretisation to be applied to each.
    """
    
    def __init__(self, equation, term_splitting, io, spatial_methods=None,
                 physics_schemes=None):
        """
        Args:
            equation (:class:`PrognosticEquation`): the prognostic equation
            term_splitting (list): a list of labels giving the terms that should
                be solved separately in the order in which this should be achieved.
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
            
        # Check that the labels in term_splitting are used in the equation
        for label in term_splitting:
            assert len(self.residual.label_map(lambda t: t.has_label(label), map_if_true=keep) > 0):
                raise ValueError(f'The {label} term in the term_splitting list does not correspond to any components in the equation.')
        


        @property
    def transporting_velocity(self):
        return "prognostic"

    def setup_scheme(self):
        self.setup_equation(self.equation)

        # Check that all terms in the equation have been specified in the 
        # term_splitting list.
        terms = self.residual.label_map(lambda t: t.has_label(time_derivative), map_if_true=drop)
        for term in term_splitting
            terms = terms.label_map(lambda t: t.has_label(term), map_if_true=drop)
            
        if len(terms) > 0:
            raise ValueError('The term_splitting list for the SplitTimestepper has not covered all terms in the equation.')

        apply_bcs = True
        self.scheme.setup(self.equation, apply_bcs, ... )
        self.setup_transporting_velocity(self.scheme)
        if self.io.output.log_courant:
            self.scheme.courant_max = self.io.courant_max

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

    def __init__(self, equation, scheme, io, spatial_methods=None,
                 physics_schemes=None,
                 prescribed_transporting_velocity=None):
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

        if prescribed_transporting_velocity is not None:
            self.velocity_projection = Projector(
                prescribed_transporting_velocity(self.t),
                self.fields('u'))
        else:
            self.velocity_projection = None

    @property
    def transporting_velocity(self):
        return self.fields('u')

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

        if self.velocity_projection is not None:
            self.velocity_projection.project()

        super().timestep()

        with timed_stage("Physics"):
            for _, scheme in self.physics_schemes:
                scheme.apply(self.x.np1(scheme.field_name), self.x.np1(scheme.field_name))
