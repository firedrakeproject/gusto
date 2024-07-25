"""Split timestepping methods for generically solving terms separately."""

from firedrake import Projector
from firedrake.fml import Label, drop
from pyop2.profiling import timed_stage
from gusto.core.labels import time_derivative, physics_label, dynamics_label
from gusto.time_discretisation.time_discretisation import ExplicitTimeDiscretisation
from gusto.timestepping.timestepper import BaseTimestepper, Timestepper

__all__ = ["SplitTimestepper", "SplitTransportDiffusionTimestepper", "SplitPhysicsTimestepper", "SplitPhysicsDynamicsTimestepper", "SplitPrescribedTransport"]


class SplitTimestepper(BaseTimestepper):
    """
    Implements a timeloop by applying separate schemes to different terms, e.g
    physics and dynamics. This splits these terms and allows a different 
    time discretisation to be applied to each.
    """
    
    def __init__(self, equation, term_splitting, io, spatial_methods=None,
                 dynamics_schemes=None, physics_schemes=None):
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
            dynamics_schemes: (:class:`TimeDiscretisation`, optional) A list of time
            discretisations for use with any dynamics schemes. Defaults to None.
            physics_schemes: (list, optional): a list of tuples of the form
                (:class:`PhysicsParametrisation`, :class:`TimeDiscretisation`),
                pairing physics parametrisations and timestepping schemes to use
                for each. Timestepping schemes for physics must be explicit.
                Defaults to None.
        """


        # As we handle physics differently to the Timestepper, these are not
        # passed to the super __init__
        super().__init__(equation, io)
        
        # If we have physics schemes, extract these first.
        if physics in term_splitting:
            if physics_schemes is None:
                raise ValueError('Physics schemes need to be specified when splitting physics terms in the SplitTimestepper')
            else:
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
            
        # Extract all non-physics or time derivative terms.
        other_terms = self.residual.label_map(lambda t: t.has_label(label), map_if_true=keep)
            
        term_count = len(other_terms)
        counts = 0
            
        # Check that the labels in term_splitting are used in the equation
        for label in term_splitting:
            if term == dynamics:
                terms = other_terms.label_map(lambda t: t.has_label(dynamics_label), map_if_true=drop)
            else:
                terms = other_terms.label_map(lambda t: t.has_label(label), map_if_true=drop)
            assert len(terms) > 0, \
                f'The {label} term in the term_splitting list does not correspond to any components in the equation.'
            counts += len(terms)
            
        assert counts == term_count, 'The terms used in the SplitTimestepper do not correctly split the equation.'
                
        self.dynamics_schemes = dynamics_schemes

    @property
    def transporting_velocity(self):
        return "prognostic"

    def setup_scheme(self):
        self.setup_equation(self.equation)

        # Check that all terms in the equation have been specified in the 
        # term_splitting list.
        terms = self.residual.label_map(lambda t: not any(t.has_label(time_derivative, physics_label), map_if_true=keep))
        for term in term_splitting:
            if term == dynamics:
                terms = terms.label_map(lambda t: t.has_label(dynamics_label), map_if_true=drop)
            else:
                terms = terms.label_map(lambda t: t.has_label(term), map_if_true=drop)
            if len(terms) > 0:
                raise ValueError('The term_splitting list for the SplitTimestepper has not covered all terms in the equation.')

        apply_bcs = True
        
        for term_type, scheme in dynamics_schemes:
            scheme.setup(self.equation, apply_bcs, term_type)
            self.setup_transporting_velocity(self.scheme)
            #if term_type == transport:
            if self.io.output.log_courant:
                self.scheme.courant_max = self.io.courant_max
                
        def timestep(self):

            # Perform timestepping in the order required:
            for term in term_splitting:
                if term == 'physics':
                    with timed_stage("Physics"):
                        for _, scheme in self.physics_schemes:
                            scheme.apply(self.x.np1(scheme.field_name), self.x.np1(scheme.field_name))
                else:
                    for name, scheme in self.dynamics_schemes:
                        scheme.apply(xnp1(name), xnp1(name))
            super().timestep()
    
            
class SplitTransportDiffusionTimestepper(SplitTimestepper):
    """
    A version of the SplitTimestepper that separates the transport and
    diffusion terms in an AdvectionDiffusion equation and solve
    these separately.
    """
    def __init__(self, equation, transport_scheme, diffusion_scheme, io, spatial_methods=None):
                 
        """
        Args:
            equation (:class:`PrognosticEquation`): the prognostic equation
            dynamics_scheme (:class:`TimeDiscretisation`): the scheme to use to
                timestep the prognostic equation
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
        term_splitting = ['transport', 'diffusion']
        
        dynamics_schemes = [('transport', transport_scheme),
                            ('diffusion', diffusion_scheme)]
        
        super().__init__(equation, term_splitting, io, spatial_methods=None,
                         dynamics_schemes=dynamics_schemes, 
                         physics_schemes=None)
            
            
class SplitPhysicsDynamicsTimestepper(SplitTimestepper):
    """
    A version of the SplitTimestepper that separates the physics from the other
    terms, which are labelled as the dynamics.
    """
    def __init__(self, equation, dynamics_scheme, io, spatial_methods=None,
                 physics_schemes=None):
                 
        """
        Args:
            equation (:class:`PrognosticEquation`): the prognostic equation
            dynamics_scheme (:class:`TimeDiscretisation`): the scheme to use to
                timestep the prognostic equation
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
        term_splitting = ['dynamics', 'physics']
        
        dynamics_schemes = [('dynamics', dynamics_scheme)]
        
        super().__init__(equation, term_splitting, io,
                         spatial_methods=spatial_methods,
                         dynamics_schemes=dynamics_schemes,
                         physics_schemes=physics_schemes)

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
