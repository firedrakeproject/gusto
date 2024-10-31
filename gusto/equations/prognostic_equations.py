"""Objects describing geophysical fluid equations to be solved in weak form."""

from abc import ABCMeta
from firedrake import (
    TestFunction, Function, inner, dx, MixedFunctionSpace, TestFunctions,
    TrialFunction, DirichletBC, split, action
)
from firedrake.fml import (
    Term, all_terms, keep, drop, Label, subject,
    replace_subject, replace_trial_function
)
from gusto.core import PrescribedFields
from gusto.core.labels import (nonlinear_time_derivative, time_derivative,
                               prognostic, linearisation, mass_weighted)
from gusto.equations.common_forms import (
    advection_form, continuity_form, tracer_conservative_form
)
from gusto.equations.active_tracers import ActiveTracer
from gusto.core.configuration import TransportEquationType
import ufl

__all__ = ["PrognosticEquation", "PrognosticEquationSet"]


class PrognosticEquation(object, metaclass=ABCMeta):
    """Base class for prognostic equations."""

    def __init__(self, domain, function_space, field_name):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            function_space (:class:`FunctionSpace`): the function space that the
                equation's prognostic is defined on.
            field_name (str): name of the prognostic field.
        """

        self.domain = domain
        self.function_space = function_space
        self.X = Function(function_space)
        self.field_name = field_name
        self.bcs = {}
        self.prescribed_fields = PrescribedFields()

        if len(function_space) > 1:
            assert hasattr(self, "field_names")
            for fname in self.field_names:
                self.bcs[fname] = []
        else:
            # To avoid confusion, only add "self.test" when not mixed FS
            self.test = TestFunction(function_space)

        self.bcs[field_name] = []

    def label_terms(self, term_filter, label):
        """
        Labels terms in the equation, subject to the term filter.


        Args:
            term_filter (func): a function, taking terms as an argument, that
                is used to filter terms.
            label (:class:`Label`): the label to be applied to the terms.
        """
        assert type(label) == Label
        self.residual = self.residual.label_map(term_filter, map_if_true=label)


class PrognosticEquationSet(PrognosticEquation, metaclass=ABCMeta):
    """
    Base class for solving a set of prognostic equations.

    A prognostic equation set contains multiple prognostic variables, which are
    solved for simultaneously in a :class:`MixedFunctionSpace`. This base class
    contains common routines for these equation sets.
    """

    def __init__(self, field_names, domain, space_names,
                 linearisation_map=None, no_normal_flow_bc_ids=None,
                 active_tracers=None):
        """
        Args:
            field_names (list): a list of strings for names of the prognostic
                variables for the equation set.
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            space_names (dict): a dictionary of strings for names of the
                function spaces to use for the spatial discretisation. The keys
                are the names of the prognostic variables.
            linearisation_map (func, optional): a function specifying which
                terms in the equation set to linearise. Defaults to None.
            no_normal_flow_bc_ids (list, optional): a list of IDs of domain
                boundaries at which no normal flow will be enforced. Defaults to
                None.
            active_tracers (list, optional): a list of `ActiveTracer` objects
                that encode the metadata for any active tracers to be included
                in the equations.. Defaults to None.
        """

        self.field_names = field_names
        self.space_names = space_names
        self.active_tracers = active_tracers
        self.linearisation_map = lambda t: False if linearisation_map is None else linearisation_map(t)

        # Build finite element spaces
        self.spaces = [domain.spaces(space_name) for space_name in
                       [self.space_names[field_name] for field_name in self.field_names]]

        # Add active tracers to the list of prognostics
        if active_tracers is None:
            active_tracers = []
        self.add_tracers_to_prognostics(domain, active_tracers)

        # Make the full mixed function space
        W = MixedFunctionSpace(self.spaces)

        # Can now call the underlying PrognosticEquation
        full_field_name = "_".join(self.field_names)
        super().__init__(domain, W, full_field_name)

        # Set up test functions, trials and prognostics
        self.tests = TestFunctions(W)
        self.trials = TrialFunction(W)
        self.X_ref = Function(W)

        # Set up no-normal-flow boundary conditions
        if no_normal_flow_bc_ids is None:
            no_normal_flow_bc_ids = []
        self.set_no_normal_flow_bcs(domain, no_normal_flow_bc_ids)

    # ======================================================================== #
    # Set up time derivative / mass terms
    # ======================================================================== #

    def generate_mass_terms(self):
        """
        Builds the weak time derivative terms for the equation set.

        Generates the weak time derivative terms ("mass terms") for all the
        prognostic variables of the equation set.

        Returns:
            :class:`LabelledForm`: a labelled form containing the mass terms.
        """

        if self.active_tracers is None:
            tracer_names = []
        else:
            tracer_names = [tracer.name for tracer in self.active_tracers]

        for i, (test, field_name) in enumerate(zip(self.tests, self.field_names)):
            prog = split(self.X)[i]
            if field_name != 'exner':
                mass = subject(prognostic(inner(prog, test)*dx, field_name), self.X)

                # Check if the field is a conservatively transported tracer. If so,
                # create a mass-weighted mass form and store this and the original
                # mass form in a mass-weighted label
                for j, tracer_name in enumerate(tracer_names):
                    if field_name == tracer_name:
                        if self.active_tracers[j].transport_eqn == TransportEquationType.tracer_conservative:
                            standard_mass_form = mass

                            # The mass-weighted mass form is multiplied by the reference density
                            ref_density_idx = self.field_names.index(self.active_tracers[j].density_name)
                            ref_density = split(self.X)[ref_density_idx]
                            q = prog*ref_density
                            mass_weighted_form = nonlinear_time_derivative(time_derivative(
                                subject(prognostic(inner(q, test)*dx, field_name), self.X)))

                            mass = mass_weighted(standard_mass_form, mass_weighted_form)
                if i == 0:
                    mass_form = time_derivative(mass)
                else:
                    mass_form += time_derivative(mass)

        return mass_form

    # ======================================================================== #
    # Linearisation Routines
    # ======================================================================== #

    def generate_linear_terms(self, residual, linearisation_map):
        """
        Generate the linearised forms for the equation set.

        Generates linear forms for each of the terms in the equation set
        (unless specified otherwise). The linear forms are then added to the
        terms through a `linearisation` :class:`Label`.

        Linear forms are generated by replacing the `subject` using the
        `ufl.derivative` to obtain the forms linearised around reference states.

        Terms that already have a `linearisation` label are left.

        Args:
            residual (:class:`LabelledForm`): the residual of the equation set.
                A labelled form containing all the terms of the equation set.
            linearisation_map (func): a function describing the terms to be
                linearised.

        Returns:
            :class:`LabelledForm`: the residual with linear terms attached to
                each term as labels.
        """

        from functools import partial

        # Function to check if term should be linearised
        def should_linearise(term):
            return (not term.has_label(linearisation) and linearisation_map(term))

        # Linearise a term, and add the linearisation as a label
        def linearise(term, X, X_ref, du):
            linear_term = Term(action(ufl.derivative(term.form, X), du), term.labels)
            return linearisation(term, replace_subject(X_ref)(linear_term))

        # Add linearisations to all terms that need linearising
        residual = residual.label_map(
            should_linearise,
            map_if_true=partial(linearise, X=self.X, X_ref=self.X_ref, du=self.trials),
            map_if_false=keep,
        )

        return residual

    def linearise_equation_set(self):
        """
        Linearises the equation set.

        Linearises the whole equation set, replacing all the equations with
        the complete linearisation. Terms without linearisations are dropped.
        All labels are carried over, and the original linearisations containing
        the trial function are kept as labels to the new terms.
        """

        # Replace all terms with their linearisations, drop terms without
        self.residual = self.residual.label_map(
            lambda t: t.has_label(linearisation),
            map_if_true=lambda t: Term(t.get(linearisation).form, t.labels),
            map_if_false=drop)

        # Replace trial functions with the prognostics
        self.residual = self.residual.label_map(
            all_terms, replace_trial_function(self.X))

    # ======================================================================== #
    # Boundary Condition Routines
    # ======================================================================== #

    def set_no_normal_flow_bcs(self, domain, no_normal_flow_bc_ids):
        """
        Sets up the boundary conditions for no-normal flow at domain boundaries.

        Sets up the no-normal-flow boundary conditions, storing the
        :class:`DirichletBC` object at each specified boundary. There must be
        a velocity variable named 'u' to apply the boundary conditions to.

        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            no_normal_flow_bc_ids (list): A list of IDs of the domain boundaries
                at which no normal flow will be enforced.

        Raises:
            NotImplementedError: if there is no velocity field (with name 'u')
                in the equation set.
        """

        if 'u' not in self.field_names:
            raise NotImplementedError(
                'No-normal-flow boundary conditions can only be applied '
                + 'when there is a variable called "u" and none was found')

        Vu = domain.spaces("HDiv")
        # we only apply no normal-flow BCs when extruded mesh is non periodic
        if Vu.extruded and not Vu.ufl_domain().topology.extruded_periodic:
            self.bcs['u'].append(DirichletBC(Vu, 0.0, "bottom"))
            self.bcs['u'].append(DirichletBC(Vu, 0.0, "top"))
        for id in no_normal_flow_bc_ids:
            self.bcs['u'].append(DirichletBC(Vu, 0.0, id))

        # Add all boundary conditions to mixed function space
        W = self.X.function_space()
        self.bcs[self.field_name] = []
        for idx, field_name in enumerate(self.field_names):
            for bc in self.bcs[field_name]:
                self.bcs[self.field_name].append(DirichletBC(W.sub(idx), bc.function_arg, bc.sub_domain))

    # ======================================================================== #
    # Active Tracer Routines
    # ======================================================================== #

    def add_tracers_to_prognostics(self, domain, active_tracers):
        """
        Augments the equation set with specified active tracer variables.

        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            active_tracers (list): A list of :class:`ActiveTracer` objects that
                encode the metadata for the active tracers.

        Raises:
            ValueError: the equation set already contains a variable with the
                name of the active tracer.
        """

        # Loop through tracer fields and add field names and spaces
        for tracer in active_tracers:
            if isinstance(tracer, ActiveTracer):
                if tracer.name not in self.field_names:
                    self.field_names.append(tracer.name)
                else:
                    raise ValueError(f'There is already a field named {tracer.name}')

                # Add name of space to self.space_names, but check for conflict
                # with the tracer's name
                if tracer.name in self.space_names:
                    assert self.space_names[tracer.name] == tracer.space, \
                        'space_name dict provided to equation has space ' \
                        + f'{self.space_names[tracer.name]} for tracer ' \
                        + f'{tracer.name} which conflicts with the space ' \
                        + f'{tracer.space} specified in the ActiveTracer object'
                else:
                    self.space_names[tracer.name] = tracer.space
                self.spaces.append(domain.spaces(tracer.space))
            else:
                raise TypeError(f'Tracers must be ActiveTracer objects, not {type(tracer)}')

    def generate_tracer_transport_terms(self, active_tracers):
        """
        Adds the transport forms for the active tracers to the equation set.

        Args:
            active_tracers (list): A list of :class:`ActiveTracer` objects that
                encode the metadata for the active tracers.

        Raises:
            ValueError: if the transport equation encoded in the active tracer
                metadata is not valid.

        Returns:
            :class:`LabelledForm`: a labelled form containing the transport
                terms for the active tracers.
        """

        # By default return None if no tracers are to be transported
        adv_form = None
        no_tracer_transported = True

        if 'u' in self.field_names:
            u_idx = self.field_names.index('u')
            u = split(self.X)[u_idx]
        elif 'u' in self.prescribed_fields._field_names:
            u = self.prescribed_fields('u')
        else:
            raise ValueError('Cannot generate tracer transport terms '
                             + 'as there is no velocity field')

        for _, tracer in enumerate(active_tracers):
            if tracer.transport_eqn != TransportEquationType.no_transport:
                idx = self.field_names.index(tracer.name)
                tracer_prog = split(self.X)[idx]
                tracer_test = self.tests[idx]
                if tracer.transport_eqn == TransportEquationType.advective:
                    tracer_adv = subject(prognostic(
                        advection_form(tracer_test, tracer_prog, u),
                        tracer.name), self.X)
                elif tracer.transport_eqn == TransportEquationType.conservative:
                    tracer_adv = subject(prognostic(
                        continuity_form(tracer_test, tracer_prog, u),
                        tracer.name), self.X)
                elif tracer.transport_eqn == TransportEquationType.tracer_conservative:
                    default_adv_form = subject(prognostic(
                        advection_form(tracer_test, tracer_prog, u),
                        tracer.name), self.X)

                    ref_density_idx = self.field_names.index(tracer.density_name)
                    ref_density = split(self.X)[ref_density_idx]
                    mass_weighted_tracer_adv = subject(prognostic(
                        tracer_conservative_form(tracer_test, tracer_prog, ref_density, u),
                        tracer.name), self.X)

                    # Store the conservative transport form in the mass_weighted label,
                    # but by default use an advective form.
                    tracer_adv = mass_weighted(default_adv_form, mass_weighted_tracer_adv)
                else:
                    raise ValueError(f'Transport eqn {tracer.transport_eqn} not recognised')
                if no_tracer_transported:
                    # We arrive here for the first tracer to be transported
                    adv_form = tracer_adv
                    no_tracer_transported = False
                else:
                    adv_form += tracer_adv

        return adv_form

    def get_active_tracer(self, field_name):
        """
        Returns the active tracer metadata object for a particular field.

        Args:
            field_name (str): the name of the field to return the metadata for.

        Returns:
            :class:`ActiveTracer`: the object storing the metadata describing
                the tracer.
        """

        active_tracer_to_return = None

        for active_tracer in self.active_tracers:
            if active_tracer.name == field_name:
                active_tracer_to_return = active_tracer
                break

        if active_tracer_to_return is None:
            raise RuntimeError(f'Unable to find active tracer {field_name}')

        return active_tracer_to_return
