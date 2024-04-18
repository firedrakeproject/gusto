"""Objects describing geophysical fluid equations to be solved in weak form."""

from abc import ABCMeta
from firedrake import (
    TestFunction, Function, sin, pi, inner, dx, div, cross,
    FunctionSpace, MixedFunctionSpace, TestFunctions, TrialFunction,
    FacetNormal, jump, avg, dS_v, dS, DirichletBC, conditional,
    SpatialCoordinate, split, Constant, action
)
from firedrake.fml import (
    Term, all_terms, keep, drop, Label, subject,
    replace_subject, replace_trial_function
)
from gusto.fields import PrescribedFields
from gusto.labels import (
    time_derivative, transport, prognostic, hydrostatic, linearisation,
    mass_weighted, pressure_gradient, coriolis, divergence, gravity,
    incompressible, sponge
)
from gusto.thermodynamics import exner_pressure
from gusto.common_forms import (
    advection_form, advection_form_1d, continuity_form,
    continuity_form_1d, vector_invariant_form,
    kinetic_energy_form, advection_equation_circulation_form,
    diffusion_form, diffusion_form_1d,
    linear_continuity_form, linear_continuity_form_1d, linear_advection_form,
    tracer_conservative_form
)
from gusto.active_tracers import ActiveTracer, Phases, TracerVariableType
from gusto.configuration import TransportEquationType
import ufl


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


class AdvectionEquation(PrognosticEquation):
    u"""Discretises the advection equation, ∂q/∂t + (u.∇)q = 0"""

    def __init__(self, domain, function_space, field_name, Vu=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            function_space (:class:`FunctionSpace`): the function space that the
                equation's prognostic is defined on.
            field_name (str): name of the prognostic field.
            Vu (:class:`FunctionSpace`, optional): the function space for the
                velocity field. If this is not specified, uses the HDiv spaces
                set up by the domain. Defaults to None.
        """
        super().__init__(domain, function_space, field_name)

        if Vu is not None:
            domain.spaces.add_space("HDiv", Vu, overwrite_space=True)
        V = domain.spaces("HDiv")
        u = self.prescribed_fields("u", V)

        test = self.test
        q = self.X
        mass_form = time_derivative(inner(q, test)*dx)
        transport_form = advection_form(test, q, u)

        self.residual = prognostic(subject(mass_form + transport_form, q), field_name)


class ContinuityEquation(PrognosticEquation):
    u"""Discretises the continuity equation, ∂q/∂t + ∇.(u*q) = 0"""

    def __init__(self, domain, function_space, field_name, Vu=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            function_space (:class:`FunctionSpace`): the function space that the
                equation's prognostic is defined on.
            field_name (str): name of the prognostic field.
            Vu (:class:`FunctionSpace`, optional): the function space for the
                velocity field. If this is not specified, uses the HDiv spaces
                set up by the domain. Defaults to None.
        """
        super().__init__(domain, function_space, field_name)

        if Vu is not None:
            domain.spaces.add_space("HDiv", Vu, overwrite_space=True)
        V = domain.spaces("HDiv")
        u = self.prescribed_fields("u", V)

        test = self.test
        q = self.X
        mass_form = time_derivative(inner(q, test)*dx)
        transport_form = continuity_form(test, q, u)

        self.residual = prognostic(subject(mass_form + transport_form, q), field_name)


class DiffusionEquation(PrognosticEquation):
    u"""Discretises the diffusion equation, ∂q/∂t = ∇.(κ∇q)"""

    def __init__(self, domain, function_space, field_name,
                 diffusion_parameters):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            function_space (:class:`FunctionSpace`): the function space that the
                equation's prognostic is defined on.
            field_name (str): name of the prognostic field.
            diffusion_parameters (:class:`DiffusionParameters`): parameters
                describing the diffusion to be applied.
        """
        super().__init__(domain, function_space, field_name)

        test = self.test
        q = self.X
        mass_form = time_derivative(inner(q, test)*dx)
        diffusive_form = diffusion_form(test, q, diffusion_parameters.kappa)

        self.residual = prognostic(subject(mass_form + diffusive_form, q), field_name)


class AdvectionDiffusionEquation(PrognosticEquation):
    u"""The advection-diffusion equation, ∂q/∂t + (u.∇)q = ∇.(κ∇q)"""

    def __init__(self, domain, function_space, field_name, Vu=None,
                 diffusion_parameters=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            function_space (:class:`FunctionSpace`): the function space that the
                equation's prognostic is defined on.
            field_name (str): name of the prognostic field.
            Vu (:class:`FunctionSpace`, optional): the function space for the
                velocity field. If this is  Defaults to None.
            diffusion_parameters (:class:`DiffusionParameters`, optional):
                parameters describing the diffusion to be applied.
        """

        super().__init__(domain, function_space, field_name)

        if Vu is not None:
            domain.spaces.add_space("HDiv", Vu, overwrite_space=True)
        V = domain.spaces("HDiv")
        u = self.prescribed_fields("u", V)

        test = self.test
        q = self.X
        mass_form = time_derivative(inner(q, test)*dx)
        transport_form = advection_form(test, q, u)
        diffusive_form = diffusion_form(test, q, diffusion_parameters.kappa)

        self.residual = prognostic(subject(
            mass_form + transport_form + diffusive_form, q), field_name)


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
            mass = subject(prognostic(inner(prog, test)*dx, field_name), self.X)
            for j, ind_tracer_name in enumerate(tracer_names):
                if field_name == ind_tracer_name:
                    if self.active_tracers[j].transport_eqn == TransportEquationType.tracer_conservative:
                        standard_mass_form = mass

                        # We need to generate a mass form that is multiplied by the density
                        ref_density_idx = self.field_names.index(self.active_tracers[j].density_name)
                        ref_density = split(self.X)[ref_density_idx]
                        q = prog*ref_density
                        mass_weighted_form = time_derivative(subject(prognostic(inner(q, test)*dx,
                                                             field_name), self.X))

                        # Store the mass form for any conservative transport
                        # in the mass_weighted label
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


class CoupledTransportEquation(PrognosticEquationSet):
    u"""
    Discretises the transport equation,               \n
    ∂q/∂t + (u.∇)q = F,
    with the application of active tracers.
    As there are multiple tracers or species that are
    interacting, q and F are vectors.
    This equation can be enhanced through the addition of
    sources or sinks (F) by applying it with physics schemes.
    """
    def __init__(self, domain, active_tracers, Vu=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            active_tracers (list): a list of `ActiveTracer` objects
                that encode the metadata for any active tracers to be included
                in the equations. This is required for using this class; if there
                is only a field to be advected, use the AdvectionEquation
                instead.
            Vu (:class:`FunctionSpace`, optional): the function space for the
                velocity field. If this is not specified, uses the HDiv spaces
                set up by the domain. Defaults to None.
        """

        self.active_tracers = active_tracers
        self.terms_to_linearise = {}
        self.field_names = []
        self.space_names = {}

        # Build finite element spaces
        self.spaces = []

        # Add active tracers to the list of prognostics
        if active_tracers is None:
            active_tracers = []
        self.add_tracers_to_prognostics(domain, active_tracers)

        # Make the full mixed function space
        W = MixedFunctionSpace(self.spaces)

        full_field_name = "_".join(self.field_names)
        PrognosticEquation.__init__(self, domain, W, full_field_name)

        if Vu is not None:
            domain.spaces.add_space("HDiv", Vu, overwrite_space=True)
        V = domain.spaces("HDiv")
        _ = self.prescribed_fields("u", V)

        self.tests = TestFunctions(W)
        self.X = Function(W)

        # Add mass forms for the tracers
        self.residual = self.generate_mass_terms()

        # Add transport of tracers
        self.residual += self.generate_tracer_transport_terms(active_tracers)


# ============================================================================ #
# Specified Equation Sets
# ============================================================================ #

class ShallowWaterEquations(PrognosticEquationSet):
    u"""
    Class for the (rotating) shallow-water equations, which evolve the velocity
    'u' and the depth field 'D', via some variant of:                         \n
    ∂u/∂t + (u.∇)u + f×u + g*∇(D+b) = 0,                                      \n
    ∂D/∂t + ∇.(D*u) = 0,                                                      \n
    for Coriolis parameter 'f' and bottom surface 'b'.
    """

    def __init__(self, domain, parameters, fexpr=None, bexpr=None,
                 space_names=None, linearisation_map='default',
                 u_transport_option='vector_invariant_form',
                 no_normal_flow_bc_ids=None, active_tracers=None,
                 thermal=False):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            parameters (:class:`Configuration`, optional): an object containing
                the model's physical parameters.
            fexpr (:class:`ufl.Expr`, optional): an expression for the Coroilis
                parameter. Defaults to None.
            bexpr (:class:`ufl.Expr`, optional): an expression for the bottom
                surface of the fluid. Defaults to None.
            space_names (dict, optional): a dictionary of strings for names of
                the function spaces to use for the spatial discretisation. The
                keys are the names of the prognostic variables. Defaults to None
                in which case the spaces are taken from the de Rham complex. Any
                buoyancy variable is taken by default to lie in the L2 space.
            linearisation_map (func, optional): a function specifying which
                terms in the equation set to linearise. If None is specified
                then no terms are linearised. Defaults to the string 'default',
                in which case the linearisation includes both time derivatives,
                the 'D' transport term and the pressure gradient term.
            u_transport_option (str, optional): specifies the transport term
                used for the velocity equation. Supported options are:
                'vector_invariant_form', 'vector_advection_form', and
                'circulation_form'.
                Defaults to 'vector_invariant_form'.
            no_normal_flow_bc_ids (list, optional): a list of IDs of domain
                boundaries at which no normal flow will be enforced. Defaults to
                None.
            active_tracers (list, optional): a list of `ActiveTracer` objects
                that encode the metadata for any active tracers to be included
                in the equations. Defaults to None.
            thermal (flag, optional): specifies whether the equations have a
                thermal or buoyancy variable that feeds back on the momentum.
                Defaults to False.

        Raises:
            NotImplementedError: active tracers are not yet implemented.
        """

        self.thermal = thermal
        field_names = ['u', 'D']

        if space_names is None:
            space_names = {'u': 'HDiv', 'D': 'L2'}

        if active_tracers is None:
            active_tracers = []

        if self.thermal:
            field_names.append('b')
            if 'b' not in space_names.keys():
                space_names['b'] = 'L2'

        if linearisation_map == 'default':
            # Default linearisation is time derivatives, pressure gradient and
            # transport term from depth equation. Don't include active tracers
            linearisation_map = lambda t: \
                t.get(prognostic) in ['u', 'D', 'b'] \
                and (any(t.has_label(time_derivative, pressure_gradient))
                     or (t.get(prognostic) in ['D', 'b'] and t.has_label(transport)))
        super().__init__(field_names, domain, space_names,
                         linearisation_map=linearisation_map,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        self.parameters = parameters
        g = parameters.g
        H = parameters.H

        w, phi = self.tests[0:2]
        u, D = split(self.X)[0:2]
        u_trial = split(self.trials)[0]

        # -------------------------------------------------------------------- #
        # Time Derivative Terms
        # -------------------------------------------------------------------- #
        mass_form = self.generate_mass_terms()

        # -------------------------------------------------------------------- #
        # Transport Terms
        # -------------------------------------------------------------------- #
        # Velocity transport term -- depends on formulation
        if u_transport_option == "vector_invariant_form":
            u_adv = prognostic(vector_invariant_form(domain, w, u, u), 'u')
        elif u_transport_option == "vector_advection_form":
            u_adv = prognostic(advection_form(w, u, u), 'u')
        elif u_transport_option == "circulation_form":
            ke_form = prognostic(kinetic_energy_form(w, u, u), 'u')
            u_adv = prognostic(advection_equation_circulation_form(domain, w, u, u), 'u') + ke_form
        else:
            raise ValueError("Invalid u_transport_option: %s" % u_transport_option)

        # Depth transport term
        D_adv = prognostic(continuity_form(phi, D, u), 'D')

        # Transport term needs special linearisation
        if self.linearisation_map(D_adv.terms[0]):
            linear_D_adv = linear_continuity_form(phi, H, u_trial)
            # Add linearisation to D_adv
            D_adv = linearisation(D_adv, linear_D_adv)

        adv_form = subject(u_adv + D_adv, self.X)

        # Add transport of tracers
        if len(active_tracers) > 0:
            adv_form += self.generate_tracer_transport_terms(active_tracers)
        # Add transport of buoyancy, if thermal shallow water equations
        if self.thermal:
            gamma = self.tests[2]
            b = split(self.X)[2]
            b_adv = prognostic(advection_form(gamma, b, u), 'b')
            adv_form += subject(b_adv, self.X)

        # -------------------------------------------------------------------- #
        # Pressure Gradient Term
        # -------------------------------------------------------------------- #
        # Add pressure gradient only if not doing thermal
        if self.thermal:
            residual = (mass_form + adv_form)
        else:
            pressure_gradient_form = pressure_gradient(
                subject(prognostic(-g*div(w)*D*dx, 'u'), self.X))

            residual = (mass_form + adv_form + pressure_gradient_form)

        # -------------------------------------------------------------------- #
        # Extra Terms (Coriolis, Topography and Thermal)
        # -------------------------------------------------------------------- #
        # TODO: Is there a better way to store the Coriolis / topography fields?
        # The current approach is that these are prescribed fields, stored in
        # the equation, and initialised when the equation is

        if fexpr is not None:
            V = FunctionSpace(domain.mesh, 'CG', 1)
            f = self.prescribed_fields('coriolis', V).interpolate(fexpr)
            coriolis_form = coriolis(subject(
                prognostic(f*inner(domain.perp(u), w)*dx, "u"), self.X))
            # Add linearisation
            if self.linearisation_map(coriolis_form.terms[0]):
                linear_coriolis = coriolis(
                    subject(prognostic(f*inner(domain.perp(u_trial), w)*dx, 'u'), self.X))
                coriolis_form = linearisation(coriolis_form, linear_coriolis)
            residual += coriolis_form

        if bexpr is not None:
            topography = self.prescribed_fields('topography', domain.spaces('DG')).interpolate(bexpr)
            if self.thermal:
                n = FacetNormal(domain.mesh)
                topography_form = subject(prognostic
                                          (-topography*div(b*w)*dx
                                           + jump(b*w, n)*avg(topography)*dS,
                                           'u'), self.X)
            else:
                topography_form = subject(prognostic
                                          (-g*div(w)*topography*dx, 'u'),
                                          self.X)
            residual += topography_form

        # thermal source terms not involving topography
        if self.thermal:
            n = FacetNormal(domain.mesh)
            source_form = subject(prognostic(-D*div(b*w)*dx
                                             - 0.5*b*div(D*w)*dx
                                             + jump(b*w, n)*avg(D)*dS
                                             + 0.5*jump(D*w, n)*avg(b)*dS,
                                             'u'), self.X)
            residual += source_form

        # -------------------------------------------------------------------- #
        # Linearise equations
        # -------------------------------------------------------------------- #
        # Add linearisations to equations
        self.residual = self.generate_linear_terms(residual, self.linearisation_map)


class LinearShallowWaterEquations(ShallowWaterEquations):
    u"""
    Class for the linear (rotating) shallow-water equations, which describe the
    velocity 'u' and the depth field 'D', solving some variant of:            \n
    ∂u/∂t + f×u + g*∇(D+b) = 0,                                               \n
    ∂D/∂t + H*∇.(u) = 0,                                                      \n
    for mean depth 'H', Coriolis parameter 'f' and bottom surface 'b'.

    This is set up the from the underlying :class:`ShallowWaterEquations`,
    which is then linearised.
    """

    def __init__(self, domain, parameters, fexpr=None, bexpr=None,
                 space_names=None, linearisation_map='default',
                 u_transport_option="vector_invariant_form",
                 no_normal_flow_bc_ids=None, active_tracers=None,
                 thermal=False):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            parameters (:class:`Configuration`, optional): an object containing
                the model's physical parameters.
            fexpr (:class:`ufl.Expr`, optional): an expression for the Coroilis
                parameter. Defaults to None.
            bexpr (:class:`ufl.Expr`, optional): an expression for the bottom
                surface of the fluid. Defaults to None.
            space_names (dict, optional): a dictionary of strings for names of
                the function spaces to use for the spatial discretisation. The
                keys are the names of the prognostic variables. Defaults to None
                in which case the spaces are taken from the de Rham complex. Any
                buoyancy variable is taken by default to lie in the L2 space.
            linearisation_map (func, optional): a function specifying which
                terms in the equation set to linearise. If None is specified
                then no terms are linearised. Defaults to the string 'default',
                in which case the linearisation includes both time derivatives,
                the 'D' transport term, pressure gradient and Coriolis terms.
            u_transport_option (str, optional): specifies the transport term
                used for the velocity equation. Supported options are:
                'vector_invariant_form', 'vector_advection_form' and
                'circulation_form'.
                Defaults to 'vector_invariant_form'.
            no_normal_flow_bc_ids (list, optional): a list of IDs of domain
                boundaries at which no normal flow will be enforced. Defaults to
                None.
            active_tracers (list, optional): a list of `ActiveTracer` objects
                that encode the metadata for any active tracers to be included
                in the equations. Defaults to None.
            thermal (flag, optional): specifies whether the equations have a
                thermal or buoyancy variable that feeds back on the momentum.
                Defaults to False.
        """

        if linearisation_map == 'default':
            # Default linearisation is time derivatives, pressure gradient,
            # Coriolis and transport term from depth equation
            linearisation_map = lambda t: \
                (any(t.has_label(time_derivative, pressure_gradient, coriolis))
                 or (t.get(prognostic) in ['D', 'b'] and t.has_label(transport)))

        super().__init__(domain, parameters,
                         fexpr=fexpr, bexpr=bexpr, space_names=space_names,
                         linearisation_map=linearisation_map,
                         u_transport_option=u_transport_option,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers, thermal=thermal)

        # Use the underlying routine to do a first linearisation of the equations
        self.linearise_equation_set()


class ShallowWaterEquations_1d(PrognosticEquationSet):

    u"""
    Class for the (rotating) 1D shallow-water equations, which describe
    the velocity 'u', 'v' and the depth field 'D', solving some variant of:  \n
    ∂u/∂t + u∂u/∂x - fv + g*∂D/∂x = 0,                                       \n
    ∂v/∂t + fu = 0,                                                          \n
    ∂D/∂t + ∂(uD)/∂x = 0,                                                    \n
    for mean depth 'H', Coriolis parameter 'f' and gravity 'g'.

    Args:
        domain (:class:`Domain`): the model's domain object, containing the
            mesh and the compatible function spaces.
        parameters (:class:`Configuration`, optional): an object containing
            the model's physical parameters.
        fexpr (:class:`ufl.Expr`, optional): an expression for the Coroilis
            parameter. Defaults to None.
        space_names (dict, optional): a dictionary of strings for names of
            the function spaces to use for the spatial discretisation. The
            keys are the names of the prognostic variables. Defaults to None
            in which case the spaces are taken from the de Rham complex.
        linearisation_map (func, optional): a function specifying which
            terms in the equation set to linearise. If None is specified
            then no terms are linearised. Defaults to the string 'default',
            in which case the linearisation includes both time derivatives,
            the 'D' transport term, pressure gradient and Coriolis terms.
        no_normal_flow_bc_ids (list, optional): a list of IDs of domain
            boundaries at which no normal flow will be enforced. Defaults to
            None.
        active_tracers (list, optional): a list of `ActiveTracer` objects
            that encode the metadata for any active tracers to be included
            in the equations. Defaults to None.
    """

    def __init__(self, domain, parameters,
                 fexpr=None,
                 space_names=None, linearisation_map='default',
                 diffusion_options=None,
                 no_normal_flow_bc_ids=None, active_tracers=None):

        field_names = ['u', 'v', 'D']
        space_names = {'u': 'HDiv', 'v': 'L2', 'D': 'L2'}

        if active_tracers is not None:
            raise NotImplementedError('Tracers not implemented for 1D shallow water equations')

        if linearisation_map == 'default':
            # Default linearisation is time derivatives, pressure gradient,
            # Coriolis and transport term from depth equation
            linearisation_map = lambda t: \
                (any(t.has_label(time_derivative, pressure_gradient, coriolis))
                 or (t.get(prognostic) == 'D' and t.has_label(transport)))

        super().__init__(field_names, domain, space_names,
                         linearisation_map=linearisation_map,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        self.parameters = parameters
        g = parameters.g
        H = parameters.H

        w1, w2, phi = self.tests
        u, v, D = split(self.X)
        u_trial = split(self.trials)[0]

        # -------------------------------------------------------------------- #
        # Time Derivative Terms
        # -------------------------------------------------------------------- #
        mass_form = self.generate_mass_terms()

        # -------------------------------------------------------------------- #
        # Transport Terms
        # -------------------------------------------------------------------- #
        # Velocity transport term
        u_adv = prognostic(advection_form_1d(w1, u, u), 'u')
        v_adv = prognostic(advection_form_1d(w2, v, u), 'v')

        # Depth transport term
        D_adv = prognostic(continuity_form_1d(phi, D, u), 'D')

        # Transport term needs special linearisation
        if self.linearisation_map(D_adv.terms[0]):
            linear_D_adv = linear_continuity_form_1d(phi, H, u_trial)
            # Add linearisation to D_adv
            D_adv = linearisation(D_adv, linear_D_adv)

        adv_form = subject(u_adv + v_adv + D_adv, self.X)

        pressure_gradient_form = pressure_gradient(subject(
            prognostic(-g * D * w1.dx(0) * dx, "u"), self.X))

        self.residual = (mass_form + adv_form
                         + pressure_gradient_form)

        if fexpr is not None:
            V = FunctionSpace(domain.mesh, 'CG', 1)
            f = self.prescribed_fields('coriolis', V).interpolate(fexpr)
            coriolis_form = coriolis(subject(
                prognostic(-f * v * w1 * dx, "u")
                + prognostic(f * u * w2 * dx, "v"), self.X))
            self.residual += coriolis_form

        if diffusion_options is not None:
            for field, diffusion in diffusion_options:
                idx = self.field_names.index(field)
                test = self.tests[idx]
                fn = split(self.X)[idx]
                self.residual += subject(
                    prognostic(diffusion_form_1d(test, fn, diffusion.kappa),
                               field),
                    self.X)

        # -------------------------------------------------------------------- #
        # Linearise equations
        # -------------------------------------------------------------------- #
        # Add linearisations to equations
        self.residual = self.generate_linear_terms(self.residual,
                                                   self.linearisation_map)


class LinearShallowWaterEquations_1d(ShallowWaterEquations_1d):
    u"""
    Class for the linear (rotating) 1D shallow-water equations, which describe
    the velocity 'u', 'v' and the depth field 'D', solving some variant of:  \n
    ∂u/∂t - fv + g*∂D/∂x = 0,                                                \n
    ∂v/∂t + fu = 0,                                                          \n
    ∂D/∂t + H*∂u/∂x = 0,                                                     \n
    for mean depth 'H', Coriolis parameter 'f' and gravity 'g'.

    This is set up the from the underlying :class:`ShallowWaterEquations_1d`,
    which is then linearised.
    """

    def __init__(self, domain, parameters, fexpr=None,
                 space_names=None, linearisation_map='default',
                 no_normal_flow_bc_ids=None, active_tracers=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            parameters (:class:`Configuration`, optional): an object containing
                the model's physical parameters.
            fexpr (:class:`ufl.Expr`, optional): an expression for the Coroilis
                parameter. Defaults to None.
            space_names (dict, optional): a dictionary of strings for names of
                the function spaces to use for the spatial discretisation. The
                keys are the names of the prognostic variables. Defaults to None
                in which case the spaces are taken from the de Rham complex. Any
                buoyancy variable is taken by default to lie in the L2 space.
            linearisation_map (func, optional): a function specifying which
                terms in the equation set to linearise. If None is specified
                then no terms are linearised. Defaults to the string 'default',
                in which case the linearisation includes both time derivatives,
                the 'D' transport term, pressure gradient and Coriolis terms.
            no_normal_flow_bc_ids (list, optional): a list of IDs of domain
                boundaries at which no normal flow will be enforced. Defaults to
                None.
            active_tracers (list, optional): a list of `ActiveTracer` objects
                that encode the metadata for any active tracers to be included
                in the equations. Defaults to None.
        """

        if linearisation_map == 'default':
            # Default linearisation is time derivatives, pressure gradient,
            # Coriolis and transport term from depth equation
            linearisation_map = lambda t: \
                (any(t.has_label(time_derivative, pressure_gradient, coriolis))
                 or (t.get(prognostic) == 'D' and t.has_label(transport)))

        super().__init__(domain, parameters,
                         fexpr=fexpr, space_names=space_names,
                         linearisation_map=linearisation_map,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        # Use the underlying routine to do a first linearisation of the equations
        self.linearise_equation_set()


class CompressibleEulerEquations(PrognosticEquationSet):
    """
    Class for the compressible Euler equations, which evolve the velocity 'u',
    the dry density 'rho' and the (virtual dry) potential temperature 'theta',
    solving:                                                                  \n
    ∂u/∂t + (u.∇)u + 2Ω×u + c_p*θ*∇Π + g = 0,                                 \n
    ∂ρ/∂t + ∇.(ρ*u) = 0,                                                      \n
    ∂θ/∂t + (u.∇)θ = 0,                                                       \n
    where Π is the Exner pressure, g is the gravitational vector, Ω is the
    planet's rotation vector and c_p is the heat capacity of dry air at constant
    pressure.
    """

    def __init__(self, domain, parameters, Omega=None, sponge_options=None,
                 extra_terms=None, space_names=None,
                 linearisation_map='default',
                 u_transport_option="vector_invariant_form",
                 diffusion_options=None,
                 no_normal_flow_bc_ids=None,
                 active_tracers=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            parameters (:class:`Configuration`, optional): an object containing
                the model's physical parameters.
            Omega (:class:`ufl.Expr`, optional): an expression for the planet's
                rotation vector. Defaults to None.
            sponge_options (:class:`SpongeLayerParameters`, optional): any
                parameters for applying a sponge layer to the upper boundary.
                Defaults to None.
            extra_terms (:class:`ufl.Expr`, optional): any extra terms to be
                included in the equation set. Defaults to None.
            space_names (dict, optional): a dictionary of strings for names of
                the function spaces to use for the spatial discretisation. The
                keys are the names of the prognostic variables. Defaults to None
                in which case the spaces are taken from the de Rham complex.
            linearisation_map (func, optional): a function specifying which
                terms in the equation set to linearise. If None is specified
                then no terms are linearised. Defaults to the string 'default',
                in which case the linearisation includes time derivatives and
                scalar transport terms.
            u_transport_option (str, optional): specifies the transport term
                used for the velocity equation. Supported options are:
                'vector_invariant_form', 'vector_advection_form' and
                'circulation_form'.
                Defaults to 'vector_invariant_form'.
            diffusion_options (:class:`DiffusionParameters`, optional): any
                options to specify for applying diffusion terms to variables.
                Defaults to None.
            no_normal_flow_bc_ids (list, optional): a list of IDs of domain
                boundaries at which no normal flow will be enforced. Defaults to
                None.
            active_tracers (list, optional): a list of `ActiveTracer` objects
                that encode the metadata for any active tracers to be included
                in the equations.. Defaults to None.

        Raises:
            NotImplementedError: only mixing ratio tracers are implemented.
        """

        field_names = ['u', 'rho', 'theta']

        if space_names is None:
            space_names = {'u': 'HDiv', 'rho': 'L2', 'theta': 'theta'}

        if active_tracers is None:
            active_tracers = []

        if linearisation_map == 'default':
            # Default linearisation is time derivatives and scalar transport terms
            # Don't include active tracers
            linearisation_map = lambda t: \
                t.get(prognostic) in ['u', 'rho', 'theta'] \
                and (t.has_label(time_derivative)
                     or (t.get(prognostic) != 'u' and t.has_label(transport)))
        super().__init__(field_names, domain, space_names,
                         linearisation_map=linearisation_map,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        self.parameters = parameters
        g = parameters.g
        cp = parameters.cp

        w, phi, gamma = self.tests[0:3]
        u, rho, theta = split(self.X)[0:3]
        u_trial = split(self.trials)[0]
        _, rho_bar, theta_bar = split(self.X_ref)[0:3]
        zero_expr = Constant(0.0)*theta
        exner = exner_pressure(parameters, rho, theta)
        n = FacetNormal(domain.mesh)

        # -------------------------------------------------------------------- #
        # Time Derivative Terms
        # -------------------------------------------------------------------- #
        mass_form = self.generate_mass_terms()

        # -------------------------------------------------------------------- #
        # Transport Terms
        # -------------------------------------------------------------------- #
        # Velocity transport term -- depends on formulation
        if u_transport_option == "vector_invariant_form":
            u_adv = prognostic(vector_invariant_form(domain, w, u, u), 'u')
        elif u_transport_option == "vector_advection_form":
            u_adv = prognostic(advection_form(w, u, u), 'u')
        elif u_transport_option == "circulation_form":
            ke_form = prognostic(kinetic_energy_form(w, u, u), 'u')
            u_adv = prognostic(advection_equation_circulation_form(domain, w, u, u), 'u') + ke_form
        else:
            raise ValueError("Invalid u_transport_option: %s" % u_transport_option)

        # Density transport (conservative form)
        rho_adv = prognostic(continuity_form(phi, rho, u), 'rho')
        # Transport term needs special linearisation
        if self.linearisation_map(rho_adv.terms[0]):
            linear_rho_adv = linear_continuity_form(phi, rho_bar, u_trial)
            rho_adv = linearisation(rho_adv, linear_rho_adv)

        # Potential temperature transport (advective form)
        theta_adv = prognostic(advection_form(gamma, theta, u), 'theta')
        # Transport term needs special linearisation
        if self.linearisation_map(theta_adv.terms[0]):
            linear_theta_adv = linear_advection_form(gamma, theta_bar, u_trial)
            theta_adv = linearisation(theta_adv, linear_theta_adv)

        adv_form = subject(u_adv + rho_adv + theta_adv, self.X)

        # Add transport of tracers
        if len(active_tracers) > 0:
            adv_form += self.generate_tracer_transport_terms(active_tracers)

        # -------------------------------------------------------------------- #
        # Pressure Gradient Term
        # -------------------------------------------------------------------- #
        # First get total mass of tracers
        tracer_mr_total = zero_expr
        for tracer in active_tracers:
            if tracer.variable_type == TracerVariableType.mixing_ratio:
                idx = self.field_names.index(tracer.name)
                tracer_mr_total += split(self.X)[idx]
            else:
                raise NotImplementedError('Only mixing ratio tracers are implemented')
        theta_v = theta / (Constant(1.0) + tracer_mr_total)

        pressure_gradient_form = pressure_gradient(subject(prognostic(
            cp*(-div(theta_v*w)*exner*dx
                + jump(theta_v*w, n)*avg(exner)*dS_v), 'u'), self.X))

        # -------------------------------------------------------------------- #
        # Gravitational Term
        # -------------------------------------------------------------------- #
        gravity_form = gravity(subject(prognostic(g*inner(domain.k, w)*dx,
                                                  'u'), self.X))

        residual = (mass_form + adv_form + pressure_gradient_form
                    + gravity_form)

        # -------------------------------------------------------------------- #
        # Moist Thermodynamic Divergence Term
        # -------------------------------------------------------------------- #
        if len(active_tracers) > 0:
            cv = parameters.cv
            c_vv = parameters.c_vv
            c_pv = parameters.c_pv
            c_pl = parameters.c_pl
            R_d = parameters.R_d
            R_v = parameters.R_v

            # Get gas and liquid moisture mixing ratios
            mr_l = zero_expr
            mr_v = zero_expr

            for tracer in active_tracers:
                if tracer.chemical == 'H2O':
                    if tracer.variable_type == TracerVariableType.mixing_ratio:
                        idx = self.field_names.index(tracer.name)
                        if tracer.phase == Phases.gas:
                            mr_v += split(self.X)[idx]
                        elif tracer.phase == Phases.liquid:
                            mr_l += split(self.X)[idx]
                    else:
                        raise NotImplementedError('Only mixing ratio tracers are implemented')

            c_vml = cv + mr_v * c_vv + mr_l * c_pl
            c_pml = cp + mr_v * c_pv + mr_l * c_pl
            R_m = R_d + mr_v * R_v

            residual += subject(prognostic(
                gamma * theta * div(u)
                * (R_m / c_vml - (R_d * c_pml) / (cp * c_vml))*dx, 'theta'), self.X)

        # -------------------------------------------------------------------- #
        # Extra Terms (Coriolis, Sponge, Diffusion and others)
        # -------------------------------------------------------------------- #
        if Omega is not None:
            # TODO: add linearisation
            residual += coriolis(subject(prognostic(
                inner(w, cross(2*Omega, u))*dx, "u"), self.X))

        if sponge_options is not None:
            W_DG = FunctionSpace(domain.mesh, "DG", 2)
            x = SpatialCoordinate(domain.mesh)
            z = x[len(x)-1]
            H = sponge_options.H
            zc = sponge_options.z_level
            assert float(zc) < float(H), \
                "The sponge level is set above the height the your domain"
            mubar = sponge_options.mubar
            muexpr = conditional(z <= zc,
                                 0.0,
                                 mubar*sin((pi/2.)*(z-zc)/(H-zc))**2)
            self.mu = self.prescribed_fields("sponge", W_DG).interpolate(muexpr)

            residual += sponge(subject(prognostic(
                self.mu*inner(w, domain.k)*inner(u, domain.k)*dx, 'u'), self.X))

        if diffusion_options is not None:
            for field, diffusion in diffusion_options:
                idx = self.field_names.index(field)
                test = self.tests[idx]
                fn = split(self.X)[idx]
                residual += subject(
                    prognostic(diffusion_form(test, fn, diffusion.kappa), field),
                    self.X)

        if extra_terms is not None:
            for field, term in extra_terms:
                idx = self.field_names.index(field)
                test = self.tests[idx]
                residual += subject(prognostic(
                    inner(test, term)*dx, field), self.X)

        # -------------------------------------------------------------------- #
        # Linearise equations
        # -------------------------------------------------------------------- #
        # Add linearisations to equations
        self.residual = self.generate_linear_terms(residual, self.linearisation_map)


class HydrostaticCompressibleEulerEquations(CompressibleEulerEquations):
    """
    The hydrostatic form of the compressible Euler equations. In this case the
    vertical velocity derivative is zero in the equations, so only 'u_h', the
    horizontal component of the velocity is allowed to vary in time. The
    equations, for velocity 'u', dry density 'rho' and (dry) potential
    temperature 'theta' are:                                                  \n
    ∂u_h/∂t + (u.∇)u_h + 2Ω×u + c_p*θ*∇Π + g = 0,                             \n
    ∂ρ/∂t + ∇.(ρ*u) = 0,                                                      \n
    ∂θ/∂t + (u.∇)θ = 0,                                                       \n
    where Π is the Exner pressure, g is the gravitational vector, Ω is the
    planet's rotation vector and c_p is the heat capacity of dry air at constant
    pressure.

    This is implemented through a hydrostatic switch to the compressible Euler
    equations.
    """

    def __init__(self, domain, parameters, Omega=None, sponge=None,
                 extra_terms=None, space_names=None, linearisation_map='default',
                 u_transport_option="vector_invariant_form",
                 diffusion_options=None,
                 no_normal_flow_bc_ids=None,
                 active_tracers=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            parameters (:class:`Configuration`, optional): an object containing
                the model's physical parameters.
            Omega (:class:`ufl.Expr`, optional): an expression for the planet's
                rotation vector. Defaults to None.
            sponge (:class:`ufl.Expr`, optional): an expression for a sponge
                layer. Defaults to None.
            extra_terms (:class:`ufl.Expr`, optional): any extra terms to be
                included in the equation set. Defaults to None.
            space_names (dict, optional): a dictionary of strings for names of
                the function spaces to use for the spatial discretisation. The
                keys are the names of the prognostic variables. Defaults to None
                in which case the spaces are taken from the de Rham complex.
            linearisation_map (func, optional): a function specifying which
                terms in the equation set to linearise. If None is specified
                then no terms are linearised. Defaults to the string 'default',
                in which case the linearisation includes time derivatives and
                scalar transport terms.
            u_transport_option (str, optional): specifies the transport term
                used for the velocity equation. Supported options are:
                'vector_invariant_form', 'vector_advection_form' and
                'circulation_form'.
                Defaults to 'vector_invariant_form'.
            diffusion_options (:class:`DiffusionOptions`, optional): any options
                to specify for applying diffusion terms to variables. Defaults
                to None.
            no_normal_flow_bc_ids (list, optional): a list of IDs of domain
                boundaries at which no normal flow will be enforced. Defaults to
                None.
            active_tracers (list, optional): a list of `ActiveTracer` objects
                that encode the metadata for any active tracers to be included
                in the equations.. Defaults to None.

        Raises:
            NotImplementedError: only mixing ratio tracers are implemented.
        """

        super().__init__(domain, parameters, Omega=Omega, sponge=sponge,
                         extra_terms=extra_terms, space_names=space_names,
                         linearisation_map=linearisation_map,
                         u_transport_option=u_transport_option,
                         diffusion_options=diffusion_options,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        self.residual = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=lambda t: hydrostatic(t, self.hydrostatic_projection(t))
        )

        k = self.domain.k
        u = split(self.X)[0]
        self.residual += hydrostatic(
            subject(
                prognostic(
                    -inner(k, self.tests[0]) * inner(k, u) * dx, "u"),
                self.X))

    def hydrostatic_projection(self, t):
        """
        Performs the 'hydrostatic' projection.

        Takes a term involving a vector prognostic variable and replaces the
        prognostic with only its horizontal components.

        Args:
            t (:class:`Term`): the term to perform the projection upon.

        Returns:
            :class:`LabelledForm`: the labelled form containing the new term.

        Raises:
            AssertionError: spherical geometry is not yet implemented.
        """

        # TODO: make this more general, i.e. should work on the sphere
        if self.domain.on_sphere:
            raise NotImplementedError("The hydrostatic projection is not yet "
                                      + "implemented for spherical geometry")
        k = Constant((*self.domain.k, 0, 0))
        X = t.get(subject)

        new_subj = X - k * inner(X, k)
        return replace_subject(new_subj)(t)


class BoussinesqEquations(PrognosticEquationSet):
    """
    Class for the Boussinesq equations, which evolve the velocity
    'u', the pressure 'p' and the buoyancy 'b'. Can be compressible or
    incompressible, depending on the value of the input flag, which defaults
    to compressible.

    The compressible form of the equations is
    ∂u/∂t + (u.∇)u + 2Ω×u + ∇p + b*k = 0,                                     \n
    ∂p/∂t + cs**2 ∇.u = p,                                                    \n
    ∂b/∂t + (u.∇)b = 0,                                                       \n
    where k is the vertical unit vector, Ω is the planet's rotation vector
    and cs is the sound speed.

    For the incompressible form of the equations, the pressure features as
    a Lagrange multiplier to enforce incompressibility. The equations are     \n
    ∂u/∂t + (u.∇)u + 2Ω×u + ∇p + b*k = 0,                                     \n
    ∇.u = p,                                                                  \n
    ∂b/∂t + (u.∇)b = 0,                                                       \n
    where k is the vertical unit vector and Ω is the planet's rotation vector.
    """

    def __init__(self, domain, parameters,
                 compressible=True,
                 Omega=None,
                 space_names=None,
                 linearisation_map='default',
                 u_transport_option="vector_invariant_form",
                 no_normal_flow_bc_ids=None,
                 active_tracers=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            parameters (:class:`Configuration`, optional): an object containing
                the model's physical parameters.
            compressible (bool, optional): flag to indicate whether the
                equations are compressible. Defaults to True
            Omega (:class:`ufl.Expr`, optional): an expression for the planet's
                rotation vector. Defaults to None.
            space_names (dict, optional): a dictionary of strings for names of
                the function spaces to use for the spatial discretisation. The
                keys are the names of the prognostic variables. Defaults to None
                in which case the spaces are taken from the de Rham complex.
            linearisation_map (func, optional): a function specifying which
                terms in the equation set to linearise. If None is specified
                then no terms are linearised. Defaults to the string 'default',
                in which case the linearisation includes time derivatives and
                scalar transport terms.
            u_transport_option (str, optional): specifies the transport term
                used for the velocity equation. Supported options are:
                'vector_invariant_form', 'vector_advection_form' and
                'circulation_form'.
                Defaults to 'vector_invariant_form'.
            no_normal_flow_bc_ids (list, optional): a list of IDs of domain
                boundaries at which no normal flow will be enforced. Defaults to
                None.
            active_tracers (list, optional): a list of `ActiveTracer` objects
                that encode the metadata for any active tracers to be included
                in the equations.. Defaults to None.

        Raises:
            NotImplementedError: active tracers are not implemented.
        """

        field_names = ['u', 'p', 'b']

        if space_names is None:
            space_names = {'u': 'HDiv', 'p': 'L2', 'b': 'theta'}

        if active_tracers is not None:
            raise NotImplementedError('Tracers not implemented for Boussinesq equations')

        if active_tracers is None:
            active_tracers = []

        if linearisation_map == 'default':
            # Default linearisation is time derivatives and scalar transport terms
            # Don't include active tracers
            linearisation_map = lambda t: \
                t.get(prognostic) in ['u', 'p', 'b'] \
                and (t.has_label(time_derivative)
                     or (t.get(prognostic) not in ['u', 'p'] and t.has_label(transport)))

        super().__init__(field_names, domain, space_names,
                         linearisation_map=linearisation_map,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        self.parameters = parameters
        self.compressible = compressible

        w, phi, gamma = self.tests[0:3]
        u, p, b = split(self.X)
        u_trial, p_trial, _ = split(self.trials)
        _, p_bar, b_bar = split(self.X_ref)

        # -------------------------------------------------------------------- #
        # Time Derivative Terms
        # -------------------------------------------------------------------- #
        mass_form = self.generate_mass_terms()

        # -------------------------------------------------------------------- #
        # Transport Terms
        # -------------------------------------------------------------------- #
        # Velocity transport term -- depends on formulation
        if u_transport_option == "vector_invariant_form":
            u_adv = prognostic(vector_invariant_form(domain, w, u, u), 'u')
        elif u_transport_option == "vector_advection_form":
            u_adv = prognostic(advection_form(w, u, u), 'u')
        elif u_transport_option == "circulation_form":
            ke_form = prognostic(kinetic_energy_form(w, u, u), 'u')
            u_adv = prognostic(advection_equation_circulation_form(domain, w, u, u), 'u') + ke_form
        else:
            raise ValueError("Invalid u_transport_option: %s" % u_transport_option)

        # Buoyancy transport
        b_adv = prognostic(advection_form(gamma, b, u), 'b')
        if self.linearisation_map(b_adv.terms[0]):
            linear_b_adv = linear_advection_form(gamma, b_bar, u_trial)
            b_adv = linearisation(b_adv, linear_b_adv)

        if compressible:
            # Pressure transport
            p_adv = prognostic(advection_form(phi, p, u), 'p')
            if self.linearisation_map(p_adv.terms[0]):
                linear_p_adv = linear_advection_form(phi, p_bar, u_trial)
                p_adv = linearisation(p_adv, linear_p_adv)
            adv_form = subject(u_adv + p_adv + b_adv, self.X)
        else:
            adv_form = subject(u_adv + b_adv, self.X)

        # Add transport of tracers
        if len(active_tracers) > 0:
            adv_form += self.generate_tracer_transport_terms(active_tracers)

        # -------------------------------------------------------------------- #
        # Pressure Gradient Term
        # -------------------------------------------------------------------- #
        pressure_gradient_form = pressure_gradient(subject(prognostic(
            -div(w)*p*dx, 'u'), self.X))

        # -------------------------------------------------------------------- #
        # Gravitational Term
        # -------------------------------------------------------------------- #
        gravity_form = gravity(subject(prognostic(
            -b*inner(w, domain.k)*dx, 'u'), self.X))

        # -------------------------------------------------------------------- #
        # Divergence Term
        # -------------------------------------------------------------------- #
        if compressible:
            cs = parameters.cs
            linear_div_form = divergence(subject(
                prognostic(cs**2 * phi * div(u_trial) * dx, 'p'), self.X))
            divergence_form = divergence(linearisation(
                subject(prognostic(cs**2 * phi * div(u) * dx, 'p'), self.X),
                linear_div_form))
        else:
            # This enforces that div(u) = 0
            # The p features here so that the div(u) evaluated in the
            # "forcing" step replaces the whole pressure field, rather than
            # merely providing an increment to it.
            linear_div_form = incompressible(
                subject(prognostic(phi*(p_trial-div(u_trial))*dx, 'p'), self.X))
            divergence_form = incompressible(linearisation(
                subject(prognostic(phi*(p-div(u))*dx, 'p'), self.X),
                linear_div_form))

        residual = (mass_form + adv_form + divergence_form
                    + pressure_gradient_form + gravity_form)

        # -------------------------------------------------------------------- #
        # Extra Terms (Coriolis)
        # -------------------------------------------------------------------- #
        if Omega is not None:
            # TODO: add linearisation
            residual += coriolis(subject(prognostic(
                inner(w, cross(2*Omega, u))*dx, 'u'), self.X))

        # -------------------------------------------------------------------- #
        # Linearise equations
        # -------------------------------------------------------------------- #
        # Add linearisations to equations
        self.residual = self.generate_linear_terms(residual, self.linearisation_map)


class LinearBoussinesqEquations(BoussinesqEquations):
    """
    Class for the Boussinesq equations, which evolve the velocity
    'u', the pressure 'p' and the buoyancy 'b'. Can be compressible or
    incompressible, depending on the value of the input flag, which defaults
    to compressible.

    The compressible form of the equations is
    ∂u/∂t + (u.∇)u + 2Ω×u + ∇p + b*k = 0,                                     \n
    ∂p/∂t + cs**2 ∇.u = p,                                                    \n
    ∂b/∂t + (u.∇)b = 0,                                                       \n
    where k is the vertical unit vector, Ω is the planet's rotation vector
    and cs is the sound speed.

    For the incompressible form of the equations, the pressure features as
    a Lagrange multiplier to enforce incompressibility. The equations are     \n
    ∂u/∂t + (u.∇)u + 2Ω×u + ∇p + b*k = 0,                                     \n
    ∇.u = p,                                                                  \n
    ∂b/∂t + (u.∇)b = 0,                                                       \n
    where k is the vertical unit vector and Ω is the planet's rotation vector.
    """

    def __init__(self, domain, parameters,
                 compressible=True,
                 Omega=None,
                 space_names=None,
                 linearisation_map='default',
                 u_transport_option="vector_invariant_form",
                 no_normal_flow_bc_ids=None,
                 active_tracers=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            parameters (:class:`Configuration`, optional): an object containing
                the model's physical parameters.
            compressible (bool, optional): flag to indicate whether the
                equations are compressible. Defaults to True
            Omega (:class:`ufl.Expr`, optional): an expression for the planet's
                rotation vector. Defaults to None.
            space_names (dict, optional): a dictionary of strings for names of
                the function spaces to use for the spatial discretisation. The
                keys are the names of the prognostic variables. Defaults to None
                in which case the spaces are taken from the de Rham complex.
            linearisation_map (func, optional): a function specifying which
                terms in the equation set to linearise. If None is specified
                then no terms are linearised. Defaults to the string 'default',
                in which case the linearisation includes time derivatives and
                scalar transport terms.
            u_transport_option (str, optional): specifies the transport term
                used for the velocity equation. Supported options are:
                'vector_invariant_form', 'vector_advection_form' and
                'circulation_form'.
                Defaults to 'vector_invariant_form'.
            no_normal_flow_bc_ids (list, optional): a list of IDs of domain
                boundaries at which no normal flow will be enforced. Defaults to
                None.
            active_tracers (list, optional): a list of `ActiveTracer` objects
                that encode the metadata for any active tracers to be included
                in the equations.. Defaults to None.

        Raises:
            NotImplementedError: active tracers are not implemented.
        """

        if linearisation_map == 'default':
            # Default linearisation is time derivatives, pressure gradient,
            # Coriolis and transport term from depth equation
            linearisation_map = lambda t: \
                (any(t.has_label(time_derivative, pressure_gradient, coriolis,
                                 gravity, divergence, incompressible))
                 or (t.get(prognostic) in ['p', 'b'] and t.has_label(transport)))
        super().__init__(domain=domain,
                         parameters=parameters,
                         compressible=compressible,
                         Omega=Omega,
                         space_names=space_names,
                         linearisation_map=linearisation_map,
                         u_transport_option=u_transport_option,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        # Use the underlying routine to do a first linearisation of
        # the equations
        self.linearise_equation_set()
