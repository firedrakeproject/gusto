"""Objects describing geophysical fluid equations to be solved in weak form."""

from abc import ABCMeta
from firedrake import (TestFunction, Function, sin, pi, inner, dx, div, cross,
                       FunctionSpace, MixedFunctionSpace, TestFunctions,
                       TrialFunction, FacetNormal, jump, avg, dS_v, dS,
                       DirichletBC, conditional, SpatialCoordinate,
                       split, Constant, action)
from gusto.fml.form_manipulation_labelling import Term, all_terms, identity, drop
from gusto.labels import (subject, time_derivative, transport, prognostic,
                          transporting_velocity, replace_subject, linearisation,
                          name, pressure_gradient, coriolis,
                          replace_trial_function, hydrostatic)
from gusto.thermodynamics import exner_pressure
from gusto.transport_forms import (advection_form, continuity_form,
                                   vector_invariant_form,
                                   vector_manifold_advection_form,
                                   kinetic_energy_form,
                                   advection_equation_circulation_form,
                                   linear_continuity_form,
                                   linear_advection_form)
from gusto.diffusion import interior_penalty_diffusion_form
from gusto.active_tracers import ActiveTracer, Phases, TracerVariableType
from gusto.configuration import TransportEquationType
import ufl


class PrognosticEquation(object, metaclass=ABCMeta):
    """Base class for prognostic equations."""

    def __init__(self, state, function_space, field_name):
        """
        Args:
            state (:class:`State`): the model's state object.
            function_space (:class:`FunctionSpace`): the function space that the
                equation's prognostic is defined on.
            field_name (str): name of the prognostic field.
        """

        self.state = state
        self.function_space = function_space
        self.field_name = field_name
        self.bcs = {}

        if len(function_space) > 1:
            assert hasattr(self, "field_names")
            state.fields(field_name, function_space,
                         subfield_names=self.field_names, pickup=True)
            for fname in self.field_names:
                state.diagnostics.register(fname)
                self.bcs[fname] = []
        else:
            state.fields(field_name, function_space)
            state.diagnostics.register(field_name)
            self.bcs[field_name] = []


class AdvectionEquation(PrognosticEquation):
    u"""Discretises the advection equation, ∂q/∂t + (u.∇)q = 0"""

    def __init__(self, state, function_space, field_name,
                 ufamily=None, udegree=None, Vu=None, **kwargs):
        """
        Args:
            state (:class:`State`): the model's state object.
            function_space (:class:`FunctionSpace`): the function space that the
                equation's prognostic is defined on.
            field_name (str): name of the prognostic field.
            ufamily (str, optional): the family of the function space to use
                for the velocity field. Only used if `Vu` is not provided.
                Defaults to None.
            udegree (int, optional): the degree of the function space to use for
                the velocity field. Only used if `Vu` is not provided. Defaults
                to None.
            Vu (:class:`FunctionSpace`, optional): the function space for the
                velocity field. If this is  Defaults to None.
            **kwargs: any keyword arguments to be passed to the advection form.
        """
        super().__init__(state, function_space, field_name)

        if not hasattr(state.fields, "u"):
            if Vu is not None:
                V = state.spaces("HDiv", V=Vu)
            else:
                assert ufamily is not None, "Specify the family for u"
                assert udegree is not None, "Specify the degree of the u space"
                V = state.spaces("HDiv", ufamily, udegree)
            state.fields("u", V)
        test = TestFunction(function_space)
        q = Function(function_space)
        mass_form = time_derivative(inner(q, test)*dx)

        self.residual = subject(
            mass_form + advection_form(state, test, q, **kwargs), q
        )


class ContinuityEquation(PrognosticEquation):
    u"""Discretises the continuity equation, ∂q/∂t + ∇(u*q) = 0"""

    def __init__(self, state, function_space, field_name,
                 ufamily=None, udegree=None, Vu=None, **kwargs):
        """
        Args:
            state (:class:`State`): the model's state object.
            function_space (:class:`FunctionSpace`): the function space that the
                equation's prognostic is defined on.
            field_name (str): name of the prognostic field.
            ufamily (str, optional): the family of the function space to use
                for the velocity field. Only used if `Vu` is not provided.
                Defaults to None.
            udegree (int, optional): the degree of the function space to use for
                the velocity field. Only used if `Vu` is not provided. Defaults
                to None.
            Vu (:class:`FunctionSpace`, optional): the function space for the
                velocity field. If this is  Defaults to None.
            **kwargs: any keyword arguments to be passed to the advection form.
        """
        super().__init__(state, function_space, field_name)

        if not hasattr(state.fields, "u"):
            if Vu is not None:
                V = state.spaces("HDiv", V=Vu)
            else:
                assert ufamily is not None, "Specify the family for u"
                assert udegree is not None, "Specify the degree of the u space"
                V = state.spaces("HDiv", ufamily, udegree)
            state.fields("u", V)
        test = TestFunction(function_space)
        q = Function(function_space)
        mass_form = time_derivative(inner(q, test)*dx)

        self.residual = subject(
            mass_form + continuity_form(state, test, q, **kwargs), q
        )


class DiffusionEquation(PrognosticEquation):
    u"""Discretises the diffusion equation, ∂q/∂t = ∇.(κ∇q)"""

    def __init__(self, state, function_space, field_name,
                 diffusion_parameters):
        """
        Args:
            state (:class:`State`): the model's state object.
            function_space (:class:`FunctionSpace`): the function space that the
                equation's prognostic is defined on.
            field_name (str): name of the prognostic field.
            diffusion_parameters (:class:`DiffusionParameters`): parameters
                describing the diffusion to be applied.
        """
        super().__init__(state, function_space, field_name)

        test = TestFunction(function_space)
        q = Function(function_space)
        mass_form = time_derivative(inner(q, test)*dx)

        self.residual = subject(
            mass_form
            + interior_penalty_diffusion_form(
                state, test, q, diffusion_parameters), q
        )


class AdvectionDiffusionEquation(PrognosticEquation):
    u"""The advection-diffusion equation, ∂q/∂t + (u.∇)q = ∇.(κ∇q)"""

    def __init__(self, state, function_space, field_name,
                 ufamily=None, udegree=None, Vu=None, diffusion_parameters=None,
                 **kwargs):
        """
        Args:
            state (:class:`State`): the model's state object.
            function_space (:class:`FunctionSpace`): the function space that the
                equation's prognostic is defined on.
            field_name (str): name of the prognostic field.
            ufamily (str, optional): the family of the function space to use
                for the velocity field. Only used if `Vu` is not provided.
                Defaults to None.
            udegree (int, optional): the degree of the function space to use for
                the velocity field. Only used if `Vu` is not provided. Defaults
                to None.
            Vu (:class:`FunctionSpace`, optional): the function space for the
                velocity field. If this is  Defaults to None.
            diffusion_parameters (:class:`DiffusionParameters`, optional):
                parameters describing the diffusion to be applied.
            **kwargs: any keyword arguments to be passed to the advection form.
        """

        super().__init__(state, function_space, field_name)

        if not hasattr(state.fields, "u"):
            if Vu is not None:
                V = state.spaces("HDiv", V=Vu)
            else:
                assert ufamily is not None, "Specify the family for u"
                assert udegree is not None, "Specify the degree of the u space"
                V = state.spaces("HDiv", ufamily, udegree)
            state.fields("u", V)
        test = TestFunction(function_space)
        q = Function(function_space)
        mass_form = time_derivative(inner(q, test)*dx)

        self.residual = subject(
            mass_form
            + advection_form(state, test, q, **kwargs)
            + interior_penalty_diffusion_form(
                state, test, q, diffusion_parameters), q
        )


class PrognosticEquationSet(PrognosticEquation, metaclass=ABCMeta):
    """
    Base class for solving a set of prognostic equations.

    A prognostic equation set contains multiple prognostic variables, which are
    solved for simultaneously in a :class:`MixedFunctionSpace`. This base class
    contains common routines for these equation sets.
    """

    def __init__(self, field_names, state, family, degree,
                 terms_to_linearise=None,
                 no_normal_flow_bc_ids=None, active_tracers=None):
        """
        Args:
            field_names (list): a list of strings for names of the prognostic
                variables for the equation set.
            state (:class:`State`): the model's state object.
            family (str): the finite element space family used for the velocity
                field. This determines the other finite element spaces used via
                the de Rham complex.
            degree (int): the element degree used for the velocity space.
            terms_to_linearise (dict, optional): a dictionary specifying which
                terms in the equation set to linearise. Defaults to None.
            no_normal_flow_bc_ids (list, optional): a list of IDs of domain
                boundaries at which no normal flow will be enforced. Defaults to
                None.
            active_tracers (list, optional): a list of `ActiveTracer` objects
                that encode the metadata for any active tracers to be included
                in the equations.. Defaults to None.
        """

        self.field_names = field_names
        self.active_tracers = active_tracers
        self.terms_to_linearise = {} if terms_to_linearise is None else terms_to_linearise

        # Build finite element spaces
        self.spaces = [space for space in self._build_spaces(state, family, degree)]

        # Add active tracers to the list of prognostics
        if active_tracers is None:
            active_tracers = []
        self.add_tracers_to_prognostics(state, active_tracers)

        # Make the full mixed function space
        W = MixedFunctionSpace(self.spaces)

        # Can now call the underlying PrognosticEquation
        full_field_name = "_".join(self.field_names)
        super().__init__(state, W, full_field_name)

        # Set up test functions, trials and prognostics
        self.tests = TestFunctions(W)
        self.trials = TrialFunction(W)
        self.X = Function(W)
        self.X_ref = Function(W)

        # Set up no-normal-flow boundary conditions
        if no_normal_flow_bc_ids is None:
            no_normal_flow_bc_ids = []
        self.set_no_normal_flow_bcs(state, no_normal_flow_bc_ids)

    def _build_spaces(self, state, family, degree):
        return state.spaces.build_compatible_spaces(family, degree)

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

        for i, (test, field_name) in enumerate(zip(self.tests, self.field_names)):
            prog = split(self.X)[i]
            mass = subject(prognostic(inner(prog, test)*dx, field_name), self.X)
            if i == 0:
                mass_form = time_derivative(mass)
            else:
                mass_form += time_derivative(mass)

        return mass_form

    # ======================================================================== #
    # Linearisation Routines
    # ======================================================================== #

    def generate_linear_terms(self, residual, terms_to_linearise):
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
            terms_to_linearise (dict): a dictionary describing the terms to be
                linearised.

        Returns:
            :class:`LabelledForm`: the residual with linear terms attached to
                each term as labels.
        """

        # TODO: Neaten up the `terms_to_linearise` variable. This should not be
        # a dictionary, it should be a filter of some sort

        from functools import partial

        # Function to check if term should be linearised
        def should_linearise(term, field):
            return (not term.has_label(linearisation)
                    and term.get(prognostic) == field
                    and any(term.has_label(*terms_to_linearise[field], return_tuple=True))
                    )

        # Linearise a term, and add the linearisation as a label
        def linearise(term, X, X_ref, du):
            linear_term = Term(action(ufl.derivative(term.form, X), du), term.labels)
            return linearisation(term, replace_subject(X_ref)(linear_term))

        # Add linearisations to all terms that need linearising
        for field in self.field_names:
            residual = residual.label_map(
                partial(should_linearise, field=field),
                map_if_true=partial(linearise, X=self.X, X_ref=self.X_ref, du=self.trials),
                map_if_false=identity,  # TODO: should "keep" be an alias for identity?
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

    def set_no_normal_flow_bcs(self, state, no_normal_flow_bc_ids):
        """
        Sets up the boundary conditions for no-normal flow at domain boundaries.

        Sets up the no-normal-flow boundary conditions, storing the
        :class:`DirichletBC` object at each specified boundary. There must be
        a velocity variable named 'u' to apply the boundary conditions to.

        Args:
            state (:class:`State`): the model's state.
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

        Vu = state.spaces("HDiv")
        if Vu.extruded:
            self.bcs['u'].append(DirichletBC(Vu, 0.0, "bottom"))
            self.bcs['u'].append(DirichletBC(Vu, 0.0, "top"))
        for id in no_normal_flow_bc_ids:
            self.bcs['u'].append(DirichletBC(Vu, 0.0, id))

    # ======================================================================== #
    # Active Tracer Routines
    # ======================================================================== #

    def add_tracers_to_prognostics(self, state, active_tracers):
        """
        Augments the equation set with specified active tracer variables.

        Args:
            state (:class:`State`): the model's state.
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
                self.spaces.append(state.spaces(tracer.space))
                # Add an item to the terms_to_linearise dictionary
                if tracer.name not in self.terms_to_linearise.keys():
                    self.terms_to_linearise[tracer.name] = []
            else:
                raise TypeError(f'Tracers must be ActiveTracer objects, not {type(tracer)}')

    def generate_tracer_transport_terms(self, state, active_tracers):
        """
        Adds the transport forms for the active tracers to the equation set.

        Args:
            state (:class:`State`): the model's state.
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

        for i, tracer in enumerate(active_tracers):
            if tracer.transport_eqn != TransportEquationType.no_transport:
                idx = self.field_names.index(tracer.name)
                tracer_prog = split(self.X)[idx]
                tracer_test = self.tests[idx]
                if tracer.transport_eqn == TransportEquationType.advective:
                    tracer_adv = prognostic(advection_form(state, tracer_test, tracer_prog), tracer.name)
                elif tracer.transport_eqn == TransportEquationType.conservative:
                    tracer_adv = prognostic(continuity_form(state, tracer_test, tracer_prog), tracer.name)
                else:
                    raise ValueError(f'Transport eqn {tracer.transport_eqn} not recognised')

                if no_tracer_transported:
                    # We arrive here for the first tracer to be transported
                    adv_form = subject(tracer_adv, self.X)
                    no_tracer_transported = False
                else:
                    adv_form += subject(tracer_adv, self.X)

        return adv_form


class ForcedAdvectionEquation(PrognosticEquationSet):

    def __init__(self, state, function_space, field_name,
                 ufamily=None, udegree=None, Vu=None, active_tracers=None,
                 **kwargs):

        self.field_names = [field_name]
        self.active_tracers = active_tracers
        self.terms_to_linearise = {}

        # Build finite element spaces
        self.spaces = [state.spaces("tracer", V=function_space)]

        # Add active tracers to the list of prognostics
        if active_tracers is None:
            active_tracers = []
        self.add_tracers_to_prognostics(state, active_tracers)

        # Make the full mixed function space
        W = MixedFunctionSpace(self.spaces)

        # Can now call the underlying PrognosticEquation
        full_field_name = "_".join(self.field_names)
        PrognosticEquation.__init__(self, state, W, full_field_name)

        if not hasattr(state.fields, "u"):
            if Vu is not None:
                V = state.spaces("HDiv", V=Vu)
            else:
                assert ufamily is not None, "Specify the family for u"
                assert udegree is not None, "Specify the degree of the u space"
                V = state.spaces("HDiv", ufamily, udegree)
            state.fields("u", V)

        self.tests = TestFunctions(W)
        self.X = Function(W)

        mass_form = self.generate_mass_terms()

        self.residual = subject(
            mass_form + advection_form(state, self.tests[0], split(self.X)[0], **kwargs), self.X
        )

# ============================================================================ #
# Specified Equation Sets
# ============================================================================ #


class ShallowWaterEquations(PrognosticEquationSet):
    u"""
    Class for the (rotating) shallow-water equations, which evolve the velocity
    'u' and the depth field 'D', via some variant of:
        ∂u/∂t + (u.∇)u + f×u + g*∇(D+b) = 0
        ∂D/∂t + ∇.(D*u) = 0
    for Coriolis parameter 'f' and bottom surface 'b'.
    """

    def __init__(self, state, family, degree, fexpr=None, bexpr=None,
                 terms_to_linearise={'D': [time_derivative, transport],
                                     'u': [time_derivative, pressure_gradient]},
                 u_transport_option='vector_invariant_form',
                 no_normal_flow_bc_ids=None, active_tracers=None,
                 thermal=False):
        """
        Args:
            state (:class:`State`): the model's state object.
            family (str): the finite element space family used for the velocity
                field. This determines the other finite element spaces used via
                the de Rham complex.
            degree (int): the element degree used for the velocity space.
            fexpr (:class:`ufl.Expr`, optional): an expression for the Coroilis
                parameter. Defaults to None.
            bexpr (:class:`ufl.Expr`, optional): an expression for the bottom
                surface of the fluid. Defaults to None.
            terms_to_linearise (dict, optional): a dictionary specifying which
                terms in the equation set to linearise. By default, includes
                both time derivatives, the 'D' transport term and the pressure
                gradient term.
            u_transport_option (str, optional): specifies the transport term
                used for the velocity equation. Supported options are:
                'vector_invariant_form', 'vector_advection_form',
                'vector_manifold_advection_form' and 'circulation_form'.
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
        field_names = ["u", "D"]

        if active_tracers is None:
            active_tracers = []

        if self.thermal:
            field_names.append("b")
            # add to the terms_to_linearise dictionary
            terms_to_linearise["b"] = []

        super().__init__(field_names, state, family, degree,
                         terms_to_linearise=terms_to_linearise,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        g = state.parameters.g
        H = state.parameters.H

        w, phi = self.tests[0:2]
        u, D = split(self.X)[0:2]
        u_trial = split(self.trials)[0]

        if self.thermal:
            gamma = self.tests[2]
            b = split(self.X)[2]
            n = FacetNormal(state.mesh)

        # -------------------------------------------------------------------- #
        # Time Derivative Terms
        # -------------------------------------------------------------------- #
        mass_form = self.generate_mass_terms()

        # -------------------------------------------------------------------- #
        # Transport Terms
        # -------------------------------------------------------------------- #
        # Velocity transport term -- depends on formulation
        if u_transport_option == "vector_invariant_form":
            u_adv = prognostic(vector_invariant_form(state, w, u), "u")
        elif u_transport_option == "vector_advection_form":
            u_adv = prognostic(advection_form(state, w, u), "u")
        elif u_transport_option == "vector_manifold_advection_form":
            u_adv = prognostic(vector_manifold_advection_form(state, w, u), "u")
        elif u_transport_option == "circulation_form":
            ke_form = prognostic(kinetic_energy_form(state, w, u), "u")
            ke_form = transport.remove(ke_form)
            ke_form = ke_form.label_map(
                lambda t: t.has_label(transporting_velocity),
                lambda t: Term(ufl.replace(
                    t.form, {t.get(transporting_velocity): u}), t.labels))
            ke_form = transporting_velocity.remove(ke_form)
            u_adv = prognostic(advection_equation_circulation_form(state, w, u), "u") + ke_form
        else:
            raise ValueError("Invalid u_transport_option: %s" % u_transport_option)

        # Depth transport term
        D_adv = prognostic(continuity_form(state, phi, D), "D")
        # Transport term needs special linearisation
        if transport in terms_to_linearise['D']:
            linear_D_adv = linear_continuity_form(state, phi, H).label_map(
                lambda t: t.has_label(transporting_velocity),
                lambda t: Term(ufl.replace(
                    t.form, {t.get(transporting_velocity): u_trial}), t.labels))
            # Add linearisation to D_adv
            D_adv = linearisation(D_adv, linear_D_adv)

        adv_form = subject(u_adv + D_adv, self.X)

        # Add transport of tracers
        if len(active_tracers) > 0:
            adv_form += self.generate_tracer_transport_terms(state, active_tracers)
        # Add transport of buoyancy, if thermal shallow water equations
        if self.thermal:
            b_adv = prognostic(continuity_form(state, gamma, b), "b")
            adv_form += subject(b_adv, self.X)

        # -------------------------------------------------------------------- #
        # Pressure Gradient Term
        # -------------------------------------------------------------------- #
        pressure_gradient_form = pressure_gradient(
            subject(prognostic(-g*div(w)*D*dx, "u"), self.X))

        residual = (mass_form + adv_form + pressure_gradient_form)

        # -------------------------------------------------------------------- #
        # Extra Terms (Coriolis, Topography and Thermal)
        # -------------------------------------------------------------------- #
        if fexpr is not None:
            V = FunctionSpace(state.mesh, "CG", 1)
            f = state.fields("coriolis", space=V)
            f.interpolate(fexpr)
            coriolis_form = coriolis(
                subject(prognostic(f*inner(state.perp(u), w)*dx, "u"), self.X))
            # Add linearisation
            linear_coriolis = coriolis(
                subject(prognostic(f*inner(state.perp(u_trial), w)*dx, "u"), self.X))
            coriolis_form = linearisation(coriolis_form, linear_coriolis)
            residual += coriolis_form

        if bexpr is not None:
            topography = state.fields("topography", state.spaces("DG"))
            topography.interpolate(bexpr)
            if self.thermal:
                topography_form = subject(prognostic
                                          (-topography*div(b*w)*dx
                                           + jump(b*w, n)*avg(topography)*dS,
                                           "u"), self.X)
            else:
                topography_form = subject(prognostic
                                          (-g*div(w)*topography*dx, "u"),
                                          self.X)
            residual += topography_form

        # thermal source terms not involving topography
        if self.thermal:
            source_form = subject(prognostic(-D*div(b*w)*dx
                                             - 0.5*b*div(D*w)*dx
                                             - 0.5*jump
                                             (b*w, n)*avg(D)*dS
                                             + 0.5*jump(D*w, n)*avg(b)*dS,
                                             "u"), self.X)
            residual += source_form

        # -------------------------------------------------------------------- #
        # Linearise equations
        # -------------------------------------------------------------------- #
        u_ref, D_ref = self.X_ref.split()[0:2]
        # Linearise about D = H
        # TODO: add interface to update linearisation state
        D_ref.assign(Constant(H))
        u_ref.assign(Constant(0.0))

        # Add linearisations to equations
        self.residual = self.generate_linear_terms(residual, self.terms_to_linearise)

    def _build_spaces(self, state, family, degree):
        spaces = [space for space in state.spaces.build_compatible_spaces
                  (family, degree)]
        if self.thermal:
            spaces.append(state.spaces("DG"))
        return spaces


class LinearShallowWaterEquations(ShallowWaterEquations):
    u"""
    Class for the linear (rotating) shallow-water equations, which describe the
    velocity 'u' and the depth field 'D', solving some variant of:
        ∂u/∂t + f×u + g*∇(D+b) = 0
        ∂D/∂t + H*∇.(u) = 0
    for mean depth 'H', Coriolis parameter 'f' and bottom surface 'b'.

    This is set up the from the underlying :class:`ShallowWaterEquations`,
    which is then linearised.
    """

    def __init__(self, state, family, degree, fexpr=None, bexpr=None,
                 terms_to_linearise={'D': [time_derivative, transport],
                                     'u': [time_derivative, pressure_gradient, coriolis]},
                 u_transport_option="vector_invariant_form",
                 no_normal_flow_bc_ids=None, active_tracers=None):
        """
        Args:
            state (:class:`State`): the model's state object.
            family (str): the finite element space family used for the velocity
                field. This determines the other finite element spaces used via
                the de Rham complex.
            degree (int): the element degree used for the velocity space.
            fexpr (:class:`ufl.Expr`, optional): an expression for the Coroilis
                parameter. Defaults to None.
            bexpr (:class:`ufl.Expr`, optional): an expression for the bottom
                surface of the fluid. Defaults to None.
            terms_to_linearise (dict, optional): a dictionary specifying which
                terms in the equation set to linearise. By default, includes
                both time derivatives, the 'D' transport term and the pressure
                gradient term.
            u_transport_option (str, optional): specifies the transport term
                used for the velocity equation. Supported options are:
                'vector_invariant_form', 'vector_advection_form',
                'vector_manifold_advection_form' and 'circulation_form'.
                Defaults to 'vector_invariant_form'.
            no_normal_flow_bc_ids (list, optional): a list of IDs of domain
                boundaries at which no normal flow will be enforced. Defaults to
                None.
            active_tracers (list, optional): a list of `ActiveTracer` objects
                that encode the metadata for any active tracers to be included
                in the equations. Defaults to None.
        """

        super().__init__(state, family, degree, fexpr=fexpr, bexpr=bexpr,
                         terms_to_linearise=terms_to_linearise,
                         u_transport_option=u_transport_option,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        # Use the underlying routine to do a first linearisation of the equations
        self.linearise_equation_set()

        # D transport term is a special case -- add facet term
        _, D = split(self.X)
        _, phi = self.tests
        D_adv = prognostic(linear_continuity_form(state, phi, D, facet_term=True), "D")
        self.residual = self.residual.label_map(
            lambda t: t.has_label(transport) and t.get(prognostic) == "D",
            map_if_true=lambda t: Term(D_adv.form, t.labels)
        )


class CompressibleEulerEquations(PrognosticEquationSet):
    """
    Class for the compressible Euler equations, which evolve the velocity 'u',
    the dry density 'rho' and the (virtual dry) potential temperature 'theta',
    solving:
        ∂u/∂t + (u.∇)u + 2Ω×u + c_p*θ*∇Π + g = 0
        ∂ρ/∂t + ∇.(ρ*u) = 0
        ∂θ/∂t + (u.∇)θ = 0,
    where Π is the Exner pressure, g is the gravitational vector, Ω is the
    planet's rotation vector and c_p is the heat capacity of dry air at constant
    pressure.
    """

    def __init__(self, state, family, degree, Omega=None, sponge=None,
                 extra_terms=None,
                 terms_to_linearise={'u': [time_derivative],
                                     'rho': [time_derivative, transport],
                                     'theta': [time_derivative, transport]},
                 u_transport_option="vector_invariant_form",
                 diffusion_options=None,
                 no_normal_flow_bc_ids=None,
                 active_tracers=None):
        """
        Args:
            state (:class:`State`): the model's state object.
            family (str): the finite element space family used for the velocity
                field. This determines the other finite element spaces used via
                the de Rham complex.
            degree (int): the element degree used for the velocity space.
            Omega (:class:`ufl.Expr`, optional): an expression for the planet's
                rotation vector. Defaults to None.
            sponge (:class:`ufl.Expr`, optional): an expression for a sponge
                layer. Defaults to None.
            extra_terms (:class:`ufl.Expr`, optional): any extra terms to be
                included in the equation set. Defaults to None.
            terms_to_linearise (dict, optional): a dictionary specifying which
                terms in the equation set to linearise. By default, includes
                the time derivatives and the scalar transport terms.
            u_transport_option (str, optional): specifies the transport term
                used for the velocity equation. Supported options are:
                'vector_invariant_form', 'vector_advection_form',
                'vector_manifold_advection_form' and 'circulation_form'.
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

        field_names = ['u', 'rho', 'theta']

        if active_tracers is None:
            active_tracers = []

        super().__init__(field_names, state, family, degree,
                         terms_to_linearise=terms_to_linearise,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        g = state.parameters.g
        cp = state.parameters.cp

        w, phi, gamma = self.tests[0:3]
        u, rho, theta = split(self.X)[0:3]
        u_trial = split(self.trials)[0]
        rhobar = state.fields("rhobar", space=state.spaces("DG"), dump=False)
        thetabar = state.fields("thetabar", space=state.spaces("theta"), dump=False)
        zero_expr = Constant(0.0)*theta
        exner = exner_pressure(state.parameters, rho, theta)
        n = FacetNormal(state.mesh)

        # -------------------------------------------------------------------- #
        # Time Derivative Terms
        # -------------------------------------------------------------------- #
        mass_form = self.generate_mass_terms()

        # -------------------------------------------------------------------- #
        # Transport Terms
        # -------------------------------------------------------------------- #
        # Velocity transport term -- depends on formulation
        if u_transport_option == "vector_invariant_form":
            u_adv = prognostic(vector_invariant_form(state, w, u), "u")
        elif u_transport_option == "vector_advection_form":
            u_adv = prognostic(advection_form(state, w, u), "u")
        elif u_transport_option == "vector_manifold_advection_form":
            u_adv = prognostic(vector_manifold_advection_form(state, w, u), "u")
        elif u_transport_option == "circulation_form":
            ke_form = prognostic(kinetic_energy_form(state, w, u), "u")
            ke_form = transport.remove(ke_form)
            ke_form = ke_form.label_map(
                lambda t: t.has_label(transporting_velocity),
                lambda t: Term(ufl.replace(
                    t.form, {t.get(transporting_velocity): u}), t.labels))
            ke_form = transporting_velocity.remove(ke_form)
            u_adv = prognostic(advection_equation_circulation_form(state, w, u), "u") + ke_form
        else:
            raise ValueError("Invalid u_transport_option: %s" % u_transport_option)

        # Density transport (conservative form)
        rho_adv = prognostic(continuity_form(state, phi, rho), "rho")
        # Transport term needs special linearisation
        if transport in terms_to_linearise['rho']:
            linear_rho_adv = linear_continuity_form(state, phi, rhobar).label_map(
                lambda t: t.has_label(transporting_velocity),
                lambda t: Term(ufl.replace(
                    t.form, {t.get(transporting_velocity): u_trial}), t.labels))
            rho_adv = linearisation(rho_adv, linear_rho_adv)

        # Potential temperature transport (advective form)
        theta_adv = prognostic(advection_form(state, gamma, theta), "theta")
        # Transport term needs special linearisation
        if transport in terms_to_linearise['theta']:
            linear_theta_adv = linear_advection_form(state, gamma, thetabar).label_map(
                lambda t: t.has_label(transporting_velocity),
                lambda t: Term(ufl.replace(
                    t.form, {t.get(transporting_velocity): u_trial}), t.labels))
            theta_adv = linearisation(theta_adv, linear_theta_adv)

        adv_form = subject(u_adv + rho_adv + theta_adv, self.X)

        # Add transport of tracers
        if len(active_tracers) > 0:
            adv_form += self.generate_tracer_transport_terms(state, active_tracers)

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

        pressure_gradient_form = name(subject(prognostic(
            cp*(-div(theta_v*w)*exner*dx
                + jump(theta_v*w, n)*avg(exner)*dS_v), "u"), self.X), "pressure_gradient")

        # -------------------------------------------------------------------- #
        # Gravitational Term
        # -------------------------------------------------------------------- #
        gravity_form = subject(prognostic(Term(g*inner(state.k, w)*dx), "u"), self.X)

        residual = (mass_form + adv_form + pressure_gradient_form + gravity_form)

        # -------------------------------------------------------------------- #
        # Moist Thermodynamic Divergence Term
        # -------------------------------------------------------------------- #
        if len(active_tracers) > 0:
            cv = state.parameters.cv
            c_vv = state.parameters.c_vv
            c_pv = state.parameters.c_pv
            c_pl = state.parameters.c_pl
            R_d = state.parameters.R_d
            R_v = state.parameters.R_v

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
                * (R_m / c_vml - (R_d * c_pml) / (cp * c_vml))*dx, "theta"), self.X)

        # -------------------------------------------------------------------- #
        # Extra Terms (Coriolis, Sponge, Diffusion and others)
        # -------------------------------------------------------------------- #
        if Omega is not None:
            # TODO: add linearisation and label for this
            residual += subject(prognostic(
                inner(w, cross(2*Omega, u))*dx, "u"), self.X)

        if sponge is not None:
            W_DG = FunctionSpace(state.mesh, "DG", 2)
            x = SpatialCoordinate(state.mesh)
            z = x[len(x)-1]
            H = sponge.H
            zc = sponge.z_level
            assert float(zc) < float(H), "you have set the sponge level above the height of your domain"
            mubar = sponge.mubar
            muexpr = conditional(z <= zc,
                                 0.0,
                                 mubar*sin((pi/2.)*(z-zc)/(H-zc))**2)
            self.mu = Function(W_DG).interpolate(muexpr)

            residual += name(subject(prognostic(
                self.mu*inner(w, state.k)*inner(u, state.k)*dx, "u"), self.X), "sponge")

        if diffusion_options is not None:
            for field, diffusion in diffusion_options:
                idx = self.field_names.index(field)
                test = self.tests[idx]
                fn = split(self.X)[idx]
                residual += subject(
                    prognostic(interior_penalty_diffusion_form(
                        state, test, fn, diffusion), field), self.X)

        if extra_terms is not None:
            for field, term in extra_terms:
                idx = self.field_names.index(field)
                test = self.tests[idx]
                residual += subject(prognostic(
                    inner(test, term)*dx, field), self.X)

        # -------------------------------------------------------------------- #
        # Linearise equations
        # -------------------------------------------------------------------- #
        # TODO: add linearisation states for variables
        # Add linearisations to equations
        self.residual = self.generate_linear_terms(residual, self.terms_to_linearise)


class HydrostaticCompressibleEulerEquations(CompressibleEulerEquations):
    """
    The hydrostatic form of the compressible Euler equations. In this case the
    vertical velocity derivative is zero in the equations, so only 'u_h', the
    horizontal component of the velocity is allowed to vary in time. The
    equations, for velocity 'u', dry density 'rho' and (dry) potential
    temperature 'theta' are:
        ∂u_h/∂t + (u.∇)u_h + 2Ω×u + c_p*θ*∇Π + g = 0
        ∂ρ/∂t + ∇.(ρ*u) = 0
        ∂θ/∂t + (u.∇)θ = 0,
    where Π is the Exner pressure, g is the gravitational vector, Ω is the
    planet's rotation vector and c_p is the heat capacity of dry air at constant
    pressure.

    This is implemented through a hydrostatic switch to the compressible Euler
    equations.
    """

    def __init__(self, state, family, degree, Omega=None, sponge=None,
                 extra_terms=None,
                 terms_to_linearise={'u': [time_derivative],
                                     'rho': [time_derivative, transport],
                                     'theta': [time_derivative, transport]},
                 u_transport_option="vector_invariant_form",
                 diffusion_options=None,
                 no_normal_flow_bc_ids=None,
                 active_tracers=None):
        """
        Args:
            state (:class:`State`): the model's state object.
            family (str): the finite element space family used for the velocity
                field. This determines the other finite element spaces used via
                the de Rham complex.
            degree (int): the element degree used for the velocity space.
            Omega (:class:`ufl.Expr`, optional): an expression for the planet's
                rotation vector. Defaults to None.
            sponge (:class:`ufl.Expr`, optional): an expression for a sponge
                layer. Defaults to None.
            extra_terms (:class:`ufl.Expr`, optional): any extra terms to be
                included in the equation set. Defaults to None.
            terms_to_linearise (dict, optional): a dictionary specifying which
                terms in the equation set to linearise. By default, includes
                the time derivatives and the scalar transport terms.
            u_transport_option (str, optional): specifies the transport term
                used for the velocity equation. Supported options are:
                'vector_invariant_form', 'vector_advection_form',
                'vector_manifold_advection_form' and 'circulation_form'.
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

        super().__init__(state, family, degree, Omega=Omega, sponge=sponge,
                         extra_terms=extra_terms,
                         terms_to_linearise=terms_to_linearise,
                         u_transport_option=u_transport_option,
                         diffusion_options=diffusion_options,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        self.residual = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=lambda t: hydrostatic(t, self.hydrostatic_projection(t))
        )

        k = self.state.k
        u = split(self.X)[0]
        self.residual += name(
            subject(
                prognostic(
                    -inner(k, self.tests[0]) * inner(k, u) * dx, "u"),
                self.X),
            "hydrostatic_form")

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
        assert not self.state.on_sphere, "the hydrostatic projection is not yet implemented for spherical geometry"
        k = Constant((*self.state.k, 0, 0))
        X = t.get(subject)

        new_subj = X - k * inner(X, k)
        return replace_subject(new_subj)(t)


class IncompressibleBoussinesqEquations(PrognosticEquationSet):
    # TODO: check that these are correct
    """
    Class for the incompressible Boussinesq equations, which evolve the velocity
    'u', the pressure 'p' and the buoyancy 'b'.

    The pressure features as a Lagrange multiplier to enforce the
    incompressibility of the equations. The equations are then
        ∂u/∂t + (u.∇)u + 2Ω×u + ∇p + b*k = 0
        ∇.u = p
        ∂b/∂t + (u.∇)b = 0,
    where k is the vertical unit vector and, Ω is the planet's rotation vector.
    """

    def __init__(self, state, family, degree, Omega=None,
                 terms_to_linearise={'u': [time_derivative],
                                     'p': [time_derivative],
                                     'b': [time_derivative, transport]},
                 u_transport_option="vector_invariant_form",
                 no_normal_flow_bc_ids=None,
                 active_tracers=None):
        """
        Args:
            state (:class:`State`): the model's state object.
            family (str): the finite element space family used for the velocity
                field. This determines the other finite element spaces used via
                the de Rham complex.
            degree (int): the element degree used for the velocity space.
            Omega (:class:`ufl.Expr`, optional): an expression for the planet's
                rotation vector. Defaults to None.
            terms_to_linearise (dict, optional): a dictionary specifying which
                terms in the equation set to linearise. By default, includes
                the time derivatives and the buoyancy transport term.
            u_transport_option (str, optional): specifies the transport term
                used for the velocity equation. Supported options are:
                'vector_invariant_form', 'vector_advection_form',
                'vector_manifold_advection_form' and 'circulation_form'.
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

        if active_tracers is not None:
            raise NotImplementedError('Tracers not implemented for Boussinesq equations')

        if active_tracers is None:
            active_tracers = []

        super().__init__(field_names, state, family, degree,
                         terms_to_linearise=terms_to_linearise,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        w, phi, gamma = self.tests[0:3]
        u, p, b = split(self.X)
        u_trial = split(self.trials)[0]
        bbar = state.fields("bbar", space=state.spaces("theta"), dump=False)
        bbar = state.fields("pbar", space=state.spaces("DG"), dump=False)

        # -------------------------------------------------------------------- #
        # Time Derivative Terms
        # -------------------------------------------------------------------- #
        mass_form = self.generate_mass_terms()

        # -------------------------------------------------------------------- #
        # Transport Terms
        # -------------------------------------------------------------------- #
        # Velocity transport term -- depends on formulation
        if u_transport_option == "vector_invariant_form":
            u_adv = prognostic(vector_invariant_form(state, w, u), "u")
        elif u_transport_option == "vector_advection_form":
            u_adv = prognostic(advection_form(state, w, u), "u")
        elif u_transport_option == "vector_manifold_advection_form":
            u_adv = prognostic(vector_manifold_advection_form(state, w, u), "u")
        elif u_transport_option == "circulation_form":
            ke_form = prognostic(kinetic_energy_form(state, w, u), "u")
            ke_form = transport.remove(ke_form)
            ke_form = ke_form.label_map(
                lambda t: t.has_label(transporting_velocity),
                lambda t: Term(ufl.replace(
                    t.form, {t.get(transporting_velocity): u}), t.labels))
            ke_form = transporting_velocity.remove(ke_form)
            u_adv = prognostic(advection_equation_circulation_form(state, w, u), "u") + ke_form
        else:
            raise ValueError("Invalid u_transport_option: %s" % u_transport_option)

        # Buoyancy transport
        b_adv = prognostic(advection_form(state, gamma, b), "b")
        linear_b_adv = linear_advection_form(state, gamma, bbar).label_map(
            lambda t: t.has_label(transporting_velocity),
            lambda t: Term(ufl.replace(
                t.form, {t.get(transporting_velocity): u_trial}), t.labels))
        b_adv = linearisation(b_adv, linear_b_adv)

        adv_form = subject(u_adv + b_adv, self.X)

        # Add transport of tracers
        if len(active_tracers) > 0:
            adv_form += self.generate_tracer_transport_terms(state, active_tracers)

        # -------------------------------------------------------------------- #
        # Pressure Gradient Term
        # -------------------------------------------------------------------- #
        pressure_gradient_form = subject(prognostic(-div(w)*p*dx, "u"), self.X)

        # -------------------------------------------------------------------- #
        # Gravitational Term
        # -------------------------------------------------------------------- #
        gravity_form = subject(prognostic(-b*inner(w, state.k)*dx, "u"), self.X)

        # -------------------------------------------------------------------- #
        # Divergence Term
        # -------------------------------------------------------------------- #
        # This enforces that div(u) = 0
        # The p features here so that the div(u) evaluated in the "forcing" step
        # replaces the whole pressure field, rather than merely providing an
        # increment to it.
        divergence_form = name(subject(prognostic(phi*(p-div(u))*dx, "p"), self.X),
                               "incompressibility")

        residual = (mass_form + adv_form + divergence_form
                    + pressure_gradient_form + gravity_form)

        # -------------------------------------------------------------------- #
        # Extra Terms (Coriolis)
        # -------------------------------------------------------------------- #
        if Omega is not None:
            # TODO: add linearisation and label for this
            residual += subject(prognostic(
                inner(w, cross(2*Omega, u))*dx, "u"), self.X)

        # -------------------------------------------------------------------- #
        # Linearise equations
        # -------------------------------------------------------------------- #
        # TODO: add linearisation states for variables
        # Add linearisations to equations
        self.residual = self.generate_linear_terms(residual, self.terms_to_linearise)
