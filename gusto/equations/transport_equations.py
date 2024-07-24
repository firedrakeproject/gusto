"""Defines variants of the transport equation, in weak form."""

from firedrake import Function, inner, dx, MixedFunctionSpace, TestFunctions
from firedrake.fml import subject
from gusto.core.labels import time_derivative, prognostic
from gusto.equations.common_forms import advection_form, continuity_form
from gusto.equations.prognostic_equations import PrognosticEquation, PrognosticEquationSet

__all__ = ["AdvectionEquation", "ContinuityEquation", "CoupledTransportEquation"]


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


class CoupledTransportEquation(PrognosticEquationSet):
    u"""
    Discretises the transport equation,                                       \n
    ∂q/∂t + (u.∇)q = F,                                                       \n
    with the application of active tracers.
    As there are multiple tracers or species that are interacting, q and F are
    vectors. This equation can be enhanced through the addition of sources or
    sinks (F) by applying it with physics schemes.
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

        # Add mass forms for the tracers, which will use
        # mass*density for any tracer_conservative terms
        self.residual = self.generate_mass_terms()

        # Add transport of tracers
        self.residual += self.generate_tracer_transport_terms(active_tracers)
