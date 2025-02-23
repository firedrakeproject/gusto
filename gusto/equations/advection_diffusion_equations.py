"""Defines the advection-diffusion equation in weak form."""

from firedrake import inner, dx
from firedrake.fml import subject
from gusto.core.labels import time_derivative, prognostic
from gusto.equations.common_forms import advection_form, diffusion_form
from gusto.equations.prognostic_equations import PrognosticEquation

__all__ = ["AdvectionDiffusionEquation"]


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
