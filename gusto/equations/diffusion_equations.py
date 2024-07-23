"""Defines the diffusion equation in weak form."""

from firedrake import inner, dx
from firedrake.fml import subject
from gusto.core.labels import time_derivative, prognostic
from gusto.equations.common_forms import diffusion_form
from gusto.equations.prognostic_equations import PrognosticEquation

__all__ = ["DiffusionEquation"]


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
