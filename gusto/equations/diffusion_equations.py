"""Defines the diffusion equation in weak form."""

from firedrake import inner, dx, TestFunctions
from firedrake.fml import subject
from gusto.core.labels import time_derivative, prognostic
from gusto.equations.common_forms import diffusion_form
from gusto.equations.prognostic_equations import PrognosticEquation, PrognosticEquationSet

# __all__ = ["DiffusionEquation", "MixedDiffusionEquation"]
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


# class MixedDiffusionEquation(PrognosticEquation):

#     def __init__(self, domain, function_space, diffusion_parameters):
#         """
#         Args:
#             domain (:class:`Domain`): the model's domain object, containing the
#                 mesh and the compatible function spaces.
#             function_space (:class:`FunctionSpace`): the function space that the
#                 equation's prognostic is defined on.
#             diffusion_parameters tuple of (field_name, :class:`DiffusionParameters`)
#                 describing the diffusion to be applied.
#         """
#         field_names = ["q1", "q2"]
#         self.field_names = field_names
#         space_names = {"q1": "HDiv", "q2": "L2"}
#         # super().__init__(field_names, domain, space_names)
#         field_name = "_".join(field_names)
#         super().__init__(domain, function_space, field_name)

#         self.tests = TestFunctions(function_space)
#         test1, test2 = self.tests
#         q = self.X
#         q1, q2 = self.X.subfunctions
        
#         mass_form = prognostic(inner(test1, q1)*dx, "q1") + prognostic(inner(test2, q2)*dx, "q2")

#         diffusive_form = prognostic(diffusion_form(test1, q1, diffusion_parameters[0].kappa), "q1") + prognostic(diffusion_form(test2, q2, diffusion_parameters[1].kappa), "q2")
#         self.residual = subject(mass_form + diffusive_form, q)
