"""Common diagnostic fields for the Shallow Water equations."""


from firedrake import (dx, TestFunction, TrialFunction, grad, inner, curl,
                       LinearVariationalProblem, LinearVariationalSolver,
                       conditional)
from gusto.diagnostics.diagnostics import DiagnosticField, Energy

__all__ = ["ShallowWaterKineticEnergy", "ShallowWaterPotentialEnergy",
           "ShallowWaterPotentialEnstrophy", "PotentialVorticity",
           "RelativeVorticity", "AbsoluteVorticity",
           "SWCO2cond_flag"]


class ShallowWaterKineticEnergy(Energy):
    """Diagnostic shallow-water kinetic energy density."""
    name = "ShallowWaterKineticEnergy"

    def __init__(self, space=None, method='interpolate'):
        """
        Args:
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        super().__init__(space=space, method=method, required_fields=("D", "u"))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        u = state_fields("u")
        D = state_fields("D")
        self.expr = self.kinetic(u, D)
        super().setup(domain, state_fields)


class ShallowWaterPotentialEnergy(Energy):
    """Diagnostic shallow-water potential energy density."""
    name = "ShallowWaterPotentialEnergy"

    def __init__(self, parameters, space=None, method='interpolate'):
        """
        Args:
            parameters (:class:`ShallowWaterParameters`): the configuration
                object containing the physical parameters for this equation.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        self.parameters = parameters
        super().__init__(space=space, method=method, required_fields=("D"))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        g = self.parameters.g
        D = state_fields("D")
        self.expr = 0.5*g*D**2
        super().setup(domain, state_fields)


class ShallowWaterPotentialEnstrophy(DiagnosticField):
    """Diagnostic (dry) compressible kinetic energy density."""
    def __init__(self, base_field_name="PotentialVorticity", space=None,
                 method='interpolate'):
        """
        Args:
            base_field_name (str, optional): the base potential vorticity field
                to compute the enstrophy from. Defaults to "PotentialVorticity".
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        base_enstrophy_names = ["PotentialVorticity", "RelativeVorticity", "AbsoluteVorticity"]
        if base_field_name not in base_enstrophy_names:
            raise ValueError(
                f"Don't know how to compute enstrophy with base_field_name={base_field_name};"
                + f"base_field_name should be one of {base_enstrophy_names}")
        # Work out required fields
        if base_field_name in ["PotentialVorticity", "AbsoluteVorticity"]:
            required_fields = (base_field_name, "D")
        elif base_field_name == "RelativeVorticity":
            required_fields = (base_field_name, "D", "coriolis")
        else:
            raise NotImplementedError(f'Enstrophy with vorticity {base_field_name} not implemented')

        super().__init__(space=space, method=method, required_fields=required_fields)
        self.base_field_name = base_field_name

    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        base_name = "SWPotentialEnstrophy"
        return "_from_".join((base_name, self.base_field_name))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        if self.base_field_name == "PotentialVorticity":
            pv = state_fields("PotentialVorticity")
            D = state_fields("D")
            self.expr = 0.5*pv**2*D
        elif self.base_field_name == "RelativeVorticity":
            zeta = state_fields("RelativeVorticity")
            D = state_fields("D")
            f = state_fields("coriolis")
            self.expr = 0.5*(zeta + f)**2/D
        elif self.base_field_name == "AbsoluteVorticity":
            zeta_abs = state_fields("AbsoluteVorticity")
            D = state_fields("D")
            self.expr = 0.5*(zeta_abs)**2/D
        else:
            raise NotImplementedError(f'Enstrophy with {self.base_field_name} not implemented')
        super().setup(domain, state_fields)


class Vorticity(DiagnosticField):
    """Base diagnostic field class for shallow-water vorticity variables."""

    def setup(self, domain, state_fields, vorticity_type=None):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
            vorticity_type (str, optional): denotes which type of vorticity to
                be computed ('relative', 'absolute' or 'potential'). Defaults to
                None.
        """

        vorticity_types = ["relative", "absolute", "potential"]
        if vorticity_type not in vorticity_types:
            raise ValueError(f"vorticity type must be one of {vorticity_types}, not {vorticity_type}")
        space = domain.spaces("H1")

        u = state_fields("u")
        if vorticity_type in ["absolute", "potential"]:
            f = state_fields("coriolis")
        if vorticity_type == "potential":
            D = state_fields("D")

        if self.method != 'solve':
            if vorticity_type == "potential":
                self.expr = (curl(u) + f) / D
            elif vorticity_type == "absolute":
                self.expr = curl(u) + f
            elif vorticity_type == "relative":
                self.expr = curl(u)

        super().setup(domain, state_fields, space=space)

        # Set up problem now that self.field has been set up
        if self.method == 'solve':
            gamma = TestFunction(space)
            q = TrialFunction(space)

            if vorticity_type == "potential":
                a = q*gamma*D*dx
            else:
                a = q*gamma*dx

            L = (- inner(domain.perp(grad(gamma)), u))*dx
            if vorticity_type != "relative":
                f = state_fields("coriolis")
                L += gamma*f*dx

            problem = LinearVariationalProblem(a, L, self.field)
            self.evaluator = LinearVariationalSolver(problem, solver_parameters={"ksp_type": "cg"})


class PotentialVorticity(Vorticity):
    u"""Diagnostic field for shallow-water potential vorticity, q=(∇×(u+f))/D"""
    name = "PotentialVorticity"

    def __init__(self, space=None, method='solve'):
        """
        Args:
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'solve'.
        """
        self.solve_implemented = True
        super().__init__(space=space, method=method,
                         required_fields=('u', 'D', 'coriolis'))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        super().setup(domain, state_fields, vorticity_type="potential")


class AbsoluteVorticity(Vorticity):
    u"""Diagnostic field for absolute vorticity, ζ=∇×(u+f)"""
    name = "AbsoluteVorticity"

    def __init__(self, space=None, method='solve'):
        """
        Args:
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'solve'.
        """
        self.solve_implemented = True
        super().__init__(space=space, method=method, required_fields=('u', 'coriolis'))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        super().setup(domain, state_fields, vorticity_type="absolute")


class RelativeVorticity(Vorticity):
    u"""Diagnostic field for relative vorticity, ζ=∇×u"""
    name = "RelativeVorticity"

    def __init__(self, space=None, method='solve'):
        """
        Args:
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'solve'.
        """
        self.solve_implemented = True
        super().__init__(space=space, method=method, required_fields=('u',))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        super().setup(domain, state_fields, vorticity_type="relative")


class SWCO2cond_flag(DiagnosticField):
    """Base diagnostic for calculating the difference between one field and a constant"""
    def __init__(self, field_name, constant):
        """
        Args:
            field_name (str): the name of the field to be subtracted from.
            constant (Functionspace?): the constant to be subtracted.
        """
        super().__init__(method='interpolate', required_fields=(field_name,))
        self.field_name = field_name
        self.constant = constant

    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        return "CO2cond_flag"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """

        field = state_fields(self.field_name)
        constant = self.constant
        topo = state_fields('topography')
        heaviside = field + topo - constant
        condit_expr = conditional(heaviside < 0, 1, 0)
        self.expr = condit_expr
        space = field.function_space()
        super().setup(domain, state_fields, space=space)