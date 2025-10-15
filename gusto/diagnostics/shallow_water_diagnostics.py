"""Common diagnostic fields for the Shallow Water equations."""


from firedrake import (
    dx, TestFunction, TrialFunction, grad, inner, curl, Function, assemble,
    LinearVariationalProblem, LinearVariationalSolver, conditional
)
from firedrake.__future__ import interpolate
from gusto.diagnostics.diagnostics import DiagnosticField, Energy

__all__ = ["ShallowWaterKineticEnergy", "ShallowWaterPotentialEnergy",
           "ShallowWaterPotentialEnstrophy", "PotentialVorticity",
           "RelativeVorticity", "AbsoluteVorticity", "PartitionedVapour",
           "PartitionedCloud", "ShallowWaterAvailablePotentialEnergy",
           "MoistConvectiveSWRelativeHumidity"]


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
                constant_jacobian = False
            else:
                a = q*gamma*dx
                constant_jacobian = True

            L = (- inner(domain.perp(grad(gamma)), u))*dx
            if vorticity_type != "relative":
                f = state_fields("coriolis")
                L += gamma*f*dx

            problem = LinearVariationalProblem(a, L, self.field,
                                               constant_jacobian=constant_jacobian)
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


class PartitionedVapour(DiagnosticField):
    """
    Diagnostic for computing the vapour in the equivalent buoyancy formulation
    of the moist thermal shallow water equations.
    """
    name = "PartitionedVapour"

    def __init__(self, equation, name='q_t', space=None,
                 method='interpolate'):
        """
        Args:
            equation (:class:`PrognosticEquation`): the model's equation.
            name (str, optional): name of the total moisture field to use to
                compute the vapour from. Defaults to total moisture, q_t.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case the default space is the domain's DG space.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        self.fname = name
        self.equation = equation
        super().__init__(space=space, method=method, required_fields=(self.fname,))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        q_t = state_fields(self.fname)
        space = domain.spaces("DG")
        self.qsat_func = Function(space)

        qsat_expr = self.equation.compute_saturation(state_fields.X(
            self.equation.field_name))
        self.qsat_interpolate = interpolate(qsat_expr, space)
        self.expr = conditional(q_t < self.qsat_func, q_t, self.qsat_func)

        super().setup(domain, state_fields, space=space)

    def compute(self):
        """Performs the computation of the diagnostic field."""
        self.qsat_func.assign(assemble(self.qsat_interpolate))
        super().compute()


class PartitionedCloud(DiagnosticField):
    """
    Diagnostic for computing the cloud in the equivalent buoyancy formulation
    of the moist thermal shallow water equations.
    """
    name = "PartitionedCloud"

    def __init__(self, equation, name='q_t', space=None,
                 method='interpolate'):
        """
        Args:
            equation (:class:`PrognosticEquation`): the model's equation.
            name (str, optional): name of the total moisture field to use to
                compute the vapour from. Defaults to total moisture, q_t.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case the default space is the domain's DG space.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        self.fname = name
        self.equation = equation
        super().__init__(space=space, method=method, required_fields=(self.fname,))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        q_t = state_fields(self.fname)
        space = domain.spaces("DG")
        self.qsat_func = Function(space)

        qsat_expr = self.equation.compute_saturation(state_fields.X(
            self.equation.field_name))
        self.qsat_interpolate = interpolate(qsat_expr, space)
        vapour = conditional(q_t < self.qsat_func, q_t, self.qsat_func)
        self.expr = q_t - vapour

        super().setup(domain, state_fields, space=space)

    def compute(self):
        """Performs the computation of the diagnostic field."""
        self.qsat_func.assign(assemble(self.qsat_interpolate))
        super().compute()


class ShallowWaterAvailablePotentialEnergy(Energy):
    """Diagnostic shallow-water available potential energy density."""
    name = "ShallowWaterAvailablePotentialEnergy"

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
        H = self.parameters.H
        D = state_fields("D")
        self.expr = 0.5*g*(D-H)**2
        super().setup(domain, state_fields)


class MoistConvectiveSWRelativeHumidity(DiagnosticField):
    """
    Diagnostic for computing relative humidity, given a saturation function
    which is a function of depth only.
    """
    name = "RelativeHumidity"

    def __init__(self, sat_func):
        """
        Args:
            sat_func (function?): saturation function being used in the model.
        """
        self.fname = "water_vapour"
        self.sat_func = sat_func
        super().__init__(method='assign', required_fields=(self.fname, "D"))

    def setup(self, domain, state_fields):
        """
        
        Args:
        """
        q_v = state_fields(self.fname)
        self.D = state_fields("D")
        space = domain.spaces("DG")
        self.sat_val = Function(space)
        # self.sat_val.interpolate(self.sat_func(self.D))
        self.expr = (q_v/self.sat_val)*100
        super().setup(domain, state_fields, space=space)
        
    def compute(self):
    
        self.sat_val.interpolate(self.sat_func(self.D))

