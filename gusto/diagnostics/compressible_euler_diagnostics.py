"""Common diagnostic fields for the compressible Euler equations."""

from firedrake import (dot, dx, Function, ln, TestFunction, TrialFunction,
                       Constant, grad, inner, LinearVariationalProblem,
                       LinearVariationalSolver, FacetNormal, ds_b, dS_v, div,
                       avg, jump, SpatialCoordinate)

from gusto.diagnostics.diagnostics import (
    DiagnosticField, Energy, IterativeDiagnosticField
)
from gusto.equations import CompressibleEulerEquations
import gusto.equations.thermodynamics as tde
from gusto.recovery import Recoverer, BoundaryMethod
from gusto.equations import TracerVariableType, Phases

__all__ = ["RichardsonNumber", "Entropy", "PhysicalEntropy", "DynamicEntropy",
           "CompressibleKineticEnergy", "Exner", "Theta_e", "InternalEnergy",
           "PotentialEnergy", "ThermodynamicKineticEnergy",
           "Temperature", "Theta_d", "RelativeHumidity", "Pressure", "Exner_Vt",
           "HydrostaticImbalance", "Precipitation", "BruntVaisalaFrequencySquared", 
           "ScorerParameterSquared", "WetBulbTemperature", "DewpointTemperature"]


class RichardsonNumber(DiagnosticField):
    """Dimensionless Richardson number diagnostic field."""
    name = "RichardsonNumber"

    def __init__(self, density_field, factor=1., space=None, method='interpolate'):
        u"""
        Args:
            density_field (str): the name of the density field.
            factor (float, optional): a factor to multiply the Brunt-Väisälä
                frequency by. Defaults to 1.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        super().__init__(space=space, method=method, required_fields=(density_field, "u_gradient"))
        self.density_field = density_field
        self.factor = Constant(factor)

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        rho_grad = self.density_field+"_gradient"
        grad_density = state_fields(rho_grad)
        gradu = state_fields("u_gradient")

        denom = 0.
        z_dim = domain.mesh.geometric_dimension() - 1
        u_dim = state_fields("u").ufl_shape[0]
        for i in range(u_dim-1):
            denom += gradu[i, z_dim]**2
        Nsq = self.factor*grad_density[z_dim]
        self.expr = Nsq/denom
        super().setup(domain, state_fields)


class Entropy(DiagnosticField):
    """Base diagnostic field for entropy diagnostic """

    def __init__(self, equations, space=None, method="interpolate"):
        """
        Args:
            equations (:class:`PrognosticEquationSet`): the equation set being
                solved by the model.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        self.equations = equations
        if isinstance(equations, CompressibleEulerEquations):
            required_fields = ['rho', 'theta']
        else:
            raise NotImplementedError(f'entropy not yet implemented for {type(equations)}')

        super().__init__(space=space, method=method, required_fields=tuple(required_fields))


class PhysicalEntropy(Entropy):
    u"""Physical entropy ρ*ln(θ) for Compressible Euler equations"""
    name = "PhysicalEntropy"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        rho = state_fields('rho')
        theta = state_fields('theta')
        self.expr = rho * ln(theta)
        super().setup(domain, state_fields)


class DynamicEntropy(Entropy):
    u"""Dynamic entropy 0.5*ρ*θ^2 for Compressible Euler equations"""
    name = "DynamicEntropy"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        rho = state_fields('rho')
        theta = state_fields('theta')
        self.expr = 0.5 * rho * theta**2
        super().setup(domain, state_fields)


class CompressibleKineticEnergy(Energy):
    """Diagnostic (dry) compressible kinetic energy density."""
    name = "CompressibleKineticEnergy"

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
        super().__init__(space=space, method=method, required_fields=("rho", "u"))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field
        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        u = state_fields("u")
        rho = state_fields("rho")
        self.expr = self.kinetic(u, rho)
        super().setup(domain, state_fields)


class Exner(DiagnosticField):
    """The diagnostic Exner pressure field."""
    def __init__(self, parameters, reference=False, space=None, method='interpolate'):
        """
        Args:
            parameters (:class:`CompressibleParameters`): the configuration
                object containing the physical parameters for this equation.
            reference (bool, optional): whether to compute the reference Exner
                pressure field or not. Defaults to False.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        self.parameters = parameters
        self.reference = reference
        if reference:
            self.rho_name = "rho_bar"
            self.theta_name = "theta_bar"
        else:
            self.rho_name = "rho"
            self.theta_name = "theta"
        super().__init__(space=space, method=method, required_fields=(self.rho_name, self.theta_name))

    @property
    def name(self):
        """Gives the name of this diagnostic field."""
        if self.reference:
            return "Exner_bar"
        else:
            return "Exner"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        rho = state_fields(self.rho_name)
        theta = state_fields(self.theta_name)
        self.expr = tde.exner_pressure(self.parameters, rho, theta)
        super().setup(domain, state_fields)


class BruntVaisalaFrequencySquared(DiagnosticField):
    """The diagnostic for the Brunt-Väisälä frequency."""
    name = "Brunt-Vaisala_squared"

    def __init__(self, equations, space=None, method='interpolate'):
        """
        Args:
            equations (:class:`PrognosticEquationSet`): the equation set being
                solved by the model.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        self.parameters = equations.parameters
        # Work out required fields
        if isinstance(equations, CompressibleEulerEquations):
            required_fields = ['theta']
            if equations.active_tracers is not None and len(equations.active_tracers) > 1:
                # TODO: I think theta here should be theta_e, which would be
                # easiest if this is a ThermodynamicDiagnostic. But in the dry
                # case, our numerical theta_e does not reduce to the numerical
                # dry theta
                raise NotImplementedError(
                    'Brunt-Vaisala diagnostic not implemented for moist equations')
        else:
            raise NotImplementedError(
                f'Brunt-Vaisala diagnostic not implemented for {type(equations)}')
        super().__init__(space=space, method=method, required_fields=tuple(required_fields))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        theta = state_fields('theta')
        self.expr = self.parameters.g/theta * dot(domain.k, grad(theta))
        super().setup(domain, state_fields)


class ScorerParameterSquared(DiagnosticField):
    """The Scorer parameter diagnostic field."""
    name = "ScorerParameter"

    def __init__(self, equations, space=None, method='interpolate'):
        """
        Args:
            equations (:class:`Prognostic EquationSet`): the equation set being
                solved by the model.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.    
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        # Work out required fields
        if isinstance(equations, CompressibleEulerEquations):
            required_fields = ['u', 'Brunt-Vaisala_squared']
        else:
            raise NotImplementedError(
                f'Scorer parameter diagnostic not implemented for {type(equations)}')
        super().__init__(space=space, method=method, required_fields=tuple(required_fields))
    
    def setup(self, domain, state_fields, space=None):
        
        u = state_fields('u')
        k = domain.k
        u_z = dot(k, u)
        N2 = state_fields('Brunt-Vaisala_squared')
        du_dz = dot(k, grad(u))
        du2_dz2 = dot(k, grad(du_dz))
        #self.expr = (N2/u_z - du2_dz2) / u_z
        # leading term only for now, as it often dominates 
        self.expr = N2 / u_z**2
        super().setup(domain, state_fields)
    

# TODO: unify thermodynamic diagnostics
class ThermodynamicDiagnostic(DiagnosticField):
    """Base thermodynamic diagnostic field, computing many common fields."""

    def __init__(self, equations, space=None, method='interpolate'):
        """
        Args:
            equations (:class:`PrognosticEquationSet`): the equation set being
                solved by the model.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        self.equations = equations
        self.parameters = equations.parameters
        # Work out required fields
        if isinstance(equations, CompressibleEulerEquations):
            required_fields = ['rho', 'theta']
            if equations.active_tracers is not None:
                for active_tracer in equations.active_tracers:
                    if active_tracer.chemical == 'H2O':
                        required_fields.append(active_tracer.name)
        else:
            raise NotImplementedError(f'Thermodynamic diagnostics not implemented for {type(equations)}')
        super().__init__(space=space, method=method, required_fields=tuple(required_fields))

    def _setup_thermodynamics(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """

        self.Vtheta = domain.spaces('theta')
        h_deg = self.Vtheta.ufl_element().degree()[0]
        v_deg = self.Vtheta.ufl_element().degree()[1]-1
        boundary_method = BoundaryMethod.extruded if (v_deg == 0 and h_deg == 0) else None

        # Extract all fields
        self.rho = state_fields("rho")
        self.theta = state_fields("theta")
        # Rho must be averaged to Vtheta
        self.rho_averaged = Function(self.Vtheta)
        self.recoverer = Recoverer(self.rho, self.rho_averaged, boundary_method=boundary_method)

        zero_expr = Constant(0.0)*self.theta
        self.r_v = zero_expr  # Water vapour
        self.r_l = zero_expr  # Liquid water
        self.r_t = zero_expr  # All water mixing ratios
        for active_tracer in self.equations.active_tracers:
            if active_tracer.chemical == "H2O":
                if active_tracer.variable_type != TracerVariableType.mixing_ratio:
                    raise NotImplementedError('Only mixing ratio tracers are implemented')
                if active_tracer.phase == Phases.gas:
                    self.r_v += state_fields(active_tracer.name)
                elif active_tracer.phase == Phases.liquid:
                    self.r_l += state_fields(active_tracer.name)
                self.r_t += state_fields(active_tracer.name)

        # Store the most common expressions
        self.exner = tde.exner_pressure(self.parameters, self.rho_averaged, self.theta)
        self.T = tde.T(self.parameters, self.theta, self.exner, r_v=self.r_v)
        self.p = tde.p(self.parameters, self.exner)

    def compute(self):
        """Compute the thermodynamic diagnostic."""
        self.recoverer.project()
        super().compute()


class Theta_e(ThermodynamicDiagnostic):
    """The moist equivalent potential temperature diagnostic field."""
    name = "Theta_e"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        self._setup_thermodynamics(domain, state_fields)
        self.expr = tde.theta_e(self.parameters, self.T, self.p, self.r_v, self.r_t)
        super().setup(domain, state_fields, space=self.Vtheta)


class InternalEnergy(ThermodynamicDiagnostic):
    """The moist compressible internal energy density."""
    name = "InternalEnergy"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        self._setup_thermodynamics(domain, state_fields)
        self.expr = tde.internal_energy(self.parameters, self.rho_averaged, self.T, r_v=self.r_v, r_l=self.r_l)
        super().setup(domain, state_fields, space=self.Vtheta)


class PotentialEnergy(ThermodynamicDiagnostic):
    """The moist compressible potential energy density."""
    name = "PotentialEnergy"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        x = SpatialCoordinate(domain.mesh)
        self._setup_thermodynamics(domain, state_fields)
        z = Function(self.rho_averaged.function_space())
        z.interpolate(dot(x, domain.k))
        self.expr = self.rho_averaged * (1 + self.r_t) * self.parameters.g * z
        super().setup(domain, state_fields, space=domain.spaces("DG"))


# TODO: this needs consolidating with energy diagnostics
class ThermodynamicKineticEnergy(ThermodynamicDiagnostic):
    """The moist compressible kinetic energy density."""
    name = "ThermodynamicKineticEnergy"

    def __init__(self, equations, space=None, method='interpolate'):
        """
        Args:
            equations (:class:`PrognosticEquationSet`): the equation set being
                solved by the model.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        self.equations = equations
        self.parameters = equations.parameters
        # Work out required fields
        if isinstance(equations, CompressibleEulerEquations):
            required_fields = ['rho', 'u']
            if equations.active_tracers is not None:
                for active_tracer in equations.active_tracers:
                    if active_tracer.chemical == 'H2O':
                        required_fields.append(active_tracer.name)
        else:
            raise NotImplementedError(f'Thermodynamic K.E. not implemented for {type(equations)}')
        super().__init__(space=space, method=method, required_fields=tuple(required_fields))

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        u = state_fields('u')
        self._setup_thermodynamics(domain, state_fields)
        self.expr = 0.5 * self.rho_averaged * (1 + self.r_t) * dot(u, u)
        super().setup(domain, state_fields, space=domain.spaces("DG"))


class DewpointTemperature(IterativeDiagnosticField):
    """
    The dewpoint temperature diagnostic field. The temperature to which air
    must be cooled in order to become saturated.

    Note: this will not give sensible answers in the absence of water vapour.
    """
    name = "Dewpoint"

    def __init__(self, equations, space=None, method='interpolate',
                 num_iterations=3, gamma=1.0):
        """
        Args:
            equations (:class:`PrognosticEquationSet`): the equation set being
                solved by the model.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
            num_iterations (integer, optional): number of times to iteratively
                evaluate the expression. Defaults to 3.
            gamma (float, optional): weight given to previous guess, which is
                used to avoid numerical instabilities. Defaults to 1.0.
        """

        self.parameters = equations.parameters
        super().__init__(space=space, method=method,
                         num_iterations=num_iterations, gamma=gamma)

    def implicit_expr(self, domain, state_fields):
        """
        The implicit UFL expression for the diagnostic, which should depend
        on self.field

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """

        theta = state_fields('theta')
        rho = state_fields('rho')
        if 'water_vapour' in state_fields._field_names:
            r_v = state_fields('water_vapour')
        else:
            raise RuntimeError('Dewpoint temperature diagnostic should only'
                               + 'be used with water vapour')

        exner = tde.exner_pressure(self.parameters, rho, theta)
        pressure = tde.p(self.parameters, exner)
        temperature = tde.T(self.parameters, theta, exner, r_v=r_v)
        r_sat = tde.r_sat(self.parameters, self.field, pressure)

        return self.field - temperature*(r_sat - r_v)

    def set_first_guess(self, domain, state_fields):
        """
        The first guess of the diagnostic, set to be the dry temperature.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        theta = state_fields('theta')
        rho = state_fields('rho')
        if 'water_vapour' in state_fields._field_names:
            r_v = state_fields('water_vapour')
        else:
            raise RuntimeError('Dewpoint temperature diagnostic should only'
                               + 'be used with water vapour')

        exner = tde.exner_pressure(self.parameters, rho, theta)
        temperature = tde.T(self.parameters, theta, exner, r_v=r_v)

        return temperature


class WetBulbTemperature(IterativeDiagnosticField):
    """
    The wet-bulb temperature diagnostic field. The temperature of air cooled to
    saturation by the evaporation of water.
    """
    name = "WetBulb"

    def __init__(self, equations, space=None, method='interpolate',
                 num_iterations=3, gamma=0.5):
        """
        Args:
            equations (:class:`PrognosticEquationSet`): the equation set being
                solved by the model.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
            num_iterations (integer, optional): number of times to iteratively
                evaluate the expression. Defaults to 3.
            gamma (float, optional): weight given to previous guess, which is
                used to avoid numerical instabilities. Defaults to 0.8.
        """

        self.parameters = equations.parameters
        super().__init__(space=space, method=method,
                         num_iterations=num_iterations, gamma=gamma)

    def implicit_expr(self, domain, state_fields):
        """
        The implicit UFL expression for the diagnostic, which should depend
        on self.field

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """

        theta = state_fields('theta')
        rho = state_fields('rho')
        if 'water_vapour' in state_fields._field_names:
            r_v = state_fields('water_vapour')
        else:
            r_v = 0.0*theta  # zero expression

        exner = tde.exner_pressure(self.parameters, rho, theta)
        pressure = tde.p(self.parameters, exner)
        temperature = tde.T(self.parameters, theta, exner, r_v=r_v)
        r_sat = tde.r_sat(self.parameters, self.field, pressure)

        # In the comments, preserve a simpler expression:
        # L_v0 = self.parameters.L_v0
        # R_v = self.parameters.R_v
        # c_v = self.parameters.cv
        # return L_v0 / R_v + (R_v*temperature - L_v0)/R_v * exp(R_v/c_v*(r_sat - r_v))

        # Reduce verbosity by introducing intermediate variables
        b = -self.parameters.L_v0 - (self.parameters.c_pl - self.parameters.c_pv)*self.parameters.T_0
        a = self.parameters.R_v + self.parameters.c_pl - self.parameters.c_pv
        A = self.parameters.c_vv
        B = self.parameters.cv

        return - b / a + (a*temperature + b) / a * ((A*r_sat + B) / (A*r_v + B))**(a/A)

    def set_first_guess(self, domain, state_fields):
        """
        The first guess of the diagnostic, set to be the dry temperature.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        theta = state_fields('theta')
        rho = state_fields('rho')
        if 'water_vapour' in state_fields._field_names:
            r_v = state_fields('water_vapour')
        else:
            r_v = 0.0*theta  # zero expression

        exner = tde.exner_pressure(self.parameters, rho, theta)
        temperature = tde.T(self.parameters, theta, exner, r_v=r_v)

        return temperature


class Temperature(ThermodynamicDiagnostic):
    """The absolute temperature diagnostic field."""
    name = "Temperature"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        self._setup_thermodynamics(domain, state_fields)
        self.expr = self.T
        super().setup(domain, state_fields, space=self.Vtheta)


class Theta_d(ThermodynamicDiagnostic):
    """The dry potential temperature diagnostic field."""
    name = "Theta_d"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        self._setup_thermodynamics(domain, state_fields)
        self.expr = self.theta / (1 + self.r_v * self.parameters.R_v / self.parameters.R_d)
        super().setup(domain, state_fields, space=self.Vtheta)


class RelativeHumidity(ThermodynamicDiagnostic):
    """The relative humidity diagnostic field."""
    name = "RelativeHumidity"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        self._setup_thermodynamics(domain, state_fields)
        self.expr = tde.RH(self.parameters, self.r_v, self.T, self.p)
        super().setup(domain, state_fields, space=self.Vtheta)


class Pressure(ThermodynamicDiagnostic):
    """The pressure field computed in the 'theta' space."""
    name = "Pressure_Vt"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        self._setup_thermodynamics(domain, state_fields)
        self.expr = self.p
        super().setup(domain, state_fields, space=self.Vtheta)


class Exner_Vt(ThermodynamicDiagnostic):
    """The Exner pressure field computed in the 'theta' space."""
    name = "Exner_Vt"

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        self._setup_thermodynamics(domain, state_fields)
        self.expr = self.exner
        super().setup(domain, state_fields, space=self.Vtheta)


# TODO: this doesn't contain the effects of moisture
# TODO: this has not been implemented for other equation sets
class HydrostaticImbalance(DiagnosticField):
    """Hydrostatic imbalance diagnostic field."""
    name = "HydrostaticImbalance"

    def __init__(self, equations, space=None, method='interpolate'):
        """
        Args:
            equations (:class:`PrognosticEquationSet`): the equation set being
                solved by the model.
            space (:class:`FunctionSpace`, optional): the function space to
                evaluate the diagnostic field in. Defaults to None, in which
                case a default space will be chosen for this diagnostic.
            method (str, optional): a string specifying the method of evaluation
                for this diagnostic. Valid options are 'interpolate', 'project',
                'assign' and 'solve'. Defaults to 'interpolate'.
        """
        # Work out required fields
        if isinstance(equations, CompressibleEulerEquations):
            required_fields = ['rho', 'theta', 'rho_bar', 'theta_bar']
            if equations.active_tracers is not None:
                for active_tracer in equations.active_tracers:
                    if active_tracer.chemical == 'H2O':
                        required_fields.append(active_tracer.name)
            self.equations = equations
            self.parameters = equations.parameters
        else:
            raise NotImplementedError(f'Hydrostatic Imbalance not implemented for {type(equations)}')
        super().__init__(space=space, method=method, required_fields=required_fields)

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        Vu = domain.spaces("HDiv")
        rho = state_fields("rho")
        rhobar = state_fields("rho_bar")
        theta = state_fields("theta")
        thetabar = state_fields("theta_bar")
        exner = tde.exner_pressure(self.parameters, rho, theta)
        exnerbar = tde.exner_pressure(self.parameters, rhobar, thetabar)

        cp = Constant(self.parameters.cp)
        n = FacetNormal(domain.mesh)

        dx_qp = dx(degree=domain.max_quad_degree)
        dS_v_qp = dS_v(degree=domain.max_quad_degree)

        # TODO: not sure about this expression!
        # Gravity does not appear, and why are there reference profiles?
        F = TrialFunction(Vu)
        w = TestFunction(Vu)
        imbalance = Function(Vu)
        a = inner(w, F)*dx
        L = (- cp*div((theta-thetabar)*w)*exnerbar*dx_qp
             + cp*jump((theta-thetabar)*w, n)*avg(exnerbar)*dS_v_qp
             - cp*div(thetabar*w)*(exner-exnerbar)*dx_qp
             + cp*jump(thetabar*w, n)*avg(exner-exnerbar)*dS_v_qp)

        bcs = self.equations.bcs['u']

        imbalanceproblem = LinearVariationalProblem(a, L, imbalance, bcs=bcs,
                                                    constant_jacobian=True)
        self.imbalance_solver = LinearVariationalSolver(imbalanceproblem)
        self.expr = dot(imbalance, domain.k)
        super().setup(domain, state_fields)

    def compute(self):
        """Compute and return the diagnostic field from the current state.
        """
        self.imbalance_solver.solve()
        super().compute()


class Precipitation(DiagnosticField):
    """
    The total precipitation falling through the domain's bottom surface.

    This is normalised by unit area, giving a result in kg / m^2.
    """
    name = "Precipitation"

    def __init__(self):
        self.solve_implemented = True
        required_fields = ('rain', 'rainfall_velocity', 'rho')
        super().__init__(method='solve', required_fields=required_fields)

    def setup(self, domain, state_fields):
        """
        Sets up the :class:`Function` for the diagnostic field.

        Args:
            domain (:class:`Domain`): the model's domain object.
            state_fields (:class:`StateFields`): the model's field container.
        """
        if not hasattr(domain.spaces, "DG0"):
            DG0 = domain.spaces.create_space("DG0", "DG", 0)
        else:
            DG0 = domain.spaces("DG0")
        assert DG0.extruded, 'Cannot compute precipitation on a non-extruded mesh'
        self.space = DG0

        # Gather fields
        rain = state_fields('rain')
        rho = state_fields('rho')
        v = state_fields('rainfall_velocity')
        # Set up problem
        self.phi = TestFunction(DG0)
        flux = TrialFunction(DG0)
        self.flux = Function(DG0)  # Flux to solve for
        area = Function(DG0)  # Need to compute normalisation (area)

        eqn_lhs = self.phi * flux * dx
        area_rhs = self.phi * ds_b
        eqn_rhs = domain.dt * self.phi * (rain * dot(- v, domain.k) * rho / area) * ds_b

        # Compute area normalisation
        area_prob = LinearVariationalProblem(eqn_lhs, area_rhs, area,
                                             constant_jacobian=True)
        area_solver = LinearVariationalSolver(area_prob)
        area_solver.solve()

        # setup solver
        rain_prob = LinearVariationalProblem(eqn_lhs, eqn_rhs, self.flux,
                                             constant_jacobian=True)
        self.solver = LinearVariationalSolver(rain_prob)
        self.field = state_fields(self.name, space=DG0, dump=True, pick_up=True)
        # Initialise field to zero, if picking up this will be overridden
        self.field.assign(0.0)

    def compute(self):
        """Increment the precipitation diagnostic."""
        self.solver.solve()
        self.field.assign(self.field + self.flux)
