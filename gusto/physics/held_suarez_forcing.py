import numpy as np
from firedrake import (Function, dx, pi, SpatialCoordinate,
                       split, conditional, ge, sin, dot, ln, cos, inner,
                       Projector, assemble)
from firedrake.fml import subject
from firedrake.__future__ import interpolate
from gusto.core.coord_transforms import lonlatr_from_xyz
from gusto.recovery import Recoverer, BoundaryMethod
from gusto.physics.physics_parametrisation import PhysicsParametrisation
from gusto.core.labels import prognostic
from gusto.equations import thermodynamics
from gusto.core.equation_configuration import HeldSuarezParameters
from gusto.core import logger


class Relaxation(PhysicsParametrisation):
    """
    Relaxation term for Held Suarez
    """

    def __init__(self, equation, variable_name, parameters, hs_parameters=None):
        """
        Args:
            equation (:class:`PrognosticEquationSet`): the model's equation.
            variable_name (str): the name of the variable
            hs_parameters (:class'Configuration'): contains the parameters for the Held-suariez test case

        """
        label_name = f'relaxation_{variable_name}'
        if hs_parameters is None:
            hs_parameters = HeldSuarezParameters(equation.domain.mesh)
            logger.warning('Using default Held-Suarez parameters')
        super().__init__(equation, label_name, hs_parameters)

        if equation.domain.on_sphere:
            x, y, z = SpatialCoordinate(equation.domain.mesh)
            _, lat, _ = lonlatr_from_xyz(x, y, z)
        else:
            # TODO: this could be determined some other way
            # Take a mid-latitude
            lat = pi / 4

        self.X = Function(equation.X.function_space())
        X = self.X
        self.domain = equation.domain
        theta_idx = equation.field_names.index('theta')
        self.theta = X.subfunctions[theta_idx]
        Vt = equation.domain.spaces('theta')
        rho_idx = equation.field_names.index('rho')
        rho = split(X)[rho_idx]

        boundary_method = BoundaryMethod.extruded if equation.domain.vertical_degree == 0 else None
        self.rho_averaged = Function(Vt)
        self.rho_recoverer = Recoverer(rho, self.rho_averaged, boundary_method=boundary_method)
        self.exner = Function(Vt)
        self.exner_interpolator = lambda: assemble(
            interpolate(thermodynamics.exner_pressure(equation.parameters, self.rho_averaged, self.theta), Vt),
            tensor=self.exner
        )
        self.sigma = Function(Vt)
        kappa = equation.parameters.kappa

        T0surf = hs_parameters.T0surf
        T0horiz = hs_parameters.T0horiz
        T0vert = hs_parameters.T0vert
        T0stra = hs_parameters.T0stra

        sigma_b = hs_parameters.sigmab
        tau_d = hs_parameters.tau_d
        tau_u = hs_parameters.tau_u

        theta_condition = (T0surf - T0horiz * sin(lat)**2 - (T0vert * ln(self.exner) * cos(lat)**2)/kappa)
        Theta_eq = conditional(T0stra/self.exner >= theta_condition, T0stra/self.exner, theta_condition)

        # timescale of temperature forcing
        tau_cond = (self.sigma**(1/kappa) - sigma_b) / (1 - sigma_b)
        newton_freq = 1 / tau_d + (1/tau_u - 1/tau_d) * conditional(0 >= tau_cond, 0, tau_cond) * cos(lat)**4
        forcing_expr = newton_freq * (self.theta - Theta_eq)

        # Create source for forcing
        self.source_relaxation = Function(Vt)
        self.source_interpolator = lambda: assemble(interpolate(forcing_expr, Vt), tensor=self.source_relaxation)

        # Add relaxation term to residual
        test = equation.tests[theta_idx]
        dx_reduced = dx(degree=equation.domain.max_quad_degree)
        forcing_form = test * self.source_relaxation * dx_reduced
        equation.residual += self.label(subject(prognostic(forcing_form, 'theta'), X), self.evaluate)

    def evaluate(self, x_in, dt):
        """
        Evalutes the source term generated by the physics.

        Args:
            x_in: (:class:`Function`): the (mixed) field to be evolved.
            dt: (:class:`Constant`): the timestep, which can be the time
                interval for the scheme.
        """
        self.X.assign(x_in)
        self.rho_recoverer.project()
        self.exner_interpolator()

        # Determine sigma:= exner / exner_surf
        exner_columnwise, index_data = self.domain.coords.get_column_data(self.exner, self.domain)
        sigma_columnwise = np.zeros_like(exner_columnwise)
        for col in range(len(exner_columnwise[:, 0])):
            sigma_columnwise[col, :] = exner_columnwise[col, :] / exner_columnwise[col, 0]
        self.domain.coords.set_field_from_column_data(self.sigma, sigma_columnwise, index_data)

        self.source_interpolator()


class RayleighFriction(PhysicsParametrisation):
    """
    Forcing term on the velocity of the form
    F_u = -u / a,
    where a is some friction factor
    """
    def __init__(self, equation, hs_parameters=None):
        """
         Args:
            equation (:class:`PrognosticEquationSet`): the model's equation.
            hs_parameters (:class'Configuration'): contains the parameters for the Held-suariez test case
        """
        label_name = 'rayleigh_friction'
        if hs_parameters is None:
            hs_parameters = HeldSuarezParameters(equation.domain.mesh)
            logger.warning('Using default Held-Suarez parameters')
        super().__init__(equation, label_name, hs_parameters)

        self.domain = equation.domain
        self.X = Function(equation.X.function_space())
        X = self.X
        k = equation.domain.k
        u_idx = equation.field_names.index('u')
        u = split(X)[u_idx]
        theta_idx = equation.field_names.index('theta')
        self.theta = X.subfunctions[theta_idx]
        rho_idx = equation.field_names.index('rho')
        rho = split(X)[rho_idx]
        Vt = equation.domain.spaces('theta')
        Vu = equation.domain.spaces('HDiv')
        u_hori = u - k*dot(u, k)

        boundary_method = BoundaryMethod.extruded if equation.domain.vertical_degree == 0 else None
        self.rho_averaged = Function(Vt)
        self.exner = Function(Vt)
        self.rho_recoverer = Recoverer(rho, self.rho_averaged, boundary_method=boundary_method)
        self.exner_interpolator = lambda: assemble(
            interpolate(thermodynamics.exner_pressure(equation.parameters, self.rho_averaged, self.theta), Vt),
            tensor=self.exner
        )

        self.sigma = Function(Vt)
        sigmab = hs_parameters.sigmab
        kappa = equation.parameters.kappa
        tau_fric = 24 * 60 * 60

        tau_cond = (self.sigma**(1/kappa) - sigmab) / (1 - sigmab)
        wind_timescale = conditional(ge(0, tau_cond), 0, tau_cond) / tau_fric
        forcing_expr = u_hori * wind_timescale

        self.source_friction = Function(Vu)
        self.source_projector = Projector(forcing_expr, self.source_friction)

        tests = equation.tests
        test = tests[u_idx]
        dx_reduced = dx(degree=equation.domain.max_quad_degree)
        source_form = inner(test, self.source_friction) * dx_reduced
        equation.residual += self.label(subject(prognostic(source_form, 'u'), X), self.evaluate)

    def evaluate(self, x_in, dt):
        """
        Evaluates the source term generated by the physics. This does nothing if
        the implicit formulation is not used.

        Args:
            x_in: (:class: 'Function'): the (mixed) field to be evolved.
            dt: (:class: 'Constant'): the timestep, which can be the time
                interval for the scheme.
        """
        self.X.assign(x_in)
        self.rho_recoverer.project()
        self.exner_interpolator()
        # Determine sigma:= exner / exner_surf
        exner_columnwise, index_data = self.domain.coords.get_column_data(self.exner, self.domain)
        sigma_columnwise = np.zeros_like(exner_columnwise)
        for col in range(len(exner_columnwise[:, 0])):
            sigma_columnwise[col, :] = exner_columnwise[col, :] / exner_columnwise[col, 0]
        self.domain.coords.set_field_from_column_data(self.sigma, sigma_columnwise, index_data)

        self.source_projector.project()
