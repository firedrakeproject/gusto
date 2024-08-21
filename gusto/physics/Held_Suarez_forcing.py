import numpy as np
from firedrake import (Interpolator, Function, dx, pi, SpatialCoordinate,
                       split, conditional, ge, sin, dot, ln, cos, inner, Projector)
from firedrake.fml import subject
from gusto.core.coord_transforms import lonlatr_from_xyz
from gusto.recovery import Recoverer, BoundaryMethod
from gusto.physics.physics_parametrisation import PhysicsParametrisation
from gusto.core.labels import prognostic
from gusto.equations import thermodynamics


class Relaxation(PhysicsParametrisation):
    """
    Relaxation term for Held Suarez
    """

    def __init__(self, equation, variable_name, parameters=None):
        """
        Args:
            equation (:class:`PrognosticEquationSet`): the model's equation.
            variable_name (str): the name of the variable

        """
        label_name = f'relaxation_{variable_name}'
        super().__init__(equation, label_name, parameters=None)

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
        self.exner_interpolator = Interpolator(
            thermodynamics.exner_pressure(equation.parameters,
                                          self.rho_averaged, self.theta), self.exner)
        self.sigma = Function(Vt)

        T0stra = 200   # Stratosphere temp
        T0surf = 315   # Surface temperature at equator
        T0horiz = 60   # Equator to pole temperature difference
        T0vert = 10    # Stability parameter
        self.kappa = equation.parameters.kappa
        sigmab = 0.7
        d = 24 * 60 * 60
        taod = 40 * d
        taou = 4 * d

        theta_condition = (T0surf - T0horiz * sin(lat)**2 - (T0vert * ln(self.exner) * cos(lat)**2)/self.kappa) 
        Theta_eq = conditional(T0stra/self.exner >= theta_condition, T0stra/self.exner, theta_condition)

        # timescale of temperature forcing
        tao_cond = (self.sigma**-self.kappa - sigmab) / (1 - sigmab)
        newton_freq = 1 / taod + (1/taou - 1/taod) * conditional(0 >= tao_cond, 0, tao_cond) * cos(lat)**4
        forcing_expr = newton_freq * (self.theta - Theta_eq) 

        # Create source for forcing
        self.source_relaxation = Function(Vt)
        self.source_interpolator = Interpolator(forcing_expr, self.source_relaxation)

        # Add relaxation term to residual
        test = equation.tests[theta_idx]
        dx_reduced = dx(degree=4)
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
        self.exner_interpolator.interpolate()

        # Determine sigma:= exner / exner_surf
        exner_columnwise, index_data = self.domain.coords.get_column_data(self.exner, self.domain)
        sigma_columnwise = np.zeros_like(exner_columnwise)
        for col in range(len(exner_columnwise[:, 0])):
            sigma_columnwise[col, :] = exner_columnwise[col, :] / exner_columnwise[col, 0]
        self.domain.coords.set_field_from_column_data(self.sigma, sigma_columnwise, index_data)

        self.source_interpolator.interpolate()


class RayleighFriction(PhysicsParametrisation):
    """
    Forcing term on the velocity of the form
    F_u = -u / a,
    where a is some friction factor
    """
    def __init__(self, equation, parameters=None):
        """
         Args:
            equation (:class:`PrognosticEquationSet`): the model's equation.
            forcing_coeff (:class:`unsure what to put here`): the coefficient
            determining rate of friction
        """
        label_name = 'rayleigh_friction'
        super().__init__(equation, label_name, parameters=parameters)
        self.parameters = equation.parameters
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

        boundary_method = BoundaryMethod.extruded if self.domain == 0 else None
        self.rho_averaged = Function(Vt)
        self.exner = Function(Vt)
        self.rho_recoverer = Recoverer(rho, self.rho_averaged, boundary_method=boundary_method)
        self.exner_interpolator = Interpolator(
            thermodynamics.exner_pressure(equation.parameters,
                                          self.rho_averaged, self.theta), self.exner)

        self.sigma = Function(Vt)
        sigmab = 0.7
        self.kappa = self.parameters.kappa
        taofric = 24 * 60 * 60

        tao_cond = (self.sigma - sigmab) / (1 - sigmab)
        wind_timescale = conditional(ge(0, tao_cond), 0, tao_cond) / taofric
        forcing_expr = u_hori * wind_timescale

        self.source_friction = Function(Vu)
        self.source_projector = Projector(forcing_expr, self.source_friction)

        tests = equation.tests
        test = tests[u_idx]
        dx_reduced = dx(degree=4)
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
        self.exner_interpolator.interpolate
        # Determine sigma:= exner / exner_surf
        exner_columnwise, index_data = self.domain.coords.get_column_data(self.exner, self.domain)
        sigma_columnwise = np.zeros_like(exner_columnwise)
        for col in range(len(exner_columnwise[:, 0])):
            sigma_columnwise[col, :] = exner_columnwise[col, :] / exner_columnwise[col, 0]
        self.domain.coords.set_field_from_column_data(self.sigma, sigma_columnwise, index_data)

        self.source_projector.project()
