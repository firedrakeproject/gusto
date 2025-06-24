"""
Objects to describe physics parametrisations for the boundary layer, such as
drag and turbulence."""

from firedrake import (
    conditional, Function, dx, sqrt, dot, Constant, grad,
    TestFunctions, split, inner, Projector, exp, avg, outer, FacetNormal,
    SpatialCoordinate, dS_v, FunctionSpace
)
from firedrake.__future__ import Interpolator
from firedrake.fml import subject
from gusto.core.equation_configuration import BoundaryLayerParameters
from gusto.recovery import Recoverer, BoundaryMethod
from gusto.equations import CompressibleEulerEquations
from gusto.core.labels import prognostic, source_label
from gusto.core.logging import logger
from gusto.equations import thermodynamics
from gusto.physics.physics_parametrisation import PhysicsParametrisation

__all__ = ["SurfaceFluxes", "WindDrag", "StaticAdjustment",
           "SuppressVerticalWind", "BoundaryLayerMixing"]


class SurfaceFluxes(PhysicsParametrisation):
    """
    Prescribed surface temperature and moisture fluxes, to be used in aquaplanet
    simulations as Sea Surface Temperature fluxes. This formulation is taken
    from the DCMIP (2016) test case document.

    These can be used either with an in-built implicit formulation, or with a
    generic time discretisation.

    Written to assume that dry density is unchanged by the parametrisation.
    """

    def __init__(self, equation, T_surface_expr, vapour_name=None,
                 implicit_formulation=False, parameters=None):
        """
        Args:
            equation (:class:`PrognosticEquationSet`): the model's equation.
            T_surface_expr (:class:`ufl.Expr`): the surface temperature.
            vapour_name (str, optional): name of the water vapour variable.
                Defaults to None, in which case no moisture fluxes are applied.
            implicit_formulation (bool, optional): whether the scheme is already
                put into a Backwards Euler formulation (which allows this scheme
                to actually be used with a Forwards Euler or other explicit time
                discretisation). Otherwise, this is formulated more generally
                and can be used with any time stepper. Defaults to False.
            parameters (:class:`BoundaryLayerParameters`): configuration object
                giving the parameters for boundary and surface level schemes.
                Defaults to None, in which case default values are used.
        """

        # -------------------------------------------------------------------- #
        # Check arguments and generic initialisation
        # -------------------------------------------------------------------- #
        if not isinstance(equation, CompressibleEulerEquations):
            raise ValueError("Surface fluxes can only be used with Compressible Euler equations")

        if vapour_name is not None:
            if vapour_name not in equation.field_names:
                raise ValueError(f"Field {vapour_name} does not exist in the equation set")

        if parameters is None:
            parameters = BoundaryLayerParameters()

        label_name = 'surface_flux'
        super().__init__(equation, label_name, parameters=None)

        self.implicit_formulation = implicit_formulation
        self.X = Function(equation.X.function_space())
        self.dt = Constant(0.0)
        self.source = Function(equation.X.function_space())

        # -------------------------------------------------------------------- #
        # Extract prognostic variables
        # -------------------------------------------------------------------- #
        u_idx = equation.field_names.index('u')
        T_idx = equation.field_names.index('theta')
        self.T_idx = T_idx
        rho_idx = equation.field_names.index('rho')
        if vapour_name is not None:
            self.m_v_idx = equation.field_names.index(vapour_name)

        X = self.X
        tests = TestFunctions(X.function_space()) if implicit_formulation else equation.tests

        u = split(X)[u_idx]
        rho = split(X)[rho_idx]
        theta_vd = split(X)[T_idx]
        test_theta = tests[T_idx]

        if vapour_name is not None:
            m_v = split(X)[self.m_v_idx]
            test_m_v = tests[self.m_v_idx]
        else:
            m_v = None

        if implicit_formulation:
            # Need to evaluate rho at theta-points
            boundary_method = BoundaryMethod.extruded if equation.domain.vertical_degree == 0 else None
            rho_averaged = Function(equation.function_space.sub(T_idx))
            self.rho_recoverer = Recoverer(rho, rho_averaged, boundary_method=boundary_method)
            exner = thermodynamics.exner_pressure(equation.parameters, rho_averaged, theta_vd)
        else:
            # Exner is more general expression
            exner = thermodynamics.exner_pressure(equation.parameters, rho, theta_vd)

        # Alternative variables
        T = thermodynamics.T(equation.parameters, theta_vd, exner, r_v=m_v)
        p = thermodynamics.p(equation.parameters, exner)

        # -------------------------------------------------------------------- #
        # Expressions for surface fluxes
        # -------------------------------------------------------------------- #
        z = equation.domain.height_above_surface
        z_a = parameters.height_surface_layer
        surface_expr = conditional(z < z_a, Constant(1.0), Constant(0.0))

        u_hori = u - equation.domain.k*dot(u, equation.domain.k)
        u_hori_mag = sqrt(dot(u_hori, u_hori))

        C_H = parameters.coeff_heat
        C_E = parameters.coeff_evap

        # Implicit formulation ----------------------------------------------- #
        # For use with ForwardEuler only, as implicit solution is hand-written
        if implicit_formulation:
            self.source_interpolators = []

            # First specify T_np1 expression
            T_np1_expr = ((T + C_H*u_hori_mag*T_surface_expr*self.dt/z_a)
                          / (1 + C_H*u_hori_mag*self.dt/z_a))

            # If moist formulation, determine next vapour value
            if vapour_name is not None:
                self.source_mv_int = self.source.subfunctions[self.m_v_idx]
                self.source_mv = split(self.source)[self.m_v_idx]
                mv_sat = thermodynamics.r_sat(equation.parameters, T, p)
                mv_np1_expr = ((m_v + C_E*u_hori_mag*mv_sat*self.dt/z_a)
                               / (1 + C_E*u_hori_mag*self.dt/z_a))
                dmv_expr = surface_expr * (mv_np1_expr - m_v) / self.dt
                source_mv_expr = test_m_v * self.source_mv * dx

                self.source_interpolators.append(Interpolator(dmv_expr, self.source_mv_int))
                equation.residual -= source_label(
                    self.label(subject(prognostic(source_mv_expr, vapour_name), self.source), self.evaluate)
                )

                # Moisture needs including in theta_vd expression
                # NB: still using old pressure here, which implies constant p?
                epsilon = equation.parameters.R_d / equation.parameters.R_v
                theta_np1_expr = (thermodynamics.theta(equation.parameters, T_np1_expr, p)
                                  * (1 + mv_np1_expr / epsilon))

            else:
                theta_np1_expr = thermodynamics.theta(equation.parameters, T_np1_expr, p)

            self.source_theta_vd = split(self.source)[self.T_idx]
            self.source_theta_vd_int = self.source.subfunctions[self.T_idx]
            dtheta_vd_expr = surface_expr * (theta_np1_expr - theta_vd) / self.dt
            source_theta_expr = test_theta * self.source_theta_vd * dx
            self.source_interpolators.append(Interpolator(dtheta_vd_expr, self.source_theta_vd_int))
            equation.residual -= source_label(
                self.label(subject(prognostic(source_theta_expr, 'theta'), self.source), self.evaluate)
            )

        # General formulation ------------------------------------------------ #
        else:
            # Construct underlying expressions
            kappa = equation.parameters.kappa
            dT_dt = surface_expr * C_H * u_hori_mag * (T_surface_expr - T) / z_a

            if vapour_name is not None:
                mv_sat = thermodynamics.r_sat(equation.parameters, T, p)
                dmv_dt = surface_expr * C_E * u_hori_mag * (mv_sat - m_v) / z_a
                source_mv_expr = test_m_v * dmv_dt * dx
                equation.residual -= self.label(
                    prognostic(subject(source_mv_expr, X),
                               vapour_name), self.evaluate)

                # Theta expression depends on dmv_dt
                epsilon = equation.parameters.R_d / equation.parameters.R_v
                dtheta_vd_dt = (dT_dt * ((1 + m_v / epsilon) / exner - kappa * theta_vd / (rho * T))
                                + dmv_dt * (T / (epsilon * exner) - kappa * theta_vd / (epsilon + m_v)))
            else:
                dtheta_vd_dt = dT_dt * (1 / exner - kappa * theta_vd / (rho * T))

            dx_reduced = dx(degree=4)
            source_theta_expr = test_theta * dtheta_vd_dt * dx_reduced

            equation.residual -= self.label(
                subject(prognostic(source_theta_expr, 'theta'), X), self.evaluate)

    def evaluate(self, x_in, dt, x_out=None):
        """
        Evaluates the source term generated by the physics. This does nothing if
        the implicit formulation is not used.

        Args:
            x_in: (:class: 'Function'): the (mixed) field to be evolved.
            dt: (:class: 'Constant'): the timestep, which can be the time
                interval for the scheme.
            x_out: (:class:`Function`, optional): the (mixed) source
                                                  field to be outputed.
        """

        logger.info(f'Evaluating physics parametrisation {self.label.label}')

        if self.implicit_formulation:
            self.X.assign(x_in)
            self.dt.assign(dt)
            self.rho_recoverer.project()
            for source_interpolator in self.source_interpolators:
                source_interpolator.interpolate()
            # If a source output is provided, assign the source term to it
            if x_out is not None:
                x_out.assign(self.source)


class WindDrag(PhysicsParametrisation):
    """
    A simple surface wind drag scheme. This formulation is taken from the DCMIP
    (2016) test case document.

    These can be used either with an in-built implicit formulation, or with a
    generic time discretisation.
    """

    def __init__(self, equation, implicit_formulation=False, parameters=None):
        """
        Args:
            equation (:class:`PrognosticEquationSet`): the model's equation.
            implicit_formulation (bool, optional): whether the scheme is already
                put into a Backwards Euler formulation (which allows this scheme
                to actually be used with a Forwards Euler or other explicit time
                discretisation). Otherwise, this is formulated more generally
                and can be used with any time stepper. Defaults to False.
            parameters (:class:`BoundaryLayerParameters`): configuration object
                giving the parameters for boundary and surface level schemes.
                Defaults to None, in which case default values are used.
        """

        # -------------------------------------------------------------------- #
        # Check arguments and generic initialisation
        # -------------------------------------------------------------------- #
        if not isinstance(equation, CompressibleEulerEquations):
            raise ValueError("Wind drag can only be used with Compressible Euler equations")

        if parameters is None:
            parameters = BoundaryLayerParameters()

        label_name = 'wind_drag'
        super().__init__(equation, label_name, parameters=None)

        k = equation.domain.k
        self.implicit_formulation = implicit_formulation
        self.X = Function(equation.X.function_space())
        self.dt = Constant(0.0)

        # -------------------------------------------------------------------- #
        # Extract prognostic variables
        # -------------------------------------------------------------------- #
        u_idx = equation.field_names.index('u')

        X = self.X
        tests = TestFunctions(X.function_space()) if implicit_formulation else equation.tests

        test = tests[u_idx]

        u = split(X)[u_idx]
        u_hori = u - k*dot(u, k)
        u_hori_mag = sqrt(dot(u_hori, u_hori))

        # -------------------------------------------------------------------- #
        # Expressions for wind drag
        # -------------------------------------------------------------------- #
        z = equation.domain.height_above_surface
        z_a = parameters.height_surface_layer
        surface_expr = conditional(z < z_a, Constant(1.0), Constant(0.0))

        C_D0 = parameters.coeff_drag_0
        C_D1 = parameters.coeff_drag_1
        C_D2 = parameters.coeff_drag_2

        C_D = conditional(u_hori_mag < 20.0, C_D0 + C_D1*u_hori_mag, C_D2)

        # Implicit formulation ----------------------------------------------- #
        # For use with ForwardEuler only, as implicit solution is hand-written
        if implicit_formulation:

            # First specify T_np1 expression
            self.source = Function(equation.X.function_space())
            source_u = split(self.source)[u_idx]
            source_u_proj = self.source.subfunctions[u_idx]
            u_np1_expr = u_hori / (1 + C_D*u_hori_mag*self.dt/z_a)

            du_expr = surface_expr * (u_np1_expr - u_hori) / self.dt

            project_params = {
                'quadrature_degree': equation.domain.max_quad_degree
            }
            self.source_projector = Projector(
                du_expr, source_u_proj, form_compiler_parameters=project_params
            )

            source_expr = inner(test, source_u - k*dot(source_u, k)) * dx
            equation.residual -= source_label(
                self.label(subject(prognostic(source_expr, 'u'), self.source), self.evaluate)
            )

        # General formulation ------------------------------------------------ #
        else:
            # Construct underlying expressions
            du_dt = -surface_expr * C_D * u_hori_mag * u_hori / z_a

            dx_reduced = dx(degree=4)
            source_expr = inner(test, du_dt) * dx_reduced

            equation.residual -= self.label(subject(prognostic(source_expr, 'u'), X), self.evaluate)

    def evaluate(self, x_in, dt, x_out=None):
        """
        Evaluates the source term generated by the physics. This does nothing if
        the implicit formulation is not used.

        Args:
            x_in: (:class: 'Function'): the (mixed) field to be evolved.
            dt: (:class: 'Constant'): the timestep, which can be the time
                interval for the scheme.
            x_out: (:class:`Function`, optional): the (mixed) source
                                                  field to be outputed.
        """

        logger.info(f'Evaluating physics parametrisation {self.label.label}')

        if self.implicit_formulation:
            self.X.assign(x_in)
            self.dt.assign(dt)
            self.source_projector.project()
            # If a source output is provided, assign the source term to it
            if x_out is not None:
                x_out.assign(self.source)


class StaticAdjustment(PhysicsParametrisation):
    """
    A scheme to provide static adjustment, by sorting the potential temperature
    values in a column so that they are increasing with height.
    """

    def __init__(self, equation, theta_variable='theta_vd'):
        """
        Args:
            equation (:class:`PrognosticEquationSet`): the model's equation.
            theta_variable (str, optional): which theta variable to sort the
                profile for. Valid options are "theta" or "theta_vd". Defaults
                to "theta_vd".
        """

        self.explicit_only = True
        from functools import partial

        # -------------------------------------------------------------------- #
        # Check arguments and generic initialisation
        # -------------------------------------------------------------------- #
        if not isinstance(equation, CompressibleEulerEquations):
            raise ValueError("Static adjustment can only be used with Compressible Euler equations")

        if theta_variable not in ['theta', 'theta_vd']:
            raise ValueError('Static adjustment: theta variable '
                             + f'{theta_variable} not valid')

        label_name = 'static_adjustment'
        super().__init__(equation, label_name, parameters=equation.parameters)

        self.X = Function(equation.X.function_space())
        self.dt = Constant(0.0)

        # -------------------------------------------------------------------- #
        # Extract prognostic variables
        # -------------------------------------------------------------------- #

        theta_idx = equation.field_names.index('theta')
        Vt = equation.spaces[theta_idx]
        self.theta_to_sort = Function(Vt)
        sorted_theta = Function(Vt)
        theta = self.X.subfunctions[theta_idx]

        if theta_variable == 'theta' and 'water_vapour' in equation.field_names:
            Rv = equation.parameters.R_v
            Rd = equation.parameters.R_d
            mv_idx = equation.field_names.index('water_vapour')
            mv = self.X.subfunctions[mv_idx]
            self.get_theta_variable = Interpolator(theta / (1 + mv*Rv/Rd), self.theta_to_sort)
            self.set_theta_variable = Interpolator(self.theta_to_sort * (1 + mv*Rv/Rd), sorted_theta)
        else:
            self.get_theta_variable = Interpolator(theta, self.theta_to_sort)
            self.set_theta_variable = Interpolator(self.theta_to_sort, sorted_theta)

        # -------------------------------------------------------------------- #
        # Set up routines to reshape data
        # -------------------------------------------------------------------- #

        domain = equation.domain
        self.get_column_data = partial(domain.coords.get_column_data,
                                       field=self.theta_to_sort,
                                       domain=domain)
        self.set_column_data = domain.coords.set_field_from_column_data

        # -------------------------------------------------------------------- #
        # Set source term
        # -------------------------------------------------------------------- #

        tests = TestFunctions(self.X.function_space())
        test = tests[theta_idx]

        source_expr = inner(test, sorted_theta - theta) / self.dt * dx

        equation.residual -= self.label(subject(prognostic(source_expr, 'theta'), equation.X), self.evaluate)

    def evaluate(self, x_in, dt, x_out=None):
        """
        Evaluates the source term generated by the physics. This does nothing if
        the implicit formulation is not used.

        Args:
            x_in: (:class: 'Function'): the (mixed) field to be evolved.
            dt: (:class: 'Constant'): the timestep, which can be the time
                interval for the scheme.
            x_out: (:class:`Function`, optional): the (mixed) source
                                                  field to be outputed.
        """

        logger.info(f'Evaluating physics parametrisation {self.label.label}')

        self.X.assign(x_in)
        self.dt.assign(dt)

        self.get_theta_variable.interpolate()
        theta_column_data, index_data = self.get_column_data()
        for col in range(theta_column_data.shape[0]):
            theta_column_data[col].sort()
        self.set_column_data(self.theta_to_sort, theta_column_data, index_data)
        self.set_theta_variable.interpolate()

        if x_out is not None:
            raise NotImplementedError("Static adjustment does not output a source term, "
                                      "or a non-interpolated/projected expression and hence "
                                      "cannot be used in a nonsplit physics formulation.")


class SuppressVerticalWind(PhysicsParametrisation):
    """
    Suppresses the model's vertical wind, reducing it to zero. This is used for
    instance in a model's spin up period.
    """

    def __init__(self, equation, spin_up_period):
        """
        Args:
            equation (:class:`PrognosticEquationSet`): the model's equation.
            spin_up_period (`ufl.Constant`): this parametrisation is applied
                while the time is less than this -- corresponding to a spin up
                period.
        """

        self.explicit_only = True

        # -------------------------------------------------------------------- #
        # Check arguments and generic initialisation
        # -------------------------------------------------------------------- #

        domain = equation.domain

        if not domain.mesh.extruded:
            raise RuntimeError("Suppress vertical wind can only be used with "
                               + "extruded meshes")

        label_name = 'suppress_vertical_wind'
        super().__init__(equation, label_name, parameters=equation.parameters)

        self.X = Function(equation.X.function_space())
        self.dt = Constant(0.0)
        self.t = domain.t
        self.spin_up_period = Constant(spin_up_period)
        self.strength = Constant(1.0)
        self.spin_up_done = False

        # -------------------------------------------------------------------- #
        # Extract prognostic variables
        # -------------------------------------------------------------------- #

        u_idx = equation.field_names.index('u')
        wind = self.X.subfunctions[u_idx]

        # -------------------------------------------------------------------- #
        # Set source term
        # -------------------------------------------------------------------- #

        tests = TestFunctions(self.X.function_space())
        test = tests[u_idx]

        # The sink should be just the value of the current vertical wind
        source_expr = -self.strength * inner(test, domain.k*dot(domain.k, wind)) / self.dt * dx

        equation.residual -= self.label(subject(prognostic(source_expr, 'u'), equation.X), self.evaluate)

    def evaluate(self, x_in, dt, x_out=None):
        """
        Evaluates the source term generated by the physics. This does nothing if
        the implicit formulation is not used.

        Args:
            x_in: (:class: 'Function'): the (mixed) field to be evolved.
            dt: (:class: 'Constant'): the timestep, which can be the time
                interval for the scheme.
            x_out: (:class:`Function`, optional): the (mixed) source
                                                  field to be outputed.
                                                  This is unused.
        """

        if float(self.t) < float(self.spin_up_period):
            logger.info(f'Evaluating physics parametrisation {self.label.label}')

            self.X.assign(x_in)
            self.dt.assign(dt)

        elif not self.spin_up_done:
            self.strength.assign(0.0)
            self.spin_up_done = True


class BoundaryLayerMixing(PhysicsParametrisation):
    """
    A simple boundary layer mixing scheme. This acts like a vertical diffusion,
    using an interior penalty method.
    """

    def __init__(self, equation, field_name, parameters=None):
        """
        Args:
            equation (:class:`PrognosticEquationSet`): the model's equation.
            field_name (str): name of the field to apply the diffusion to.
            ibp (:class:`IntegrateByParts`, optional): how many times to
                integrate the term by parts. Defaults to IntegrateByParts.ONCE.
            parameters (:class:`BoundaryLayerParameters`): configuration object
                giving the parameters for boundary and surface level schemes.
                Defaults to None, in which case default values are used.
        """

        # -------------------------------------------------------------------- #
        # Check arguments and generic initialisation
        # -------------------------------------------------------------------- #

        if not isinstance(equation, CompressibleEulerEquations):
            raise ValueError("Boundary layer mixing can only be used with Compressible Euler equations")

        if field_name not in equation.field_names:
            raise ValueError(f'field {field_name} not found in equation')

        if parameters is None:
            parameters = BoundaryLayerParameters()

        label_name = f'boundary_layer_mixing_{field_name}'
        super().__init__(equation, label_name, parameters=None)

        self.X = Function(equation.X.function_space())

        # -------------------------------------------------------------------- #
        # Extract prognostic variables
        # -------------------------------------------------------------------- #

        u_idx = equation.field_names.index('u')
        T_idx = equation.field_names.index('theta')
        rho_idx = equation.field_names.index('rho')

        u = split(self.X)[u_idx]
        rho = split(self.X)[rho_idx]
        theta_vd = split(self.X)[T_idx]

        boundary_method = BoundaryMethod.extruded if equation.domain.vertical_degree == 0 else None
        rho_averaged = Function(equation.function_space.sub(T_idx))
        self.rho_recoverer = Recoverer(rho, rho_averaged, boundary_method=boundary_method)
        exner = thermodynamics.exner_pressure(equation.parameters, rho_averaged, theta_vd)

        # Alternative variables
        p = thermodynamics.p(equation.parameters, exner)
        p_top = Constant(85000.)
        p_strato = Constant(10000.)

        # -------------------------------------------------------------------- #
        # Expressions for diffusivity coefficients
        # -------------------------------------------------------------------- #
        z_a = parameters.height_surface_layer

        domain = equation.domain
        u_hori = u - domain.k*dot(u, domain.k)
        u_hori_mag = sqrt(dot(u_hori, u_hori))

        if field_name == 'u':
            C_D0 = parameters.coeff_drag_0
            C_D1 = parameters.coeff_drag_1
            C_D2 = parameters.coeff_drag_2
            R = FunctionSpace(domain.mesh, "R", 0)
            C_D3 = Function(R).assign(0.0)

            C_D = conditional(u_hori_mag < 20.0,
                              C_D0 + C_D1*u_hori_mag,
                              # To avoid a free index error in UFL, don't just
                              # have a single real in the False condition
                              C_D2 + C_D3*u_hori_mag)
            K = conditional(p > p_top,
                            C_D*u_hori_mag*z_a,
                            C_D*u_hori_mag*z_a*exp(-((p_top - p)/p_strato)**2))

        else:
            C_E = parameters.coeff_evap
            K = conditional(p > p_top,
                            C_E*u_hori_mag*z_a,
                            C_E*u_hori_mag*z_a*exp(-((p_top - p)/p_strato)**2))

        # -------------------------------------------------------------------- #
        # Make source expression
        # -------------------------------------------------------------------- #

        dx_reduced = dx(degree=4)
        dS_v_reduced = dS_v(degree=4)
        # Need to be careful with order of operations here, to correctly index
        # when the field is vector-valued
        d_dz = lambda q: outer(domain.k, dot(grad(q), domain.k))
        n = FacetNormal(domain.mesh)
        # Work out vertical height
        xyz = SpatialCoordinate(domain.mesh)
        Vr = domain.spaces('L2')
        dz = Function(Vr)
        dz.interpolate(dot(domain.k, d_dz(dot(domain.k, xyz))))
        mu = parameters.mu

        X = self.X
        tests = equation.tests

        idx = equation.field_names.index(field_name)
        test = tests[idx]
        field = X.subfunctions[idx]

        if field_name == 'u':
            # Horizontal diffusion only
            field = field - domain.k*dot(field, domain.k)

        # Interior penalty discretisation of vertical diffusion
        source_expr = (
            # Volume term
            rho_averaged*K*inner(d_dz(test/rho_averaged), d_dz(field))*dx_reduced
            # Interior penalty surface term
            - 2*inner(avg(outer(K*field, n)), avg(d_dz(test)))*dS_v_reduced
            - 2*inner(avg(outer(test, n)), avg(d_dz(K*field)))*dS_v_reduced
            + 4*mu*avg(dz)*inner(avg(outer(K*field, n)), avg(outer(test, n)))*dS_v_reduced
        )

        equation.residual += self.label(
            subject(prognostic(source_expr, field_name), X), self.evaluate)

    def evaluate(self, x_in, dt, x_out=None):
        """
        Evaluates the source term generated by the physics. This only recovers
        the density field.

        Args:
            x_in: (:class: 'Function'): the (mixed) field to be evolved.
            dt: (:class: 'Constant'): the timestep, which can be the time
                interval for the scheme.
            x_out: (:class:`Function`, optional): the (mixed) source
                                                  field to be outputed.
        """

        logger.info(f'Evaluating physics parametrisation {self.label.label}')

        self.X.assign(x_in)
        self.rho_recoverer.project()

        if x_out is not None:
            raise NotImplementedError("Boundary layer mixing does not output a source term, "
                                      "or a non-interpolated/projected expression and hence "
                                      "cannot be used in a nonsplit physics formulation.")
