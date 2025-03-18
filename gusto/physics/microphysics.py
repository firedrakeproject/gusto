"""
Defines microphysics routines to parametrise moist processes for the
compressible Euler equations.
"""

from firedrake import (
    conditional, Function, dx, min_value, max_value, Constant, pi,
    Projector, assemble, split
)
from firedrake.__future__ import interpolate
from firedrake.fml import identity, Term, subject
from gusto.equations import Phases, TracerVariableType
from gusto.recovery import Recoverer, BoundaryMethod
from gusto.equations import CompressibleEulerEquations
from gusto.core.labels import transporting_velocity, transport, prognostic, source_label
from gusto.core.logging import logger
from gusto.equations import thermodynamics
from gusto.physics.physics_parametrisation import PhysicsParametrisation
import ufl
import math
from enum import Enum


__all__ = ["SaturationAdjustment", "Fallout", "Coalescence",
           "EvaporationOfRain", "AdvectedMoments"]


class SaturationAdjustment(PhysicsParametrisation):
    """
    Represents the phase change between water vapour and cloud liquid.

    This class computes updates to water vapour, cloud liquid and (virtual dry)
    potential temperature, representing the action of condensation of water
    vapour and/or evaporation of cloud liquid, with the associated latent heat
    change. The parametrisation follows the saturation adjustment used in Bryan
    and Fritsch (2002).

    Currently this is only implemented for use with mixing ratio variables, and
    with "theta" assumed to be the virtual dry potential temperature. Latent
    heating effects are always assumed, and the total mixing ratio is conserved.
    A filter is applied to prevent generation of negative mixing ratios.
    """

    def __init__(self, equation, vapour_name='water_vapour',
                 cloud_name='cloud_water', latent_heat=True, parameters=None):
        """
        Args:
            equation (:class:`PrognosticEquationSet`): the model's equation.
            vapour_name (str, optional): name of the water vapour variable.
                Defaults to 'water_vapour'.
            cloud_name (str, optional): name of the cloud water variable.
                Defaults to 'cloud_water'.
            latent_heat (bool, optional): whether to have latent heat exchange
                feeding back from the phase change. Defaults to True.
            parameters (:class:`Configuration`, optional): parameters containing
                the values of gas constants. Defaults to None, in which case the
                parameters are obtained from the equation.

        Raises:
            NotImplementedError: currently this is only implemented for the
                CompressibleEulerEquations.
        """

        label_name = 'saturation_adjustment'
        self.explicit_only = True
        super().__init__(equation, label_name, parameters=parameters)

        # TODO: make a check on the variable type of the active tracers
        # if not a mixing ratio, we need to convert to mixing ratios
        # this will be easier if we change equations to have dictionary of
        # active tracer metadata corresponding to variable names

        # Check that fields exist
        if vapour_name not in equation.field_names:
            raise ValueError(f"Field {vapour_name} does not exist in the equation set")
        if cloud_name not in equation.field_names:
            raise ValueError(f"Field {cloud_name} does not exist in the equation set")

        # Make prognostic for physics scheme
        parameters = self.parameters
        self.X = Function(equation.X.function_space())
        self.latent_heat = latent_heat
        W = equation.function_space

        # Vapour and cloud variables are needed for every form of this scheme
        cloud_idx = equation.field_names.index(cloud_name)
        vap_idx = equation.field_names.index(vapour_name)
        cloud_water = self.X.subfunctions[cloud_idx]
        water_vapour = self.X.subfunctions[vap_idx]

        # Indices of variables in mixed function space
        V_idxs = [vap_idx, cloud_idx]
        V = equation.function_space.sub(vap_idx)  # space in which to do the calculation

        # Get variables used to calculate saturation curve
        if isinstance(equation, CompressibleEulerEquations):
            rho_idx = equation.field_names.index('rho')
            theta_idx = equation.field_names.index('theta')
            rho = self.X.subfunctions[rho_idx]
            theta = self.X.subfunctions[theta_idx]
            if latent_heat:
                V_idxs.append(theta_idx)

            # need to evaluate rho at theta-points, and do this via recovery
            boundary_method = BoundaryMethod.extruded if equation.domain.vertical_degree == 0 else None
            rho_averaged = Function(V)
            self.rho_recoverer = Recoverer(rho, rho_averaged, boundary_method=boundary_method)

            exner = thermodynamics.exner_pressure(parameters, rho_averaged, theta)
            T = thermodynamics.T(parameters, theta, exner, r_v=water_vapour)
            p = thermodynamics.p(parameters, exner)

        else:
            raise NotImplementedError(
                'Saturation adjustment only implemented for the Compressible Euler equations')

        # -------------------------------------------------------------------- #
        # Compute heat capacities and calculate saturation curve
        # -------------------------------------------------------------------- #
        # Loop through variables to extract all liquid components
        liquid_water = cloud_water
        for active_tracer in equation.active_tracers:
            if (active_tracer.phase == Phases.liquid
                    and active_tracer.chemical == 'H2O' and active_tracer.name != cloud_name):
                liq_idx = equation.field_names.index(active_tracer.name)
                liquid_water += self.X.subfunctions[liq_idx]

        # define some parameters as attributes
        self.dt = Constant(0.0)
        R_d = parameters.R_d
        cp = parameters.cp
        cv = parameters.cv
        c_pv = parameters.c_pv
        c_pl = parameters.c_pl
        c_vv = parameters.c_vv
        R_v = parameters.R_v

        # make useful fields
        L_v = thermodynamics.Lv(parameters, T)
        R_m = R_d + R_v * water_vapour
        c_pml = cp + c_pv * water_vapour + c_pl * liquid_water
        c_vml = cv + c_vv * water_vapour + c_pl * liquid_water

        # use Teten's formula to calculate the saturation curve
        sat_expr = thermodynamics.r_sat(parameters, T, p)

        # -------------------------------------------------------------------- #
        # Saturation adjustment expression
        # -------------------------------------------------------------------- #
        # make appropriate condensation rate
        sat_adj_expr = (water_vapour - sat_expr) / self.dt
        if latent_heat:
            # As condensation/evaporation happens, the temperature changes
            # so need to take this into account with an extra factor
            sat_adj_expr = sat_adj_expr / (1.0 + ((L_v ** 2.0 * sat_expr)
                                                  / (cp * R_v * T ** 2.0)))

        # adjust the rate so that so negative values don't occur
        sat_adj_expr = conditional(sat_adj_expr < 0,
                                   max_value(sat_adj_expr, -cloud_water / self.dt),
                                   min_value(sat_adj_expr, water_vapour / self.dt))

        # -------------------------------------------------------------------- #
        # Factors for multiplying source for different variables
        # -------------------------------------------------------------------- #
        # Factors need to have same shape as V_idxs
        factors = [Constant(1.0), Constant(-1.0)]
        if latent_heat and isinstance(equation, CompressibleEulerEquations):
            factors.append(-theta * (cv * L_v / (c_vml * cp * T) - R_v * cv * c_pml / (R_m * cp * c_vml)))

        # -------------------------------------------------------------------- #
        # Add terms to equations and make interpolators
        # -------------------------------------------------------------------- #
        self.source = Function(W)
        self.source_expr = [split(self.source)[V_idx] for V_idx in V_idxs]
        self.source_int = [self.source.subfunctions[V_idx] for V_idx in V_idxs]
        self.source_interpolate = [interpolate(sat_adj_expr*factor, source)
                                   for source, factor in zip(self.source_int, factors)]

        tests = [equation.tests[idx] for idx in V_idxs]

        # Add source terms to residual
        for test, source in zip(tests, self.source_expr):
            equation.residual += source_label(self.label(subject(test * source * dx,
                                                                 equation.X), self.evaluate))

    def evaluate(self, x_in, dt, x_out=None):
        """
        Evaluates the source/sink for the saturation adjustment process.

        Args:
            x_in (:class:`Function`): the (mixed) field to be evolved.
            dt (:class:`Constant`): the time interval for the scheme.
            x_out: (:class:`Function`, optional): the (mixed) source
                                                  field to be outputed.
        """
        logger.info(f'Evaluating physics parametrisation {self.label.label}')
        # Update the values of internal variables
        self.dt.assign(dt)
        self.X.assign(x_in)
        if isinstance(self.equation, CompressibleEulerEquations):
            self.rho_recoverer.project()
        # Evaluate the source
        for interpolator, src in zip(self.source_interpolate, self.source_int):
            src.assign(assemble(interpolator))

        # If a source output is provided, assign the source term to it
        if x_out is not None:
            x_out.assign(self.source)


class AdvectedMoments(Enum):
    """
    Enumerator describing the moments in the raindrop size distribution.

    This stores enumerators for the number of moments used to describe the
    size distribution curve of raindrops. This can be used for deciding which
    moments to advect in a precipitation scheme.
    """

    M0 = 0  # don't advect the distribution -- advect all rain at the same speed
    M3 = 1  # advect the mass of the distribution


class Fallout(PhysicsParametrisation):
    """
    Represents the fallout process of hydrometeors.

    Precipitation is described by downwards transport of tracer species. This
    process determines the precipitation velocity from the `AdvectedMoments`
    enumerator, which either:
    (a) sets a terminal velocity of 5 m/s
    (b) determines a rainfall size distribution based on a Gamma distribution,
    as in Paluch (1979). The droplets are based on the mean mass of the rain
    droplets (aka a single-moment scheme).

    Outflow boundary conditions are applied to the transport, so the rain will
    flow out of the bottom of the domain.

    This is currently only implemented for "rain" in the compressible Euler
    equation set. This variable must be a mixing ratio, It is only implemented
    for Cartesian geometry.
    """

    def __init__(self, equation, rain_name, domain, transport_method,
                 moments=AdvectedMoments.M3):
        """
        Args:
            equation (:class:`PrognosticEquationSet`): the model's equation.
            rain_name (str, optional): name of the rain variable. Defaults to
                'rain'.
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            transport_method (:class:`TransportMethod`): the spatial method
                used for transporting the rain.
            moments (int, optional): an :class:`AdvectedMoments` enumerator,
                representing the number of moments of the size distribution of
                raindrops to be transported. Defaults to `AdvectedMoments.M3`.
        """

        label_name = f'fallout_{rain_name}'
        super().__init__(equation, label_name, parameters=None)

        # Check that fields exist
        if rain_name not in equation.field_names:
            raise ValueError(f"Field {rain_name} does not exist in the equation set")

        # Check if variable is a mixing ratio
        rain_tracer = equation.get_active_tracer(rain_name)
        if rain_tracer.variable_type != TracerVariableType.mixing_ratio:
            raise NotImplementedError('Fallout only implemented when rain '
                                      + 'variable is a mixing ratio')

        # Set up rain and velocity
        self.X = Function(equation.X.function_space())

        rain_idx = equation.field_names.index(rain_name)
        rain = self.X.subfunctions[rain_idx]

        Vu = domain.spaces("HDiv")
        # TODO: there must be a better way than forcing this into the equation
        v = equation.prescribed_fields(name='rainfall_velocity', space=Vu)

        # -------------------------------------------------------------------- #
        # Create physics term -- which is actually a transport term
        # -------------------------------------------------------------------- #

        assert transport_method.outflow, \
            'Fallout requires a transport method with outflow=True'
        adv_term = transport_method.form
        # Add rainfall velocity by replacing transport_velocity in term
        adv_term = adv_term.label_map(identity,
                                      map_if_true=lambda t: Term(
                                          ufl.replace(t.form, {t.get(transporting_velocity): v}),
                                          t.labels))

        # We don't want this term to be picked up by normal transport, so drop
        # the transport label
        adv_term = transport.remove(adv_term)

        adv_term = prognostic(subject(adv_term, equation.X), rain_name)
        equation.residual += self.label(adv_term, self.evaluate)

        # -------------------------------------------------------------------- #
        # Expressions for determining rainfall velocity
        # -------------------------------------------------------------------- #
        self.moments = moments

        if moments == AdvectedMoments.M0:
            # all rain falls at terminal velocity
            terminal_velocity = Constant(5)  # in m/s
            v.project(-terminal_velocity*domain.k)
        elif moments == AdvectedMoments.M3:
            self.explicit_only = True
            # this advects the third moment M3 of the raindrop
            # distribution, which corresponds to the mean mass
            rho_idx = equation.field_names.index('rho')
            rho = self.X.subfunctions[rho_idx]
            rho_w = Constant(1000.0)  # density of liquid water
            # assume n(D) = n_0 * D^mu * exp(-Lambda*D)
            # n_0 = N_r * Lambda^(1+mu) / gamma(1 + mu)
            N_r = Constant(10**5)  # number of rain droplets per m^3
            mu = 0.0  # shape constant of droplet gamma distribution
            # assume V(D) = a * D^b * exp(-f*D) * (rho_0 / rho)^g
            # take f = 0
            a = Constant(362.)  # intercept for velocity distr. in log space
            b = 0.65  # inverse scale parameter for velocity distr.
            rho0 = Constant(1.22)  # reference density in kg/m^3
            g = Constant(0.5)  # scaling of density correction
            # we keep mu in the expressions even though mu = 0
            threshold = Constant(10**-10)  # only do rainfall for r > threshold
            Lambda = (N_r * pi * rho_w * math.gamma(4 + mu)
                      / (6 * math.gamma(1 + mu) * rho * rain)) ** (1. / 3)
            Lambda0 = (N_r * pi * rho_w * math.gamma(4 + mu)
                       / (6 * math.gamma(1 + mu) * rho * threshold)) ** (1. / 3)
            v_expression = conditional(rain > threshold,
                                       (a * math.gamma(4 + b + mu)
                                        / (math.gamma(4 + mu) * Lambda ** b)
                                        * (rho0 / rho) ** g),
                                       (a * math.gamma(4 + b + mu)
                                        / (math.gamma(4 + mu) * Lambda0 ** b)
                                        * (rho0 / rho) ** g))
        else:
            raise NotImplementedError(
                'Currently there are only implementations for zero and one '
                + 'moment schemes for rainfall. Valid options are '
                + 'AdvectedMoments.M0 and AdvectedMoments.M3')

        if moments != AdvectedMoments.M0:
            project_params = {
                'quadrature_degree': equation.domain.max_quad_degree
            }
            self.determine_v = Projector(
                -v_expression*domain.k, v,
                form_compiler_parameters=project_params
            )

    def evaluate(self, x_in, dt, x_out=None):
        """
        Evaluates the source/sink corresponding to the fallout process.

        Args:
            x_in (:class:`Function`): the (mixed) field to be evolved.
            dt (:class:`Constant`): the time interval for the scheme.
            x_out: (:class:`Function`, optional): the (mixed) source
                                                  field to be outputed.
        """
        logger.info(f'Evaluating physics parametrisation {self.label.label}')
        self.X.assign(x_in)
        if self.moments != AdvectedMoments.M0:
            self.determine_v.project()

        if x_out is not None:
            raise NotImplementedError("Fallout does not output a source term, "
                                      "or a non-interpolated/projected expression and hence "
                                      "cannot be used in a nonsplit physics formulation.")


class Coalescence(PhysicsParametrisation):
    """
    Represents the coalescence of cloud droplets to form rain droplets.

    Coalescence is the process of forming rain droplets from cloud droplets.
    This scheme performs that process, using two parts: accretion, which is
    independent of the rain concentration, and auto-accumulation, which is
    accelerated by the existence of rain. These parametrisations come from Klemp
    and Wilhelmson (1978). The rate of change is limited to prevent production
    of negative moisture values.

    This is only implemented for mixing ratio variables.
    """

    def __init__(self, equation, cloud_name='cloud_water', rain_name='rain',
                 accretion=True, accumulation=True):
        """
        Args:
            equation (:class:`PrognosticEquationSet`): the model's equation.
            cloud_name (str, optional): name of the cloud variable. Defaults to
                'cloud_water'.
            rain_name (str, optional): name of the rain variable. Defaults to
                'rain'.
            accretion (bool, optional): whether to include the accretion process
                in the parametrisation. Defaults to True.
            accumulation (bool, optional): whether to include the accumulation
                process in the parametrisation. Defaults to True.
        """

        self.explicit_only = True
        label_name = "coalescence"
        if accretion:
            label_name += "_accretion"
        if accumulation:
            label_name += "_accumulation"
        super().__init__(equation, label_name, parameters=None)

        # Check that fields exist
        if cloud_name not in equation.field_names:
            raise ValueError(f"Field {cloud_name} does not exist in the equation set")
        if rain_name not in equation.field_names:
            raise ValueError(f"Field {rain_name} does not exist in the equation set")

        self.cloud_idx = equation.field_names.index(cloud_name)
        self.rain_idx = equation.field_names.index(rain_name)
        Vcl = equation.function_space.sub(self.cloud_idx)
        Vr = equation.function_space.sub(self.rain_idx)
        W = equation.function_space
        self.cloud_water = Function(Vcl)
        self.rain = Function(Vr)

        # declare function space and source field
        self.source = Function(W)
        self.source_expr = split(self.source)[self.cloud_idx]
        self.source_int = self.source.subfunctions[self.cloud_idx]

        # define some parameters as attributes
        self.dt = Constant(0.0)
        # TODO: should these parameters be hard-coded or configurable?
        k_1 = Constant(0.001)  # accretion rate in 1/s
        k_2 = Constant(2.2)  # accumulation rate in 1/s
        a = Constant(0.001)  # min cloud conc in kg/kg
        b = Constant(0.875)  # power for rain in accumulation

        # make default rates to be zero
        accr_rate = Constant(0.0)
        accu_rate = Constant(0.0)

        if accretion:
            accr_rate = k_1 * (self.cloud_water - a)
        if accumulation:
            accu_rate = k_2 * self.cloud_water * self.rain ** b

        # Expression for rain increment, with conditionals to prevent negative values
        rain_expr = conditional(self.rain < 0.0,  # if rain is negative do only accretion
                                conditional(accr_rate < 0.0,
                                            0.0,
                                            min_value(accr_rate, self.cloud_water / self.dt)),
                                # don't turn rain back into cloud
                                conditional(accr_rate + accu_rate < 0.0,
                                            0.0,
                                            # if accretion rate is negative do only accumulation
                                            conditional(accr_rate < 0.0,
                                                        min_value(accu_rate, self.cloud_water / self.dt),
                                                        min_value(accr_rate + accu_rate, self.cloud_water / self.dt))))

        self.source_interpolate = interpolate(rain_expr, self.source_int)

        # Add term to equation's residual
        test_cl = equation.tests[self.cloud_idx]
        test_r = equation.tests[self.rain_idx]
        equation.residual += source_label(
            self.label(
                subject(
                    test_cl * self.source_expr * dx - test_r * self.source_expr * dx,
                    self.source
                ),
                self.evaluate
            )
        )

    def evaluate(self, x_in, dt, x_out=None):
        """
        Evaluates the source/sink for the coalescence process.

        Args:
            x_in (:class:`Function`): the (mixed) field to be evolved.
            dt (:class:`Constant`): the time interval for the scheme.
            x_out: (:class:`Function`, optional): the (mixed) source
                                                  field to be outputed.
        """
        logger.info(f'Evaluating physics parametrisation {self.label.label}')
        # Update the values of internal variables
        self.dt.assign(dt)
        self.rain.assign(x_in.subfunctions[self.rain_idx])
        self.cloud_water.assign(x_in.subfunctions[self.cloud_idx])

        self.source_int.assign(assemble(self.source_interpolate))
        # If a source output is provided, assign the source term to it
        if x_out is not None:
            x_out.assign(self.source)


class EvaporationOfRain(PhysicsParametrisation):
    """
    Represents the evaporation of rain into water vapour.

    This describes the evaporation of rain into water vapour, with the
    associated latent heat change. This parametrisation comes from Klemp and
    Wilhelmson (1978). The rate of change is limited to prevent production of
    negative moisture values.

    This is only implemented for mixing ratio variables, and when the prognostic
    is the virtual dry potential temperature.
    """

    def __init__(self, equation, rain_name='rain', vapour_name='water_vapour',
                 latent_heat=True):
        """
        Args:
            equation (:class:`PrognosticEquationSet`): the model's equation.
            cloud_name (str, optional): name of the rain variable. Defaults to
                'rain'.
            vapour_name (str, optional): name of the water vapour variable.
                Defaults to 'water_vapour'.
            latent_heat (bool, optional): whether to have latent heat exchange
                feeding back from the phase change. Defaults to True.

        Raises:
            NotImplementedError: currently this is only implemented for the
                CompressibleEulerEquations.
        """

        self.explicit_only = True
        label_name = 'evaporation_of_rain'
        super().__init__(equation, label_name, parameters=None)

        # TODO: make a check on the variable type of the active tracers
        # if not a mixing ratio, we need to convert to mixing ratios
        # this will be easier if we change equations to have dictionary of
        # active tracer metadata corresponding to variable names

        # Check that fields exist
        if vapour_name not in equation.field_names:
            raise ValueError(f"Field {vapour_name} does not exist in the equation set")
        if rain_name not in equation.field_names:
            raise ValueError(f"Field {rain_name} does not exist in the equation set")

        # Make prognostic for physics scheme
        self.X = Function(equation.X.function_space())
        parameters = self.parameters
        self.latent_heat = latent_heat

        # Vapour and cloud variables are needed for every form of this scheme
        rain_idx = equation.field_names.index(rain_name)
        vap_idx = equation.field_names.index(vapour_name)
        rain = self.X.subfunctions[rain_idx]
        water_vapour = self.X.subfunctions[vap_idx]

        # Indices of variables in mixed function space
        V_idxs = [rain_idx, vap_idx]
        V = equation.function_space.sub(rain_idx)  # space in which to do the calculation
        W = equation.function_space

        # Get variables used to calculate saturation curve
        if isinstance(equation, CompressibleEulerEquations):
            rho_idx = equation.field_names.index('rho')
            theta_idx = equation.field_names.index('theta')
            rho = self.X.subfunctions[rho_idx]
            theta = self.X.subfunctions[theta_idx]
            if latent_heat:
                V_idxs.append(theta_idx)

            # need to evaluate rho at theta-points, and do this via recovery
            boundary_method = BoundaryMethod.extruded if equation.domain.vertical_degree == 0 else None
            rho_averaged = Function(V)
            self.rho_recoverer = Recoverer(rho, rho_averaged, boundary_method=boundary_method)

            exner = thermodynamics.exner_pressure(parameters, rho_averaged, theta)
            T = thermodynamics.T(parameters, theta, exner, r_v=water_vapour)
            p = thermodynamics.p(parameters, exner)

        # -------------------------------------------------------------------- #
        # Compute heat capacities and calculate saturation curve
        # -------------------------------------------------------------------- #
        # Loop through variables to extract all liquid components
        liquid_water = rain
        for active_tracer in equation.active_tracers:
            if (active_tracer.phase == Phases.liquid
                    and active_tracer.chemical == 'H2O' and active_tracer.name != rain_name):
                liq_idx = equation.field_names.index(active_tracer.name)
                liquid_water += self.X.subfunctions[liq_idx]

        # define some parameters as attributes
        self.dt = Constant(0.0)
        R_d = parameters.R_d
        cp = parameters.cp
        cv = parameters.cv
        c_pv = parameters.c_pv
        c_pl = parameters.c_pl
        c_vv = parameters.c_vv
        R_v = parameters.R_v

        # make useful fields
        L_v = thermodynamics.Lv(parameters, T)
        R_m = R_d + R_v * water_vapour
        c_pml = cp + c_pv * water_vapour + c_pl * liquid_water
        c_vml = cv + c_vv * water_vapour + c_pl * liquid_water

        # use Teten's formula to calculate the saturation curve
        sat_expr = thermodynamics.r_sat(parameters, T, p)

        # -------------------------------------------------------------------- #
        # Evaporation expression
        # -------------------------------------------------------------------- #
        # TODO: should these parameters be hard-coded or configurable?
        # expression for ventilation factor
        a = Constant(1.6)
        b = Constant(124.9)
        c = Constant(0.2046)
        C = a + b * (rho_averaged * rain) ** c

        # make appropriate condensation rate
        f = Constant(5.4e5)
        g = Constant(2.55e6)
        h = Constant(0.525)
        evap_rate = (((1 - water_vapour / sat_expr) * C * (rho_averaged * rain) ** h)
                     / (rho_averaged * (f + g / (p * sat_expr))))

        # adjust evap rate so negative rain doesn't occur
        evap_rate = conditional(evap_rate < 0, 0.0,
                                conditional(rain < 0.0, 0.0,
                                            min_value(evap_rate, rain / self.dt)))

        # -------------------------------------------------------------------- #
        # Factors for multiplying source for different variables
        # -------------------------------------------------------------------- #
        # Factors need to have same shape as V_idxs
        factors = [Constant(-1.0), Constant(1.0)]
        if latent_heat and isinstance(equation, CompressibleEulerEquations):
            factors.append(-theta * (cv * L_v / (c_vml * cp * T) - R_v * cv * c_pml / (R_m * cp * c_vml)))

        # -------------------------------------------------------------------- #
        # Add terms to equations and make interpolators
        # -------------------------------------------------------------------- #
        self.source = Function(W)
        self.source_expr = [split(self.source)[V_idx] for V_idx in V_idxs]
        self.source_int = [self.source.subfunctions[V_idx] for V_idx in V_idxs]
        self.source_interpolate = [
            interpolate(evap_rate * factor, source)
            for source, factor in zip(self.source_int, factors)
        ]

        tests = [equation.tests[idx] for idx in V_idxs]

        # Add source terms to residual
        for test, source in zip(tests, self.source_expr):
            equation.residual += source_label(self.label(subject(test * source * dx,
                                                                 self.source), self.evaluate))

    def evaluate(self, x_in, dt, x_out=None):
        """
        Applies the process to evaporate rain droplets.

        Args:
            x_in (:class:`Function`): the (mixed) field to be evolved.
            dt (:class:`Constant`): the time interval for the scheme.
            x_out: (:class:`Function`, optional): the (mixed) source
                                                  field to be outputed.
        """
        logger.info(f'Evaluating physics parametrisation {self.label.label}')
        # Update the values of internal variables
        self.dt.assign(dt)
        self.X.assign(x_in)
        if isinstance(self.equation, CompressibleEulerEquations):
            self.rho_recoverer.project()

        for interpolator, src in zip(self.source_interpolate, self.source_int):
            src.assign(assemble(interpolator))

        # If a source output is provided, assign the source term to it
        if x_out is not None:
            x_out.assign(self.source)
