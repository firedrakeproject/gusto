"""
Objects to perform parametrisations of physical processes, or "physics".

"PhysicsParametrisation" schemes are routines to compute updates to prognostic fields that
represent the action of non-fluid processes, or those fluid processes that are
unresolved. This module contains a set of these processes in the form of classes
with "evaluate" methods.
"""

from abc import ABCMeta, abstractmethod
from gusto.active_tracers import Phases, TracerVariableType
from gusto.configuration import BoundaryLayerParameters
from gusto.recovery import Recoverer, BoundaryMethod
from gusto.equations import CompressibleEulerEquations
from gusto.fml import identity, Term, subject
from gusto.labels import PhysicsLabel, transporting_velocity, transport, prognostic
from gusto.logging import logger
from firedrake import (Interpolator, conditional, Function, dx, sqrt, dot,
                       min_value, max_value, Constant, pi, Projector, grad,
                       TestFunctions, split, inner, TestFunction, exp, avg,
                       outer, FacetNormal, SpatialCoordinate, dS_v,
                       NonlinearVariationalProblem, NonlinearVariationalSolver)
from gusto import thermodynamics
import ufl
import math
from enum import Enum
from types import FunctionType


__all__ = ["SaturationAdjustment", "Fallout", "Coalescence", "EvaporationOfRain",
           "AdvectedMoments", "InstantRain", "SWSaturationAdjustment",
           "SourceSink", "SurfaceFluxes", "WindDrag", "StaticAdjustment",
           "SuppressVerticalWind", "BoundaryLayerMixing"]


class PhysicsParametrisation(object, metaclass=ABCMeta):
    """Base class for the parametrisation of physical processes for Gusto."""

    def __init__(self, equation, label_name, parameters=None):
        """
        Args:
            equation (:class:`PrognosticEquationSet`): the model's equation.
            label_name (str): name of physics scheme, to be passed to its label.
            parameters (:class:`Configuration`, optional): parameters containing
                the values of gas constants. Defaults to None, in which case the
                parameters are obtained from the equation.
        """

        self.label = PhysicsLabel(label_name)
        self.equation = equation
        if parameters is None and hasattr(equation, 'parameters'):
            self.parameters = equation.parameters
        else:
            self.parameters = parameters

    @abstractmethod
    def evaluate(self):
        """
        Computes the value of physics source and sink terms.
        """
        pass


class SourceSink(PhysicsParametrisation):
    """
    The source or sink of some variable, described through a UFL expression.

    A scheme representing the general source or sink of a variable, described
    through a UFL expression. The expression should be for the rate of change
    of the variable. It is intended that the source/sink is independent of the
    prognostic variables.

    The expression can also be a time-varying expression. In which case a
    function should be provided, taking a :class:`Constant` as an argument (to
    represent the time.)
    """

    def __init__(self, equation, variable_name, rate_expression,
                 time_varying=False, method='interpolate'):
        """
        Args:
            equation (:class:`PrognosticEquationSet`): the model's equation.
            variable_name (str): the name of the variable
            rate_expression (:class:`ufl.Expr` or func): an expression giving
                the rate of change of the variable. If a time-varying expression
                is needed, this should be a function taking a single argument
                representing the time. Then the `time_varying` argument must
                be set to True.
            time_varying (bool, optional): whether the source/sink expression
                varies with time. Defaults to False.
            method (str, optional): the method to use to evaluate the expression
                for the source. Valid options are 'interpolate' or 'project'.
                Default is 'interpolate'.
        """

        label_name = f'source_sink_{variable_name}'
        super().__init__(equation, label_name, parameters=None)

        if method not in ['interpolate', 'project']:
            raise ValueError(f'Method {method} for source/sink evaluation not valid')
        self.method = method
        self.time_varying = time_varying
        self.variable_name = variable_name

        # Check the variable exists
        if hasattr(equation, "field_names"):
            assert variable_name in equation.field_names, \
                f'Field {variable_name} does not exist in the equation set'
        else:
            assert variable_name == equation.field_name, \
                f'Field {variable_name} does not exist in the equation'

        # Work out the appropriate function space
        if hasattr(equation, "field_names"):
            V_idx = equation.field_names.index(variable_name)
            W = equation.function_space
            V = W.sub(V_idx)
            test = equation.tests[V_idx]
        else:
            V = equation.function_space
            test = equation.test

        # Make source/sink term
        self.source = Function(V)
        equation.residual += self.label(subject(test * self.source * dx, equation.X),
                                        self.evaluate)

        # Handle whether the expression is time-varying or not
        if self.time_varying:
            expression = rate_expression(equation.domain.t)
        else:
            expression = rate_expression

        # Handle method of evaluating source/sink
        if self.method == 'interpolate':
            self.source_interpolator = Interpolator(expression, V)
        else:
            self.source_projector = Projector(expression, V)

        # If not time-varying, evaluate for the first time here
        if not self.time_varying:
            if self.method == 'interpolate':
                self.source.assign(self.source_interpolator.interpolate())
            else:
                self.source.assign(self.source_projector.project())

    def evaluate(self, x_in, dt):
        """
        Evalutes the source term generated by the physics.

        Args:
            x_in: (:class:`Function`): the (mixed) field to be evolved. Unused.
            dt: (:class:`Constant`): the timestep, which can be the time
                interval for the scheme. Unused.
        """
        if self.time_varying:
            logger.info(f'Evaluating physics parametrisation {self.label.label}')
            if self.method == 'interpolate':
                self.source.assign(self.source_interpolator.interpolate())
            else:
                self.source.assign(self.source_projector.project())
        else:
            pass

class Relaxation(PhysicsParametrisation):
    """
    Relaxation term
    """

    def __init__(self, equation, variable_name, relaxation_expression,
                 time_varying=None, method='interpolate'):
        """
        Args:
            equation (:class:`PrognosticEquationSet`): the model's equation.
            variable_name (str): the name of the variable
            rate_expression (:class:`ufl.Expr` or func): an expression giving
                the rate of change of the variable. If a time-varying expression
                is needed, this should be a function taking a single argument
                representing the time. Then the `time_varying` argument must
                be set to True.
            time_varying (bool, optional): whether the source/sink expression
                varies with time. Defaults to False.
            method (str, optional): the method to use to evaluate the expression
                for the source. Valid options are 'interpolate' or 'project'.
                Default is 'interpolate'.
        """

        label_name = f'Relaxation_{variable_name}'
        super().__init__(equation, label_name, parameters=None)

        if method not in ['interpolate', 'project']:
            raise ValueError(f'Method {method} for source/sink evaluation not valid')
        self.method = method
        self.time_varying = time_varying
        self.variable_name = variable_name

        # Check the variable exists
        if hasattr(equation, "field_names"):
            assert variable_name in equation.field_names, \
                f'Field {variable_name} does not exist in the equation set'
        else:
            assert variable_name == equation.field_name, \
                f'Field {variable_name} does not exist in the equation'

        # Work out the appropriate function space
        if hasattr(equation, "field_names"):
            V_idx = equation.field_names.index(variable_name)
            W = equation.function_space
            V = W.sub(V_idx)
            test = equation.tests[V_idx]
        else:
            test = equation.test

        # Add relaxation term to residual
        equation.residual += self.label(subject(test * relaxation_expression * dx, equation.X),
                                        self.evaluate)
        
        expression = relaxation_expression
        # Handle method of evaluating source/sink
        if self.method == 'interpolate':
            self.source_interpolator = Interpolator(expression, V)
        else:
            self.source_projector = Projector(expression, V)

        # If not time-varying, evaluate for the first time here
        if not self.time_varying:
            if self.method == 'interpolate':
                self.source.assign(self.source_interpolator.interpolate())
            else:
                self.source.assign(self.source_projector.project())

    def evaluate(self, x_in, dt):
        """
        Evalutes the source term generated by the physics.

        Args:
            x_in: (:class:`Function`): the (mixed) field to be evolved. Unused.
            dt: (:class:`Constant`): the timestep, which can be the time
                interval for the scheme. Unused.
        """
        pass

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
        self.source = [Function(V) for factor in factors]
        self.source_interpolators = [Interpolator(sat_adj_expr*factor, source)
                                     for factor, source in zip(factors, self.source)]

        tests = [equation.tests[idx] for idx in V_idxs]

        # Add source terms to residual
        for test, source in zip(tests, self.source):
            equation.residual += self.label(subject(test * source * dx,
                                                    equation.X), self.evaluate)

    def evaluate(self, x_in, dt):
        """
        Evaluates the source/sink for the saturation adjustment process.

        Args:
            x_in (:class:`Function`): the (mixed) field to be evolved.
            dt (:class:`Constant`): the time interval for the scheme.
        """
        logger.info(f'Evaluating physics parametrisation {self.label.label}')
        # Update the values of internal variables
        self.dt.assign(dt)
        self.X.assign(x_in)
        if isinstance(self.equation, CompressibleEulerEquations):
            self.rho_recoverer.project()
        # Evaluate the source
        for interpolator in self.source_interpolators:
            interpolator.interpolate()


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
            # TODO: introduce reduced projector
            test = TestFunction(Vu)
            dx_reduced = dx(degree=4)
            proj_eqn = inner(test, v + v_expression*domain.k)*dx_reduced
            proj_prob = NonlinearVariationalProblem(proj_eqn, v)
            self.determine_v = NonlinearVariationalSolver(proj_prob)

    def evaluate(self, x_in, dt):
        """
        Evaluates the source/sink corresponding to the fallout process.

        Args:
            x_in (:class:`Function`): the (mixed) field to be evolved.
            dt (:class:`Constant`): the time interval for the scheme.
        """
        logger.info(f'Evaluating physics parametrisation {self.label.label}')
        self.X.assign(x_in)
        if self.moments != AdvectedMoments.M0:
            self.determine_v.solve()


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
        self.cloud_water = Function(Vcl)
        self.rain = Function(Vr)

        # declare function space and source field
        Vt = self.cloud_water.function_space()
        self.source = Function(Vt)

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

        self.source_interpolator = Interpolator(rain_expr, self.source)

        # Add term to equation's residual
        test_cl = equation.tests[self.cloud_idx]
        test_r = equation.tests[self.rain_idx]
        equation.residual += self.label(subject(test_cl * self.source * dx
                                                - test_r * self.source * dx,
                                                equation.X),
                                        self.evaluate)

    def evaluate(self, x_in, dt):
        """
        Evaluates the source/sink for the coalescence process.

        Args:
            x_in (:class:`Function`): the (mixed) field to be evolved.
            dt (:class:`Constant`): the time interval for the scheme.
        """
        logger.info(f'Evaluating physics parametrisation {self.label.label}')
        # Update the values of internal variables
        self.dt.assign(dt)
        self.rain.assign(x_in.subfunctions[self.rain_idx])
        self.cloud_water.assign(x_in.subfunctions[self.cloud_idx])
        # Evaluate the source
        self.source.assign(self.source_interpolator.interpolate())


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
        self.source = [Function(V) for factor in factors]
        self.source_interpolators = [Interpolator(evap_rate*factor, source)
                                     for factor, source in zip(factors, self.source)]

        tests = [equation.tests[idx] for idx in V_idxs]

        # Add source terms to residual
        for test, source in zip(tests, self.source):
            equation.residual += self.label(subject(test * source * dx,
                                                    equation.X), self.evaluate)

    def evaluate(self, x_in, dt):
        """
        Applies the process to evaporate rain droplets.

        Args:
            x_in (:class:`Function`): the (mixed) field to be evolved.
            dt (:class:`Constant`): the time interval for the scheme.
        """
        logger.info(f'Evaluating physics parametrisation {self.label.label}')
        # Update the values of internal variables
        self.dt.assign(dt)
        self.X.assign(x_in)
        if isinstance(self.equation, CompressibleEulerEquations):
            self.rho_recoverer.project()
        # Evaluate the source
        for interpolator in self.source_interpolators:
            interpolator.interpolate()


class InstantRain(PhysicsParametrisation):
    """
    The process of converting vapour above the saturation curve to rain.

    A scheme to move vapour directly to rain. If convective feedback is true
    then this process feeds back directly on the height equation. If rain is
    accumulating then excess vapour is being tracked and stored as rain;
    otherwise converted vapour is not recorded. The process can happen over the
    timestep dt or over a specified time interval tau.
     """

    def __init__(self, equation, saturation_curve,
                 time_varying_saturation=False,
                 vapour_name="water_vapour", rain_name=None, gamma_r=1,
                 convective_feedback=False, beta1=None, tau=None,
                 parameters=None):
        """
        Args:
            equation (:class:`PrognosticEquationSet`): the model's equation.
            saturation_curve (:class:`ufl.Expr` or func): the curve above which
                excess moisture is converted to rain. Is either prescribed or
                dependent on a prognostic field.
            time_varying_saturation (bool, optional): set this to True if the
                saturation curve is changing in time. Defaults to False.
            vapour_name (str, optional): name of the water vapour variable.
                Defaults to "water_vapour".
            rain_name (str, optional): name of the rain variable. Defaults to
                None.
            gamma_r (float, optional): Fraction of vapour above the threshold
                which is converted to rain. Defaults to one, in which case all
                vapour above the threshold is converted.
            convective_feedback (bool, optional): True if the conversion of
                vapour affects the height equation. Defaults to False.
            beta1 (float, optional): Condensation proportionality constant,
                used if convection causes a response in the height equation.
                Defaults to None, but must be specified if convective_feedback
                is True.
            tau (float, optional): Timescale for condensation. Defaults to None,
                in which case the timestep dt is used.
            parameters (:class:`Configuration`, optional): parameters containing
                the values of gas constants. Defaults to None, in which case the
                parameters are obtained from the equation.
        """

        self.explicit_only = True
        label_name = 'instant_rain'
        super().__init__(equation, label_name, parameters=parameters)

        self.convective_feedback = convective_feedback
        self.time_varying_saturation = time_varying_saturation

        # check for the correct fields
        assert vapour_name in equation.field_names, f"Field {vapour_name} does not exist in the equation set"
        self.Vv_idx = equation.field_names.index(vapour_name)

        if rain_name is not None:
            assert rain_name in equation.field_names, f"Field {rain_name} does not exist in the equation set "

        if self.convective_feedback:
            assert "D" in equation.field_names, "Depth field must exist for convective feedback"
            assert beta1 is not None, "If convective feedback is used, beta1 parameter must be specified"

        # obtain function space and functions; vapour needed for all cases
        W = equation.function_space
        Vv = W.sub(self.Vv_idx)
        test_v = equation.tests[self.Vv_idx]

        # depth needed if convective feedback
        if self.convective_feedback:
            self.VD_idx = equation.field_names.index("D")
            VD = W.sub(self.VD_idx)
            test_D = equation.tests[self.VD_idx]
            self.D = Function(VD)

        # the source function is the difference between the water vapour and
        # the saturation function
        self.water_v = Function(Vv)
        self.source = Function(Vv)

        # tau is the timescale for conversion (may or may not be the timestep)
        if tau is not None:
            self.set_tau_to_dt = False
            self.tau = tau
        else:
            self.set_tau_to_dt = True
            self.tau = Constant(0)
            logger.info("Timescale for rain conversion has been set to dt. If this is not the intention then provide a tau parameter as an argument to InstantRain.")

        if self.time_varying_saturation:
            if isinstance(saturation_curve, FunctionType):
                self.saturation_computation = saturation_curve
                self.saturation_curve = Function(Vv)
            else:
                raise NotImplementedError(
                    "If time_varying_saturation is True then saturation must be a Python function of a prognostic field.")
        else:
            assert not isinstance(saturation_curve, FunctionType), "If time_varying_saturation is not True then saturation cannot be a Python function."
            self.saturation_curve = saturation_curve

        # lose proportion of vapour above the saturation curve
        equation.residual += self.label(subject(test_v * self.source * dx,
                                                equation.X),
                                        self.evaluate)

        # if rain is not none then the excess vapour is being tracked and is
        # added to rain
        if rain_name is not None:
            Vr_idx = equation.field_names.index(rain_name)
            test_r = equation.tests[Vr_idx]
            equation.residual -= self.label(subject(test_r * self.source * dx,
                                                    equation.X),
                                            self.evaluate)

        # if feeding back on the height adjust the height equation
        if convective_feedback:
            equation.residual += self.label(subject(test_D * beta1 * self.source * dx,
                                                    equation.X),
                                            self.evaluate)

        # interpolator does the conversion of vapour to rain
        self.source_interpolator = Interpolator(conditional(
            self.water_v > self.saturation_curve,
            (1/self.tau)*gamma_r*(self.water_v - self.saturation_curve),
            0), Vv)

    def evaluate(self, x_in, dt):
        """
        Evalutes the source term generated by the physics.

        Computes the physics contributions (loss of vapour, accumulation of
        rain and loss of height due to convection) at each timestep.

        Args:
            x_in: (:class: 'Function'): the (mixed) field to be evolved.
            dt: (:class: 'Constant'): the timestep, which can be the time
                interval for the scheme.
        """
        logger.info(f'Evaluating physics parametrisation {self.label.label}')
        if self.convective_feedback:
            self.D.assign(x_in.subfunctions[self.VD_idx])
        if self.time_varying_saturation:
            self.saturation_curve.interpolate(self.saturation_computation(x_in))
        if self.set_tau_to_dt:
            self.tau.assign(dt)
        self.water_v.assign(x_in.subfunctions[self.Vv_idx])
        self.source.assign(self.source_interpolator.interpolate())


class SWSaturationAdjustment(PhysicsParametrisation):
    """
    Represents the process of adjusting water vapour and cloud water according
    to a saturation function, via condensation and evaporation processes.

    This physics scheme follows that of Zerroukat and Allen (2015).

    """

    def __init__(self, equation, saturation_curve, L_v=None,
                 time_varying_saturation=False, vapour_name='water_vapour',
                 cloud_name='cloud_water', convective_feedback=False,
                 beta1=None, thermal_feedback=False, beta2=None, gamma_v=1,
                 time_varying_gamma_v=False, tau=None,
                 parameters=None):
        """
        Args:
            equation (:class:`PrognosticEquationSet`): the model's equation
            saturation_curve (:class:`ufl.Expr` or func): the curve which
                dictates when phase changes occur. In a saturated atmosphere
                vapour above the saturation curve becomes cloud, and if the
                atmosphere is sub-saturated and there is cloud present cloud
                will become vapour until the saturation curve is reached. The
                saturation curve is either prescribed or dependent on a
                prognostic field.
            time_varying_saturation (bool, optional): set this to True if the
                saturation curve is changing in time. Defaults to False.
            L_v (float, optional): The air expansion factor multiplied by the
                latent heat due to phase change divided by the specific heat
                capacity. For the atmosphere we take L_v to be 10, following A.2
                in Zerroukat and Allen (2015). Defaults to None but must be
                specified if using thermal feedback.
            vapour_name (str, optional): name of the water vapour variable.
                Defaults to 'water_vapour'.
            cloud_name (str, optional): name of the cloud variable. Defaults to
                'cloud_water'.
            convective_feedback (bool, optional): True if the conversion of
                vapour affects the height equation. Defaults to False.
            beta1 (float, optional): Condensation proportionality constant for
                height feedback, used if convection causes a response in the
                height equation. Defaults to None, but must be specified if
                convective_feedback is True.
            thermal_feedback (bool, optional): True if moist conversions
                affect the buoyancy equation. Defaults to False.
            beta2 (float, optional): Condensation proportionality constant
                for thermal feedback. Defaults to None, but must be specified
                if thermal_feedback is True.
            gamma_v (ufl expression or :class: `function`): The proportion of
                moist species that is converted when a conversion between
                vapour and cloud is taking place. Defaults to one, in which
                case the full amount of species to bring vapour to the
                saturation curve will undergo a conversion. Converting only a
                fraction avoids a two-timestep oscillation between vapour and
                cloud when saturation is tempertature/height-dependent.
            time_varying_gamma_v (bool, optional): set this to True
                if the fraction of moist species converted changes in time
                (if gamma_v is temperature/height-dependent).
            tau (float, optional): Timescale for condensation and evaporation.
                Defaults to None, in which case the timestep dt is used.
            parameters (:class:`Configuration`, optional): parameters containing
                the values of constants. Defaults to None, in which case the
                parameters are obtained from the equation.
        """

        self.explicit_only = True
        label_name = 'saturation_adjustment'
        super().__init__(equation, label_name, parameters=parameters)

        self.time_varying_saturation = time_varying_saturation
        self.convective_feedback = convective_feedback
        self.thermal_feedback = thermal_feedback
        self.time_varying_gamma_v = time_varying_gamma_v

        # Check for the correct fields
        assert vapour_name in equation.field_names, f"Field {vapour_name} does not exist in the equation set"
        assert cloud_name in equation.field_names, f"Field {cloud_name} does not exist in the equation set"
        self.Vv_idx = equation.field_names.index(vapour_name)
        self.Vc_idx = equation.field_names.index(cloud_name)

        if self.convective_feedback:
            assert "D" in equation.field_names, "Depth field must exist for convective feedback"
            assert beta1 is not None, "If convective feedback is used, beta1 parameter must be specified"

        if self.thermal_feedback:
            assert "b" in equation.field_names, "Buoyancy field must exist for thermal feedback"
            assert beta2 is not None, "If thermal feedback is used, beta2 parameter must be specified"
            assert L_v is not None, "If thermal feedback is used, L_v parameter must be specified"

        # Obtain function spaces and functions
        W = equation.function_space
        Vv = W.sub(self.Vv_idx)
        Vc = W.sub(self.Vc_idx)
        V_idxs = [self.Vv_idx, self.Vc_idx]

        # Source functions for both vapour and cloud
        self.water_v = Function(Vv)
        self.cloud = Function(Vc)

        # depth needed if convective feedback
        if self.convective_feedback:
            self.VD_idx = equation.field_names.index("D")
            VD = W.sub(self.VD_idx)
            self.D = Function(VD)
            V_idxs.append(self.VD_idx)

        # buoyancy needed if thermal feedback
        if self.thermal_feedback:
            self.Vb_idx = equation.field_names.index("b")
            Vb = W.sub(self.Vb_idx)
            self.b = Function(Vb)
            V_idxs.append(self.Vb_idx)

        # tau is the timescale for condensation/evaporation (may or may not be the timestep)
        if tau is not None:
            self.set_tau_to_dt = False
            self.tau = tau
        else:
            self.set_tau_to_dt = True
            self.tau = Constant(0)
            logger.info("Timescale for moisture conversion between vapour and cloud has been set to dt. If this is not the intention then provide a tau parameter as an argument to SWSaturationAdjustment.")

        if self.time_varying_saturation:
            if isinstance(saturation_curve, FunctionType):
                self.saturation_computation = saturation_curve
                self.saturation_curve = Function(Vv)
            else:
                raise NotImplementedError(
                    "If time_varying_saturation is True then saturation must be a Python function of at least one prognostic field.")
        else:
            assert not isinstance(saturation_curve, FunctionType), "If time_varying_saturation is not True then saturation cannot be a Python function."
            self.saturation_curve = saturation_curve

        # Saturation adjustment expression, adjusted to stop negative values
        sat_adj_expr = (self.water_v - self.saturation_curve) / self.tau
        sat_adj_expr = conditional(sat_adj_expr < 0,
                                   max_value(sat_adj_expr,
                                             -self.cloud / self.tau),
                                   min_value(sat_adj_expr,
                                             self.water_v / self.tau))

        # If gamma_v depends on variables
        if self.time_varying_gamma_v:
            if isinstance(gamma_v, FunctionType):
                self.gamma_v_computation = gamma_v
                self.gamma_v = Function(Vv)
            else:
                raise NotImplementedError(
                    "If time_varying_thermal_feedback is True then gamma_v must be a Python function of at least one prognostic field.")
        else:
            assert not isinstance(gamma_v, FunctionType), "If time_varying_thermal_feedback is not True then gamma_v cannot be a Python function."
            self.gamma_v = gamma_v

        # Factors for multiplying source for different variables
        factors = [self.gamma_v, -self.gamma_v]
        if convective_feedback:
            factors.append(self.gamma_v*beta1)
        if thermal_feedback:
            factors.append(parameters.g*L_v*self.gamma_v*beta2)

        # Add terms to equations and make interpolators
        self.source = [Function(Vc) for factor in factors]
        self.source_interpolators = [Interpolator(sat_adj_expr*factor, source)
                                     for factor, source in zip(factors, self.source)]

        tests = [equation.tests[idx] for idx in V_idxs]

        # Add source terms to residual
        for test, source in zip(tests, self.source):
            equation.residual += self.label(subject(test * source * dx,
                                                    equation.X), self.evaluate)

    def evaluate(self, x_in, dt):
        """
        Evaluates the source term generated by the physics.

        Computes the phyiscs contributions to water vapour and cloud water at
        each timestep.

        Args:
            x_in: (:class: 'Function'): the (mixed) field to be evolved.
            dt: (:class: 'Constant'): the timestep, which can be the time
                interval for the scheme.
        """
        logger.info(f'Evaluating physics parametrisation {self.label.label}')
        if self.convective_feedback:
            self.D.assign(x_in.split()[self.VD_idx])
        if self.thermal_feedback:
            self.b.assign(x_in.split()[self.Vb_idx])
        if self.time_varying_saturation:
            self.saturation_curve.interpolate(self.saturation_computation(x_in))
        if self.set_tau_to_dt:
            self.tau.assign(dt)
        self.water_v.assign(x_in.split()[self.Vv_idx])
        self.cloud.assign(x_in.split()[self.Vc_idx])
        if self.time_varying_gamma_v:
            self.gamma_v.interpolate(self.gamma_v_computation(x_in))
        for interpolator in self.source_interpolators:
            interpolator.interpolate()


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

        # -------------------------------------------------------------------- #
        # Extract prognostic variables
        # -------------------------------------------------------------------- #
        u_idx = equation.field_names.index('u')
        T_idx = equation.field_names.index('theta')
        rho_idx = equation.field_names.index('rho')
        if vapour_name is not None:
            m_v_idx = equation.field_names.index(vapour_name)

        X = self.X
        tests = TestFunctions(X.function_space()) if implicit_formulation else equation.tests

        u = split(X)[u_idx]
        rho = split(X)[rho_idx]
        theta_vd = split(X)[T_idx]
        test_theta = tests[T_idx]

        if vapour_name is not None:
            m_v = split(X)[m_v_idx]
            test_m_v = tests[m_v_idx]
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
            Vtheta = equation.spaces[T_idx]
            T_np1_expr = ((T + C_H*u_hori_mag*T_surface_expr*self.dt/z_a)
                          / (1 + C_H*u_hori_mag*self.dt/z_a))

            # If moist formulation, determine next vapour value
            if vapour_name is not None:
                source_mv = Function(Vtheta)
                mv_sat = thermodynamics.r_sat(equation.parameters, T, p)
                mv_np1_expr = ((m_v + C_E*u_hori_mag*mv_sat*self.dt/z_a)
                               / (1 + C_E*u_hori_mag*self.dt/z_a))
                dmv_expr = surface_expr * (mv_np1_expr - m_v) / self.dt
                source_mv_expr = test_m_v * source_mv * dx

                self.source_interpolators.append(Interpolator(dmv_expr, source_mv))
                equation.residual -= self.label(subject(prognostic(source_mv_expr, vapour_name),
                                                        X), self.evaluate)

                # Moisture needs including in theta_vd expression
                # NB: still using old pressure here, which implies constant p?
                epsilon = equation.parameters.R_d / equation.parameters.R_v
                theta_np1_expr = (thermodynamics.theta(equation.parameters, T_np1_expr, p)
                                  * (1 + mv_np1_expr / epsilon))

            else:
                theta_np1_expr = thermodynamics.theta(equation.parameters, T_np1_expr, p)

            source_theta_vd = Function(Vtheta)
            dtheta_vd_expr = surface_expr * (theta_np1_expr - theta_vd) / self.dt
            source_theta_expr = test_theta * source_theta_vd * dx
            self.source_interpolators.append(Interpolator(dtheta_vd_expr, source_theta_vd))
            equation.residual -= self.label(subject(prognostic(source_theta_expr, 'theta'),
                                                    X), self.evaluate)

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

    def evaluate(self, x_in, dt):
        """
        Evaluates the source term generated by the physics. This does nothing if
        the implicit formulation is not used.

        Args:
            x_in: (:class: 'Function'): the (mixed) field to be evolved.
            dt: (:class: 'Constant'): the timestep, which can be the time
                interval for the scheme.
        """

        logger.info(f'Evaluating physics parametrisation {self.label.label}')

        if self.implicit_formulation:
            self.X.assign(x_in)
            self.dt.assign(dt)
            self.rho_recoverer.project()
            for source_interpolator in self.source_interpolators:
                source_interpolator.interpolate()


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
            Vu = equation.spaces[u_idx]
            source_u = Function(Vu)
            u_np1_expr = u_hori / (1 + C_D*u_hori_mag*self.dt/z_a)

            du_expr = surface_expr * (u_np1_expr - u_hori) / self.dt

            # TODO: introduce reduced projector
            test_Vu = TestFunction(Vu)
            dx_reduced = dx(degree=4)
            proj_eqn = inner(test_Vu, source_u - du_expr)*dx_reduced
            proj_prob = NonlinearVariationalProblem(proj_eqn, source_u)
            self.source_projector = NonlinearVariationalSolver(proj_prob)

            source_expr = inner(test, source_u - k*dot(source_u, k)) * dx
            equation.residual -= self.label(subject(prognostic(source_expr, 'u'),
                                                    X), self.evaluate)

        # General formulation ------------------------------------------------ #
        else:
            # Construct underlying expressions
            du_dt = -surface_expr * C_D * u_hori_mag * u_hori / z_a

            dx_reduced = dx(degree=4)
            source_expr = inner(test, du_dt) * dx_reduced

            equation.residual -= self.label(subject(prognostic(source_expr, 'u'), X), self.evaluate)

    def evaluate(self, x_in, dt):
        """
        Evaluates the source term generated by the physics. This does nothing if
        the implicit formulation is not used.

        Args:
            x_in: (:class: 'Function'): the (mixed) field to be evolved.
            dt: (:class: 'Constant'): the timestep, which can be the time
                interval for the scheme.
        """

        logger.info(f'Evaluating physics parametrisation {self.label.label}')

        if self.implicit_formulation:
            self.X.assign(x_in)
            self.dt.assign(dt)
            self.source_projector.solve()


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

    def evaluate(self, x_in, dt):
        """
        Evaluates the source term generated by the physics. This does nothing if
        the implicit formulation is not used.

        Args:
            x_in: (:class: 'Function'): the (mixed) field to be evolved.
            dt: (:class: 'Constant'): the timestep, which can be the time
                interval for the scheme.
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

    def evaluate(self, x_in, dt):
        """
        Evaluates the source term generated by the physics. This does nothing if
        the implicit formulation is not used.

        Args:
            x_in: (:class: 'Function'): the (mixed) field to be evolved.
            dt: (:class: 'Constant'): the timestep, which can be the time
                interval for the scheme.
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

            C_D = conditional(u_hori_mag < 20.0, C_D0 + C_D1*u_hori_mag, C_D2)
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

    def evaluate(self, x_in, dt):
        """
        Evaluates the source term generated by the physics. This only recovers
        the density field.

        Args:
            x_in: (:class: 'Function'): the (mixed) field to be evolved.
            dt: (:class: 'Constant'): the timestep, which can be the time
                interval for the scheme.
        """

        logger.info(f'Evaluating physics parametrisation {self.label.label}')

        self.X.assign(x_in)
        self.rho_recoverer.project()
