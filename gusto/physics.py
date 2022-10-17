"""
Objects to perform parametrisations of physical processes, or "physics".

"Physics" schemes are routines to compute updates to prognostic fields that
represent the action of non-fluid processes, or those fluid processes that are
unresolved. This module contains a set of these processes in the form of classes
with "apply" methods.
"""

from abc import ABCMeta, abstractmethod
from gusto.recovery import Recoverer, Boundary_Method
from gusto.time_discretisation import SSPRK3
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
from gusto.equations import AdvectionEquation
from gusto.labels import subject, physics
from gusto.limiters import ThetaLimiter, NoLimiter
from gusto.configuration import logger, EmbeddedDGOptions, RecoveredOptions
from firedrake import (Interpolator, conditional, Function, dx,
                       min_value, max_value, as_vector, BrokenElement,
                       FunctionSpace, Constant, pi, Projector)
from gusto import thermodynamics
from math import gamma
from enum import Enum


__all__ = ["Condensation", "Fallout", "Coalescence", "Evaporation", "AdvectedMoments", "InstantRain"]


class Physics(object, metaclass=ABCMeta):
    """Base class for the parametrisation of physical processes for Gusto."""

    def __init__(self, state):
        """
        Args:
            state (:class:`State`): the model's state object.
        """
        self.state = state

    @abstractmethod
    def apply(self):
        """
        Computes the value of certain prognostic fields, representing the
        action of the physical process.
        """
        pass


class Condensation(Physics):
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

    def __init__(self, state, iterations=1):
        """
        Args:
            state (:class:`State`): the model's state object.
            iterations (int, optional): number of saturation adjustment
                iterations to perform for each call of the step. Defaults to 1.
        """
        super().__init__(state)

        self.iterations = iterations
        # obtain our fields
        self.theta = state.fields('theta')
        self.water_v = state.fields('vapour_mixing_ratio')
        self.water_c = state.fields('cloud_liquid_mixing_ratio')
        rho = state.fields('rho')
        try:
            # TODO: use the phase flag for the tracers here
            rain = state.fields('rain_mixing_ratio')
            water_l = self.water_c + rain
        except NotImplementedError:
            water_l = self.water_c

        # declare function space
        Vt = self.theta.function_space()

        # make rho variables
        # we recover rho into theta space
        h_deg = rho.function_space().ufl_element().degree()[0]
        v_deg = rho.function_space().ufl_element().degree()[1]
        if v_deg == 0 and h_deg == 0:
            boundary_method = Boundary_Method.physics
        else:
            boundary_method = None
        Vt_broken = FunctionSpace(state.mesh, BrokenElement(Vt.ufl_element()))
        rho_averaged = Function(Vt)
        self.rho_recoverer = Recoverer(rho, rho_averaged, VDG=Vt_broken, boundary_method=boundary_method)

        # define some parameters as attributes
        dt = state.dt
        R_d = state.parameters.R_d
        cp = state.parameters.cp
        cv = state.parameters.cv
        c_pv = state.parameters.c_pv
        c_pl = state.parameters.c_pl
        c_vv = state.parameters.c_vv
        R_v = state.parameters.R_v

        # make useful fields
        exner = thermodynamics.exner_pressure(state.parameters, rho_averaged, self.theta)
        T = thermodynamics.T(state.parameters, self.theta, exner, r_v=self.water_v)
        p = thermodynamics.p(state.parameters, exner)
        L_v = thermodynamics.Lv(state.parameters, T)
        R_m = R_d + R_v * self.water_v
        c_pml = cp + c_pv * self.water_v + c_pl * water_l
        c_vml = cv + c_vv * self.water_v + c_pl * water_l

        # use Teten's formula to calculate w_sat
        w_sat = thermodynamics.r_sat(state.parameters, T, p)

        # make appropriate condensation rate
        dot_r_cond = ((self.water_v - w_sat)
                      / (dt * (1.0 + ((L_v ** 2.0 * w_sat)
                                      / (cp * R_v * T ** 2.0)))))

        # make cond_rate function, that needs to be the same for all updates in one time step
        cond_rate = Function(Vt)

        # adjust cond rate so negative concentrations don't occur
        self.lim_cond_rate = Interpolator(conditional(dot_r_cond < 0,
                                                      max_value(dot_r_cond, - self.water_c / dt),
                                                      min_value(dot_r_cond, self.water_v / dt)), cond_rate)

        # tell the prognostic fields what to update to
        self.water_v_new = Interpolator(self.water_v - dt * cond_rate, Vt)
        self.water_c_new = Interpolator(self.water_c + dt * cond_rate, Vt)
        self.theta_new = Interpolator(self.theta
                                      * (1.0 + dt * cond_rate
                                         * (cv * L_v / (c_vml * cp * T)
                                            - R_v * cv * c_pml / (R_m * cp * c_vml))), Vt)

    def apply(self):
        """Applies the condensation/evaporation process."""
        self.rho_recoverer.project()
        for i in range(self.iterations):
            self.lim_cond_rate.interpolate()
            self.theta.assign(self.theta_new.interpolate())
            self.water_v.assign(self.water_v_new.interpolate())
            self.water_c.assign(self.water_c_new.interpolate())


class AdvectedMoments(Enum):
    """
    Enumerator describing the moments in the raindrop size distribution.

    This stores enumerators for the number of moments used to describe the
    size distribution curve of raindrops. This can be used for deciding which
    moments to advect in a precipitation scheme.
    """

    M0 = 0  # don't advect the distribution -- advect all rain at the same speed
    M3 = 1  # advect the mass of the distribution


class Fallout(Physics):
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

    def __init__(self, state, moments=AdvectedMoments.M3, limit=True):
        """
        Args:
            state (:class:`State`): the model's state object
            moments (int, optional): an :class:`AdvectedMoments` enumerator,
                representing the number of moments of the size distribution of
                raindrops to be transported. Defaults to `AdvectedMoments.M3`.
            limit (bool, optional): whether to apply a limiter to the transport.
                Defaults to True.

        Raises:
            NotImplementedError: the limiter is only implemented for specific
                spaces (the equispaced DG1 field and the degree 1 temperature
                space).
        """
        super().__init__(state)

        # function spaces
        Vt = state.spaces("theta")
        Vu = state.spaces("HDiv")

        # declare properties of class
        self.state = state
        self.moments = moments
        self.v = state.fields('rainfall_velocity', Vu)
        self.limit = limit

        # determine whether to do recovered space advection scheme
        # if horizontal and vertical degrees are 0 do recovered spac
        h_deg = Vt.ufl_element().degree()[0]
        v_deg = Vt.ufl_element().degree()[1] - 1
        if v_deg == 0 and h_deg == 0:
            VDG1 = state.spaces("DG1_equispaced")
            VCG1 = FunctionSpace(Vt.mesh(), "CG", 1)
            Vbrok = FunctionSpace(Vt.mesh(), BrokenElement(Vt.ufl_element()))
            boundary_method = Boundary_Method.dynamics
            advect_options = RecoveredOptions(embedding_space=VDG1,
                                              recovered_space=VCG1,
                                              broken_space=Vbrok,
                                              boundary_method=boundary_method)
        else:
            advect_options = EmbeddedDGOptions()

        # need to define advection equation before limiter (as it is needed for the ThetaLimiter)
        # TODO: check if rain is a mixing ratio
        advection_equation = AdvectionEquation(state, Vt, "rain_mixing_ratio", outflow=True)
        self.rain = state.fields("rain_mixing_ratio")

        if moments == AdvectedMoments.M0:
            # all rain falls at terminal velocity
            terminal_velocity = Constant(5)  # in m/s
            if state.mesh.geometric_dimension() == 2:
                self.v.project(as_vector([0, -terminal_velocity]))
            elif state.mesh.geometric_dimension() == 3:
                self.v.project(as_vector([0, 0, -terminal_velocity]))
        elif moments == AdvectedMoments.M3:
            # this advects the third moment M3 of the raindrop
            # distribution, which corresponds to the mean mass
            rho = state.fields('rho')
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
            Lambda = (N_r * pi * rho_w * gamma(4 + mu)
                      / (6 * gamma(1 + mu) * rho * self.rain)) ** (1. / 3)
            Lambda0 = (N_r * pi * rho_w * gamma(4 + mu)
                       / (6 * gamma(1 + mu) * rho * threshold)) ** (1. / 3)
            v_expression = conditional(self.rain > threshold,
                                       (a * gamma(4 + b + mu)
                                        / (gamma(4 + mu) * Lambda ** b)
                                        * (rho0 / rho) ** g),
                                       (a * gamma(4 + b + mu)
                                        / (gamma(4 + mu) * Lambda0 ** b)
                                        * (rho0 / rho) ** g))
        else:
            raise NotImplementedError('Currently we only have implementations for zero and one moment schemes for rainfall. Valid options are AdvectedMoments.M0 and AdvectedMoments.M3')

        if moments != AdvectedMoments.M0:
            # TODO: implement for spherical geometry. Raise an error if the
            # geometry is not Cartesian
            if state.mesh.geometric_dimension() == 2:
                self.determine_v = Projector(as_vector([0, -v_expression]), self.v)
            elif state.mesh.geometric_dimension() == 3:
                self.determine_v = Projector(as_vector([0, 0, -v_expression]), self.v)

        # decide which limiter to use
        if self.limit:
            if h_deg == 0 and v_deg == 0:
                limiter = VertexBasedLimiter(VDG1)
            elif h_deg == 1 and v_deg == 1:
                limiter = ThetaLimiter(Vt)
            else:
                logger.warning("There is no limiter yet implemented for the spaces used. NoLimiter() is being used for the rainfall in this case.")
                limiter = NoLimiter()
        else:
            limiter = None

        # sedimentation will happen using a full advection method
        self.advection_method = SSPRK3(state, options=advect_options, limiter=limiter)
        self.advection_method.setup(advection_equation, self.v)

    def apply(self):
        """Applies the precipitation process."""
        if self.moments != AdvectedMoments.M0:
            self.determine_v.project()
        self.advection_method.apply(self.rain, self.rain)


class Coalescence(Physics):
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

    def __init__(self, state, accretion=True, accumulation=True):
        """
        Args:
            state (:class:`State`): the model's state object.
            accretion (bool, optional): whether to include the accretion process
                in the parametrisation. Defaults to True.
            accumulation (bool, optional): whether to include the accumulation
                process in the parametrisation. Defaults to True.
        """
        super().__init__(state)

        # obtain our fields
        self.water_c = state.fields('cloud_liquid_mixing_ratio')
        self.rain = state.fields('rain_mixing_ratio')

        # declare function space
        Vt = self.water_c.function_space()

        # define some parameters as attributes
        dt = state.dt
        k_1 = Constant(0.001)  # accretion rate in 1/s
        k_2 = Constant(2.2)  # accumulation rate in 1/s
        a = Constant(0.001)  # min cloud conc in kg/kg
        b = Constant(0.875)  # power for rain in accumulation

        # make default rates to be zero
        accr_rate = Constant(0.0)
        accu_rate = Constant(0.0)

        if accretion:
            accr_rate = k_1 * (self.water_c - a)
        if accumulation:
            accu_rate = k_2 * self.water_c * self.rain ** b

        # make coalescence rate function, that needs to be the same for all updates in one time step
        coalesce_rate = Function(Vt)

        # adjust coalesce rate using min_value so negative cloud concentration doesn't occur
        self.lim_coalesce_rate = Interpolator(conditional(self.rain < 0.0,  # if rain is negative do only accretion
                                                          conditional(accr_rate < 0.0,
                                                                      0.0,
                                                                      min_value(accr_rate, self.water_c / dt)),
                                                          # don't turn rain back into cloud
                                                          conditional(accr_rate + accu_rate < 0.0,
                                                                      0.0,
                                                                      # if accretion rate is negative do only accumulation
                                                                      conditional(accr_rate < 0.0,
                                                                                  min_value(accu_rate, self.water_c / dt),
                                                                                  min_value(accr_rate + accu_rate, self.water_c / dt)))),
                                              coalesce_rate)

        # tell the prognostic fields what to update to
        self.water_c_new = Interpolator(self.water_c - dt * coalesce_rate, Vt)
        self.rain_new = Interpolator(self.rain + dt * coalesce_rate, Vt)

    def apply(self):
        """Applies the coalescence process."""
        self.lim_coalesce_rate.interpolate()
        self.rain.assign(self.rain_new.interpolate())
        self.water_c.assign(self.water_c_new.interpolate())


class Evaporation(Physics):
    """
    Represents the evaporation of rain into water vapour.

    This describes the evaporation of rain into water vapour, with the
    associated latent heat change. This parametrisation comes from Klemp and
    Wilhelmson (1978). The rate of change is limited to prevent production of
    negative moisture values.

    This is only implemented for mixing ratio variables, and when the prognostic
    is the virtual dry potential temperature.
    """

    def __init__(self, state):
        """
        Args:
            state (:class:`State`): the model's state object.
        """
        super().__init__(state)

        # obtain our fields
        self.theta = state.fields('theta')
        self.water_v = state.fields('vapour_mixing_ratio')
        self.rain = state.fields('rain_mixing_ratio')
        rho = state.fields('rho')
        try:
            water_c = state.fields('cloud_liquid_mixing_ratio')
            water_l = self.rain + water_c
        except NotImplementedError:
            water_l = self.rain

        # declare function space
        Vt = self.theta.function_space()

        # make rho variables
        # we recover rho into theta space
        h_deg = rho.function_space().ufl_element().degree()[0]
        v_deg = rho.function_space().ufl_element().degree()[1]
        if v_deg == 0 and h_deg == 0:
            boundary_method = Boundary_Method.physics
        else:
            boundary_method = None
        Vt_broken = FunctionSpace(state.mesh, BrokenElement(Vt.ufl_element()))
        rho_averaged = Function(Vt)
        self.rho_recoverer = Recoverer(rho, rho_averaged, VDG=Vt_broken, boundary_method=boundary_method)

        # define some parameters as attributes
        dt = state.dt
        R_d = state.parameters.R_d
        cp = state.parameters.cp
        cv = state.parameters.cv
        c_pv = state.parameters.c_pv
        c_pl = state.parameters.c_pl
        c_vv = state.parameters.c_vv
        R_v = state.parameters.R_v

        # make useful fields
        exner = thermodynamics.exner_pressure(state.parameters, rho_averaged, self.theta)
        T = thermodynamics.T(state.parameters, self.theta, exner, r_v=self.water_v)
        p = thermodynamics.p(state.parameters, exner)
        L_v = thermodynamics.Lv(state.parameters, T)
        R_m = R_d + R_v * self.water_v
        c_pml = cp + c_pv * self.water_v + c_pl * water_l
        c_vml = cv + c_vv * self.water_v + c_pl * water_l

        # use Teten's formula to calculate w_sat
        w_sat = thermodynamics.r_sat(state.parameters, T, p)

        # expression for ventilation factor
        a = Constant(1.6)
        b = Constant(124.9)
        c = Constant(0.2046)
        C = a + b * (rho_averaged * self.rain) ** c

        # make appropriate condensation rate
        f = Constant(5.4e5)
        g = Constant(2.55e6)
        h = Constant(0.525)
        dot_r_evap = (((1 - self.water_v / w_sat) * C * (rho_averaged * self.rain) ** h)
                      / (rho_averaged * (f + g / (p * w_sat))))

        # make evap_rate function, needs to be the same for all updates in one time step
        evap_rate = Function(Vt)

        # adjust evap rate so negative rain doesn't occur
        self.lim_evap_rate = Interpolator(conditional(dot_r_evap < 0,
                                                      0.0,
                                                      conditional(self.rain < 0.0,
                                                                  0.0,
                                                                  min_value(dot_r_evap, self.rain / dt))),
                                          evap_rate)

        # tell the prognostic fields what to update to
        self.water_v_new = Interpolator(self.water_v + dt * evap_rate, Vt)
        self.rain_new = Interpolator(self.rain - dt * evap_rate, Vt)
        self.theta_new = Interpolator(self.theta
                                      * (1.0 - dt * evap_rate
                                         * (cv * L_v / (c_vml * cp * T)
                                            - R_v * cv * c_pml / (R_m * cp * c_vml))), Vt)

    def apply(self):
        """Applies the process to evaporate rain droplets."""
        self.rho_recoverer.project()
        self.lim_evap_rate.interpolate()
        self.theta.assign(self.theta_new.interpolate())
        self.water_v.assign(self.water_v_new.interpolate())
        self.rain.assign(self.rain_new.interpolate())


class InstantRain(object):
    """
    The process of converting moisture above the saturation curve to rain.
    :arg state: :class:`.State.` object.
    :arg saturation_curve: the saturation function,
        above which excess moisture is converted to
        rain
    """

    def __init__(self, equation, saturation_curve):

        self.Vm_idx = equation.field_names.index("water_v")
        Vr_idx = equation.field_names.index("rain_mixing_ratio")

        # obtain function space and functions
        W = equation.function_space
        Vm = W.sub(self.Vm_idx)
        Vr = W.sub(Vr_idx)

        # the source function is the difference between the water
        # vapour and the saturation
        self.water_v = Function(Vm)
        self.source = Function(Vm)
        self.dt = Constant(0.0)

        test_m = equation.tests[self.Vm_idx]
        test_r = equation.tests[Vr_idx]
        equation.residual += physics(subject(test_m * self.source * dx
                                             - test_r * self.source * dx,
                                             equation.X),
                                     self.evaluate)

        # convert moisture above saturation curve to rain
        self.source_interpolator = Interpolator(conditional(
            self.water_v > saturation_curve,
            (1/self.dt)*(self.water_v - saturation_curve),
            0), Vm)

    def evaluate(self, x_in, dt):
        self.dt.assign(dt)
        self.water_v.assign(x_in.split()[self.Vm_idx])
        self.source.assign(self.source_interpolator.interpolate())
