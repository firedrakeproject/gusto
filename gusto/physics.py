from abc import ABCMeta, abstractmethod
from gusto.transport_equation import EmbeddedDGAdvection
from gusto.advection import SSPRK3, Recoverer
from firedrake import Interpolator, conditional, Function, \
    min_value, max_value, as_vector, BrokenElement, FunctionSpace
from gusto import thermodynamics


__all__ = ["Condensation", "Fallout"]


class Physics(object, metaclass=ABCMeta):
    """
    Base class for physics processes for Gusto.

    :arg state: :class:`.State` object.
    """

    def __init__(self, state):
        self.state = state

    @abstractmethod
    def apply(self):
        """
        Function computes the value of specific
        fields at the next time step.
        """
        pass


class Condensation(Physics):
    """
    The process of condensation of water vapour
    into liquid water and evaporation of liquid
    water into water vapour, with the associated
    latent heat changes.

    :arg state: :class:`.State.` object.
    """

    def __init__(self, state):
        super(Condensation, self).__init__(state)

        # obtain our fields
        self.theta = state.fields('theta')
        self.water_v = state.fields('water_v')
        self.water_c = state.fields('water_c')
        rho = state.fields('rho')
        try:
            rain = state.fields('rain')
            water_l = self.water_c + rain
        except:
            water_l = self.water_c

        # declare function space
        Vt = self.theta.function_space()

        # make rho variables
        # we recover rho into theta space
        rho_averaged = Function(Vt)
        self.rho_broken = Function(FunctionSpace(state.mesh, BrokenElement(Vt.ufl_element())))
        self.rho_interpolator = Interpolator(rho, self.rho_broken.function_space())
        self.rho_recoverer = Recoverer(self.rho_broken, rho_averaged)

        # define some parameters as attributes
        dt = state.timestepping.dt
        R_d = state.parameters.R_d
        cp = state.parameters.cp
        cv = state.parameters.cv
        c_pv = state.parameters.c_pv
        c_pl = state.parameters.c_pl
        c_vv = state.parameters.c_vv
        R_v = state.parameters.R_v

        # make useful fields
        Pi = thermodynamics.pi(state.parameters, rho_averaged, self.theta)
        T = thermodynamics.T(state.parameters, self.theta, Pi, r_v=self.water_v)
        p = thermodynamics.p(state.parameters, Pi)
        L_v = thermodynamics.Lv(state.parameters, T)
        R_m = R_d + R_v * self.water_v
        c_pml = cp + c_pv * self.water_v + c_pl * water_l
        c_vml = cv + c_vv * self.water_v + c_pl * water_l

        # use Teten's formula to calculate w_sat
        w_sat = thermodynamics.r_sat(state.parameters, T, p)

        # make appropriate condensation rate
        dot_r_cond = ((self.water_v - w_sat) /
                      (dt * (1.0 + ((L_v ** 2.0 * w_sat) /
                                    (cp * R_v * T ** 2.0)))))

        # make cond_rate function, that needs to be the same for all updates in one time step
        cond_rate = Function(Vt)

        # adjust cond rate so negative concentrations don't occur
        self.lim_cond_rate = Interpolator(conditional(dot_r_cond < 0,
                                                      max_value(dot_r_cond, - self.water_c / dt),
                                                      min_value(dot_r_cond, self.water_v / dt)), cond_rate)

        # tell the prognostic fields what to update to
        self.water_v_new = Interpolator(self.water_v - dt * cond_rate, Vt)
        self.water_c_new = Interpolator(self.water_c + dt * cond_rate, Vt)
        self.theta_new = Interpolator(self.theta *
                                      (1.0 + dt * cond_rate *
                                       (cv * L_v / (c_vml * cp * T) -
                                        R_v * cv * c_pml / (R_m * cp * c_vml))), Vt)

    def apply(self):
        self.rho_broken.assign(self.rho_interpolator.interpolate())
        self.rho_recoverer.project()
        self.lim_cond_rate.interpolate()
        self.theta.assign(self.theta_new.interpolate())
        self.water_v.assign(self.water_v_new.interpolate())
        self.water_c.assign(self.water_c_new.interpolate())


class Fallout(Physics):
    """
    The fallout process of hydrometeors.

    :arg state :class: `.State.` object.
    :arg moments: an integer denoting which rainfall scheme to use.
                  Corresponds to the number of moments of the raindrop
                  distribution to be advected. Valid options:
                  0 -- rainfall all at terminal velocity 5 m/s.
                  1 -- droplet size depends upon density. Advect the mean
                  of the droplet size distribution.
    """

    def __init__(self, state):
        super(Fallout, self).__init__(state, moments)

        self.state = state
        self.moments = moments
        self.rain = state.fields('rain')
        self.v = Function(state.fields('u').function_space())

        # function spaces
        Vt = self.rain.function_space()
        Vu = state.fields('u').function_space()

        # introduce sedimentation rate
        # for now assume all rain falls at terminal velocity
        

        if moments == 0:
            # all rain falls at terminal velocity
            terminal_velocity = Constant(5)  # in m/s
            self.v.project(as_vector([0, -terminal_velocity]))
        elif moments == 1:
            rho = state.fields('rho')
            v_expression = rho
            raise NotImplementedError('sorry!')
        else:
            raise NotImplementedError('Currently we only have implementations for 0th and 1st moments of rainfall')

        if moments > 1:
            self.determine_v = Projector(as_vector([0, -v_expression]), self.v)

        # sedimentation will happen using a full advection method
        advection_equation = EmbeddedDGAdvection(state, Vt, equation_form="advective", outflow=True)
        self.advection_method = SSPRK3(state, self.rain, advection_equation)

    def apply(self):
        if self.moments > 0:
            self.determine_v.project()
        self.advection_method.update_ubar(self.v, self.v, 0)
        self.advection_method.apply(self.rain, self.rain)
