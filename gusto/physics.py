from abc import ABCMeta, abstractmethod
from firedrake import exp, Interpolator, conditional, interpolate


class Physics(object):
    """
    Base class for physics processes for Gusto.

    :arg state: :class:`.State` object.
    """
    __metaclass__ = ABCMeta

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

        # declare function space
        V = self.theta.function_space()

        param = self.state.parameters

        # define some parameters as attributes
        dt = self.state.timestepping.dt
        R_d = param.R_d
        p_0 = param.p_0
        kappa = param.kappa
        cp = param.cp
        cv = param.cv
        c_pv = param.c_pv
        c_pl = param.c_pl
        c_vv = param.c_vv
        R_v = param.R_v
        L_v0 = param.L_v0
        T_0 = param.T_0
        w_sat1 = param.w_sat1
        w_sat2 = param.w_sat2
        w_sat3 = param.w_sat3
        w_sat4 = param.w_sat4

        # make useful fields
        Pi = ((R_d * rho * self.theta / p_0)
              ** (kappa / (1.0 - kappa)))
        T = Pi * self.theta * R_d / (R_d + self.water_v * R_v)
        p = p_0 * Pi ** (1.0 / kappa)
        L_v = L_v0 - (c_pl - c_pv) * (T - T_0)
        R_m = R_d + R_v * self.water_v
        c_pml = cp + c_pv * self.water_v + c_pl * self.water_c
        c_vml = cv + c_vv * self.water_v + c_pl * self.water_c

        # use Teten's formula to calculate w_sat
        w_sat = (w_sat1 /
                 (p * exp(w_sat2 * (T - T_0) / (T - w_sat3)) - w_sat4))

        # correct w_sat to be positive
        w_sat = interpolate(conditional(w_sat > 0, w_sat, 999.0), V)

        # make appropriate condensation rate
        cond_rate = ((self.water_v - w_sat) /
                     (dt * (1.0 + ((L_v ** 2.0 * w_sat) /
                                   cp * R_v * T ** 2.0))))

        # adjust cond rate so negative concentrations don't occur
        cond_rate = interpolate(conditional(cond_rate < 0,
                                conditional(-cond_rate > self.water_c / dt,
                                            -self.water_c / dt,
                                            cond_rate),
                                conditional(cond_rate > self.water_v / dt,
                                            self.water_v / dt,
                                            cond_rate)), V)

        self.water_v_new = Interpolator(self.water_v - dt * cond_rate, V)
        self.water_c_new = Interpolator(self.water_c + dt * cond_rate, V)
        self.theta_new = Interpolator(self.theta *
                                      (1.0 + dt * cond_rate *
                                       (cv * L_v / (c_vml * cp * T) -
                                        R_v * cv * c_pml / (R_m * cp * c_vml))), V)

    def apply(self):
        self.water_v.assign(self.water_v_new.interpolate())
        self.water_c.assign(self.water_c_new.interpolate())
        self.theta.assign(self.theta_new.interpolate())
