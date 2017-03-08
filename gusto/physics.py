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

        # define some parameters as attributes
        dt = state.timestepping.dt
        R_d = state.parameters.R_d
        p_0 = state.parameters.p_0
        kappa = state.parameters.kappa
        cp = state.parameters.cp
        cv = state.parameters.cv
        c_pv = state.parameters.c_pv
        c_pl = state.parameters.c_pl
        c_vv = state.parameters.c_vv
        R_v = state.parameters.R_v
        L_v0 = state.parameters.L_v0
        T_0 = state.parameters.T_0
        w_sat1 = state.parameters.w_sat1
        w_sat2 = state.parameters.w_sat2
        w_sat3 = state.parameters.w_sat3
        w_sat4 = state.parameters.w_sat4

        # obtain our fields
        self.theta = getattr(state.fields, 'theta')
        self.water_v = getattr(state.fields, 'water_v')
        self.water_c = getattr(state.fields, 'water_c')
        self.rho = getattr(state.fields, 'rho')

        # declare function space
        V = self.theta.function_space()

        # make useful fields
        Pi = (R_d * rho * theta / p_0) ** (kappa / (1.0 - kappa))
        T = Pi * theta * R_d / (R_d + water_v * R_v)
        p = p_0 * Pi ** (1.0 / kappa)
        L_v = L_v0 - (c_pl - c_pv) * (T - T_0)
        R_m = R_d + R_v * water_v
        c_pml = cp + c_pv * water_v + c_pl * water_c
        c_vml = cv + c_vv * water_v + c_pl * water_c

        # use Teten's formula to calculate 

        

    def apply(self):
        self.water_v.assign(self.water_v_new)
        self.water_c.assign(self.water_c_new)
        self.theta.assign(self.theta_new) 
