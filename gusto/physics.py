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

    def apply(self):
        self.water_v.assign(self.water_v_new)
        self.water_c.assign(self.water_c_new)
        self.theta.assign(self.theta_new) 
