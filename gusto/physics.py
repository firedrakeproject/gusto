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

    def apply(self):
        self.water_v.assign( ... )
        self.water_c.assign( ... )
        self.theta.assign( ... )
