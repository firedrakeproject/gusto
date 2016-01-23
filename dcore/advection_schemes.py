from abc import ABCMeta, abstractmethod

class AdvectionScheme(object):
    """
    Base class for advection schemes for dcore.

    :arg state: x :class:`.State` object.
    """
    __metaclass__ = ABCMeta

    def __init__(self, state):
        self.state = state
    
    @abstractmethod
    def apply(self, x, x_out):
        """
        Function takes x as input, and returns 
        x_out
        as output.

        :arg x: :class:`.Function` object, the input Function.
        :arg x_out: :class:`.Function` object, the output Function.
        """
        pass


