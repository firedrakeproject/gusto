from abc import ABCMeta, abstractmethod

class Advection(object):
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
        Function takes x as input, computes F(x) and returns x_out
        as output.

        :arg x: :class:`.Function` object, the input Function.
        :arg x_out: :class:`.Function` object, the output Function.
        """
        pass    

class NoAdvection(Advection):
    """
    An non-advection scheme that does nothing.
    """

    def __init__(self, state):
        self.state = state

        #create a ubar field even though we don't use it.
        self.ubar = Function(state.V1)

    def apply(self, x_in, x_out):

        x_out.assign(x_in)

class LinearAdvection(Advection):
    """
    An advection scheme that uses the linearised background state 
    in evaluation of the advection term.
    """

    def __init__(self, state, vertical_only = False):
        self.state = state
        self.ubar = Function(state.V1)


