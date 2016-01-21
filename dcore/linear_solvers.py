class TimesteppingSolver(object):
    """
    Base class for timestepping linear solvers for dcore.

    This is a dummy base class where the input is just copied to the output.

    :arg state: x_in :class:`.Function` object for the input
    :arg state: x_out :class:`.Function` object for the output
    """

    def __init__(x_in, x_out):
        self.x_in = x_in
        self.x_out = x_out
    
    def solve():
        """
        Function to execute the solver.
        """
        #This is a base class so we just copy x_in to x_out
        self.x_out.assign(x_in)
