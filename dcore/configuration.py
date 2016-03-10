"""
Some simple tools for making model configuration nicer.
"""


class Configuration(object):

    def __init__(self, **kwargs):

        for name, value in kwargs.iteritems():
            self.__setattr__(name, value)

    def __setattr__(self, name, value):
        """Cause setting an unknown attribute to be an error"""
        if not hasattr(self, name):
            raise AttributeError("'%s' object has no attribute '%s'" % (type(self).__name__, name))
        object.__setattr__(self, name, value)


class TimesteppingParameters(Configuration):

    """
    Timestepping parameters for dcore
    """
    dt = None
    alpha = 0.5
    maxk = 2
    maxi = 2


class OutputParameters(Configuration):

    """
    Output parameters for dcore
    """

    Verbose = False
    dumpfreq = 10
    dumplist = (True,True,True)


class CompressibleParameters(Configuration):

    """
    Physical parameters for 3d Compressible Euler
    """

    g = 9.81
    N = 0.01
    cp = 1004.5
    R_d = 287.
    p_0 = 1000.0*1000.0
    kappa = 2.0/7.0
    k = None
    Omega = None
