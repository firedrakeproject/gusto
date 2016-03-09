"""
Some simple tools for making model configuration nicer.
"""


class Configuration(object):
    def __init__(self, **kwargs):

        self.__setattr__ = self._proto__setattr__
        for name, value in kwargs.iteritems():
            print name, value
            self.__setattr__(name, value)

    def _proto__setattr__(self, name, value):
        """Cause setting an unknown attribute to be an error"""
        self.__getattribute__(name)
        object.__setattr__(self, name, value)

class TimesteppingParameters(Configuration):

    """
    Timestepping parameters for dcore
    """

    def __init__(self, **kwargs):
        self.dt = None
        self.alpha = 0.5
        self.maxk = 2
        self.maxi = 2

        super(TimesteppingParameters, self).__init__(**kwargs)

class OutputParameters(Configuration):

    """
    Output parameters for dcore
    """

    def __init__(self, **kwargs):
        self.Verbose = False
        self.dumpfreq = 10
        self.dumplist = (True,True,True)

        super(OutputParameters, self).__init__(**kwargs)

class CompressibleParameters(Configuration):

    """
    Physical parameters for 3d Compressible Euler
    """

    def __init__(self, **kwargs):
        self.g = 9.81
        self.N = 0.01
        self.cp = 1004.5
        self.R_d = 287.
        self.p_0 = 1000.0*1000.0
        self.kappa = 2.0/7.0
        self.k = None
        self.Omega = None

        super(CompressibleParameters, self).__init__(**kwargs)

### Example configuration starts here.

if __name__=="__main__":

    class MyConfiguration(Configuration):

        #: Documentation for myparam
        myparam = None
    
        #: As in GEM, manual suggests 0.1
        dt = 0.1

    c = MyConfiguration(dt=2)

    print c.dt
