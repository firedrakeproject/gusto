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
    Timestepping parameters for Gusto
    """
    dt = None
    alpha = 0.5
    maxk = 4
    maxi = 1


class OutputParameters(Configuration):

    """
    Output parameters for Gusto
    """

    Verbose = False
    dumpfreq = 1
    dumplist = None
    dumplist_latlon = []
    dirname = None
    #: Should the output fields be interpolated or projected to
    #: a linear space?  Default is interpolation.
    project_fields = False
    #: List of fields to dump error fields for steady state simulation
    steady_state_error_fields = []
    #: List of fields for computing perturbations
    perturbation_fields = []


class CompressibleParameters(Configuration):

    """
    Physical parameters for Compressible Euler
    """
    g = 9.810616
    N = 0.01  # Brunt-Vaisala frequency (1/s)
    cp = 1004.5  # SHC of dry air at const. pressure (J/kg/K)
    R_d = 287.  # Gas constant for dry air (J/kg/K)
    kappa = 2.0/7.0  # R_d/c_p
    p_0 = 1000.0*100.0  # reference pressure (Pa, not hPa)
    cv = 717.  # SHC of dry air at const. volume (J/kg/K)
    c_pl = 4186.  # SHC of liq. wat. at const. pressure (J/kg/K)
    c_pv = 1885.  # SHC of wat. vap. at const. pressure (J/kg/K)
    c_vv = 1424.  # SHC of wat. vap. at const. pressure (J/kg/K)
    R_v = 461.  # gas constant of water vapour
    L_v0 = 2.5e6  # ref. value for latent heat of vap. (J/kg)
    T_0 = 273.15  # ref. temperature
    w_sat1 = 380.3  # first const. in Teten's formula (Pa)
    w_sat2 = -17.27  # second const. in Teten's formula (no units)
    w_sat3 = 35.86  # third const. in Teten's formula (K)
    w_sat4 = 610.9  # fourth const. in Teten's formula (Pa)


class ShallowWaterParameters(Configuration):

    """
    Physical parameters for 3d Compressible Euler
    """
    g = 9.80616
    Omega = 7.292e-5  # rotation rate
    H = None  # mean depth


class EadyParameters(Configuration):

    """
    Physical parameters for nonlinear eady
    """
    Nsq = 2.5e-05  # squared Brunt-Vaisala frequency (1/s)
    dbdy = -1.0e-07
    H = None
    geopotential = False  # use geopotential for gravity term
