"""Some simple tools for configuring the model."""
from abc import ABCMeta, abstractproperty
from enum import Enum
import logging
from logging import DEBUG, INFO, WARNING
from firedrake import sqrt, Constant


__all__ = ["WARNING", "INFO", "DEBUG", "IntegrateByParts",
           "TransportEquationType", "OutputParameters",
           "CompressibleParameters", "ShallowWaterParameters",
           "ConvectiveMoistShallowWaterParameters", "logger",
           "EmbeddedDGOptions", "RecoveryOptions", "SUPGOptions",
           "SpongeLayerParameters", "DiffusionParameters"]

logger = logging.getLogger("gusto")


def set_log_handler(comm):
    """
    Sets the handler for logging.

    Args:
        comm (:class:`MPI.Comm`): MPI communicator.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt="%(name)s:%(levelname)s %(message)s"))
    if logger.hasHandlers():
        logger.handlers.clear()
    if comm.rank == 0:
        logger.addHandler(handler)
    else:
        logger.addHandler(logging.NullHandler())


class IntegrateByParts(Enum):
    """Enumerator for setting the number of times to integrate by parts."""

    NEVER = 0
    ONCE = 1
    TWICE = 2


class TransportEquationType(Enum):
    u"""
    Enumerator for describing types of transport equation.

    For transporting velocity 'u' and transported quantity 'q', different types
    of transport equation include:

    advective: ∂q/∂t + (u.∇)q = 0
    conservative: ∂q/∂t + ∇.(u*q) = 0
    vector_invariant: ∂q/∂t + (∇×q)×u + (1/2)∇(q.u) + (1/2)[(∇q).u -(∇u).q)] = 0
    """

    no_transport = 702
    advective = 19
    conservative = 291
    vector_invariant = 9081


class Configuration(object):
    """A base configuration object, for storing aspects of the model."""

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: attributes and their values to be stored in the object.
        """
        for name, value in kwargs.items():
            self.__setattr__(name, value)

    def __setattr__(self, name, value):
        """
        Sets the model configuration attributes.

        When attributes are provided as floats or integers, these are converted
        to Firedrake :class:`Constant` objects, other than a handful of special
        integers (dumpfreq, pddumpfreq, chkptfreq and log_level).

        Args:
            name: the attribute's name.
            value: the value to provide to the attribute.

        Raises:
            AttributeError: if the :class:`Configuration` object does not have
                this attribute pre-defined.
        """
        if not hasattr(self, name):
            raise AttributeError("'%s' object has no attribute '%s'" % (type(self).__name__, name))

        # Almost all parameters should be Constants -- but there are some
        # specific exceptions which should be kept as integers
        if type(value) in [float, int] and name not in ['dumpfreq', 'pddumpfreq', 'chkptfreq', 'log_level']:
            object.__setattr__(self, name, Constant(value))
        else:
            object.__setattr__(self, name, value)


class OutputParameters(Configuration):
    """Parameters for controlling outputting."""

    #: log_level for logger, can be DEBUG, INFO or WARNING. Takes
    #: default value "warning"
    log_level = WARNING
    dump_vtus = True
    dumpfreq = 1
    pddumpfreq = None
    dumplist = None
    dumplist_latlon = []
    dump_diagnostics = True
    checkpoint = True
    checkpoint_pickup_filename = None
    chkptfreq = 1
    dirname = None
    #: TODO: Should the output fields be interpolated or projected to
    #: a linear space?  Default is interpolation.
    project_fields = False
    #: List of ordered pairs (name, points) where name is the field
    # name and points is the points at which to dump them
    point_data = []
    tolerance = None


class CompressibleParameters(Configuration):
    """Physical parameters for the Compressible Euler equations."""

    g = 9.810616
    N = 0.01  # Brunt-Vaisala frequency (1/s)
    cp = 1004.5  # SHC of dry air at const. pressure (J/kg/K)
    R_d = 287.  # Gas constant for dry air (J/kg/K)
    kappa = 2.0/7.0  # R_d/c_p
    p_0 = 1000.0*100.0  # reference pressure (Pa, not hPa)
    cv = 717.5  # SHC of dry air at const. volume (J/kg/K)
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
    """Physical parameters for the shallow-water equations."""

    g = 9.80616
    Omega = 7.292e-5  # rotation rate
    H = None  # mean depth


class ConvectiveMoistShallowWaterParameters(ShallowWaterParameters):

    """
    Physical parameters for the Bouchut et al moist shallow water equations
    """
    gamma = None  # condensation proportionality constant
    tau = None  # timescale of condensation
    q_0 = None  # factor in the saturation humidity expr
    alpha = None  # exponential factor in the saturation humidity expr


class TransportOptions(Configuration, metaclass=ABCMeta):
    """Base class for specifying options for a transport scheme."""

    @abstractproperty
    def name(self):
        pass


class EmbeddedDGOptions(TransportOptions):
    """Specifies options for an embedded DG method."""

    name = "embedded_dg"
    embedding_space = None


class RecoveryOptions(TransportOptions):
    """Specifies options for a recovery wrapper method."""

    name = "recovered"
    embedding_space = None
    recovered_space = None
    boundary_method = None
    injection_method = 'interpolate'
    project_high_method = 'interpolate'
    project_low_method = 'project'
    broken_method = 'interpolate'


class SUPGOptions(TransportOptions):
    """Specifies options for an SUPG scheme."""

    name = "supg"
    tau = None
    default = 1/sqrt(15)
    ibp = IntegrateByParts.TWICE


class SpongeLayerParameters(Configuration):
    """Specifies parameters describing a 'sponge' (damping) layer."""

    H = None
    z_level = None
    mubar = None


class DiffusionParameters(Configuration):
    """Parameters for a diffusion term with an interior penalty method."""

    kappa = None
    mu = None
