"""Some simple tools for configuring the model."""
from abc import ABCMeta, abstractproperty
from enum import Enum
from firedrake import sqrt, Constant


__all__ = [
    "Configuration",
    "IntegrateByParts", "TransportEquationType", "OutputParameters",
    "BoussinesqParameters", "CompressibleParameters",
    "ShallowWaterParameters",
    "EmbeddedDGOptions", "ConservativeEmbeddedDGOptions", "RecoveryOptions",
    "ConservativeRecoveryOptions", "SUPGOptions", "MixedFSOptions",
    "SpongeLayerParameters", "DiffusionParameters", "BoundaryLayerParameters",
    "SubcyclingOptions"
]


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
                                                                              \n
    advective: ∂q/∂t + (u.∇)q = 0                                             \n
    conservative: ∂q/∂t + ∇.(u*q) = 0                                         \n
    vector_invariant: ∂q/∂t + (∇×q)×u + (1/2)∇(q.u) + (1/2)[(∇q).u -(∇u).q)] = 0
    circulation: ∂q/∂t + (∇×q)×u + non-transport terms = 0
    tracer_conservative: ∂(q*rho)/∂t + ∇.(u*q*rho) = 0, for a reference density of rho
    for the tracer, q.
    """

    no_transport = 702
    advective = 19
    conservative = 291
    vector_invariant = 9081
    circulation = 512
    tracer_conservative = 296


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
        if type(value) in [float, int] and name not in ['dumpfreq', 'pddumpfreq', 'chkptfreq']:
            object.__setattr__(self, name, Constant(value))
        else:
            object.__setattr__(self, name, value)


class OutputParameters(Configuration):
    """Parameters for controlling outputting."""

    dump_vtus = True
    dump_nc = False
    dumpfreq = 1
    pddumpfreq = None
    dumplist = None
    dumplist_latlon = []
    dump_diagnostics = True
    diagfreq = 1
    checkpoint = False
    checkpoint_method = 'checkpointfile'
    checkpoint_pickup_filename = None
    chkptfreq = 1
    dirname = None
    log_courant = True
    #: TODO: Should the output fields be interpolated or projected to
    #: a linear space?  Default is interpolation.
    project_fields = False
    #: List of ordered pairs (name, points) where name is the field
    # name and points is the points at which to dump them
    point_data = []
    tolerance = None


class BoussinesqParameters(Configuration):
    """Physical parameters for the Boussinesq equations."""

    g = 9.810616
    N = 0.01  # Brunt-Vaisala frequency (1/s)
    cs = 340  # sound speed (for compressible case) (m/s)
    Omega = None


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
    Omega = None    # Rotation rate


class ShallowWaterParameters(Configuration):
    """Physical parameters for the shallow-water equations."""

    g = 9.80616
    Omega = 7.292e-5  # rotation rate
    H = None  # mean depth


class WrapperOptions(Configuration, metaclass=ABCMeta):
    """Base class for specifying options for a transport scheme."""

    @abstractproperty
    def name(self):
        pass


class EmbeddedDGOptions(WrapperOptions):
    """Specifies options for an embedded DG method."""

    name = "embedded_dg"
    project_back_method = 'project'
    embedding_space = None


class ConservativeEmbeddedDGOptions(EmbeddedDGOptions):
    """Specifies options for a conservative embedded DG method."""

    project_back_method = 'conservative_project'
    rho_name = None
    orig_rho_space = None


class RecoveryOptions(WrapperOptions):
    """Specifies options for a recovery wrapper method."""

    name = "recovered"
    embedding_space = None
    recovered_space = None
    boundary_method = None
    injection_method = 'interpolate'
    project_high_method = 'interpolate'
    project_low_method = 'project'
    broken_method = 'interpolate'


class ConservativeRecoveryOptions(RecoveryOptions):
    """Specifies options for a conservative recovery wrapper method."""

    rho_name = None
    orig_rho_space = None
    project_high_method = 'conservative_project'
    project_low_method = 'conservative_project'


class SUPGOptions(WrapperOptions):
    """Specifies options for an SUPG scheme."""

    name = "supg"
    tau = None
    default = 1/sqrt(15)
    ibp = IntegrateByParts.TWICE

    # Dictionary containing keys field_name and values term_labels
    # field_name (str): name of the field for SUPG to be applied to
    # term_label (list): labels of terms for test function to be altered
    #                    by SUPG
    suboptions = None


class MixedFSOptions(WrapperOptions):
    """Specifies options for a mixed finite element formulation
    where different suboptions are applied to different
    prognostic variables."""

    name = "mixed_options"

    # Dictionary containing keys field_name and values suboption
    # field_name (str): name of the field for suboption to be applied to
    # suboption (:class:`WrapperOptions`): Wrapper options to be applied
    #                                      to the provided field
    suboptions = None


class SpongeLayerParameters(Configuration):
    """Specifies parameters describing a 'sponge' (damping) layer."""

    H = None
    z_level = None
    mubar = None


class DiffusionParameters(Configuration):
    """Parameters for a diffusion term with an interior penalty method."""

    kappa = None
    mu = None


class BoundaryLayerParameters(Configuration):
    """
    Parameters for the idealised wind drag, surface flux and boundary layer
    mixing schemes.
    """

    coeff_drag_0 = 7e-4         # Zeroth drag coefficient (dimensionless)
    coeff_drag_1 = 6.5e-5       # First drag coefficient (s/m)
    coeff_drag_2 = 2e-3         # Second drag coefficient (dimensionless)
    coeff_heat = 1.1e-3         # Dimensionless surface sensible heat coefficient
    coeff_evap = 1.1e-3         # Dimensionless surface evaporation coefficient
    height_surface_layer = 75.  # Height (m) of surface level (usually lowest level)
    mu = 100.                   # Interior penalty coefficient for vertical diffusion


class SubcyclingOptions(Configuration):
    """
    Describes the process of subcycling a time discretisation, by dividing the
    time step into a number of smaller substeps.

    NB: cannot provide both the fixed_subcycles and max_subcycles parameters,
    which will raise an error.
    """

    # Either None, or an integer, giving the number of subcycles to take
    fixed_subcycles = None

    # If adaptive subcycling, the maximum number of subcycles to take
    max_subcycles = 10

    # Either None or a float, giving the maximum Courant number for one step
    subcycle_by_courant = None

    def check_options(self):
        """Checks that the subcycling options are valid."""

        if (self.fixed_subcycles is not None
                and self.subcycle_by_courant is not None):
            raise ValueError(
                "Cannot provide both fixed_subcycles and subcycle_by_courant"
                + "parameters.")
