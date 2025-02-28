"""Some simple tools for configuring the model."""
from abc import ABCMeta, abstractproperty
from enum import Enum
from firedrake import sqrt


__all__ = [
    "Configuration",
    "IntegrateByParts", "TransportEquationType", "OutputParameters",
    "EmbeddedDGOptions", "ConservativeEmbeddedDGOptions", "RecoveryOptions",
    "ConservativeRecoveryOptions", "SUPGOptions", "MixedFSOptions",
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
        integers.

        Args:
            name: the attribute's name.
            value: the value to provide to the attribute.

        Raises:
            AttributeError: if the :class:`Configuration` object does not have
                this attribute pre-defined.
        """
        if not hasattr(self, name):
            raise AttributeError(f"{type(self).__name__} object has no attribute {name}.")

        # Almost all parameters should be functions on the real space
        # -- but there are some specific exceptions which should be
        # kept as integers
        non_constants = [
            'dumpfreq', 'pddumpfreq', 'chkptfreq',
            'fixed_subcycles', 'max_subcycles', 'subcycle_by_courant'
        ]
        if type(value) in [float, int] and name not in non_constants:
            raise AttributeError(f"Attribute {name} requires a mesh.")

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
