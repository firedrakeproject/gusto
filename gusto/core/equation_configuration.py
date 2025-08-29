"""Some simple tools for configuring the model."""
from firedrake import Function, FunctionSpace, Constant
import inspect


__all__ = [
    "BoussinesqParameters", "CompressibleParameters",
    "ShallowWaterParameters",
    "SpongeLayerParameters", "DiffusionParameters", "BoundaryLayerParameters",
    "PMLParameters"
]


class EquationParameters(object):
    """A base configuration object for storing equation parameters."""

    mesh = None

    def __init__(self, mesh, **kwargs):
        """
        Args:
            mesh: for creating the real function space
            **kwargs: attributes and their values to be stored in the object.
        """
        self.mesh = mesh
        typecheck = lambda val: type(val) in [float, int, Constant]
        params = dict(inspect.getmembers(self, typecheck))
        params.update(kwargs.items())
        for name, value in params.items():
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
            raise AttributeError("'%s' object has no attribute '%s'" % (type(self).__name__, name))

        # Almost all parameters should be functions on the real space
        # -- but there are some specific exceptions which should be
        # kept as integers
        if self.mesh is not None:
            # This check is required so that on instantiation we do
            # not hit this line while self.mesh is still None
            R = FunctionSpace(self.mesh, 'R', 0)
        if type(value) in [float, int, Constant]:
            object.__setattr__(self, name, Function(R, val=float(value)))
        else:
            object.__setattr__(self, name, value)


class BoussinesqParameters(EquationParameters):
    """Physical parameters for the Boussinesq equations."""

    g = 9.810616
    N = 0.01  # Brunt-Vaisala frequency (1/s)
    cs = 340  # sound speed (for compressible case) (m/s)
    Omega = None


class CompressibleParameters(EquationParameters):
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


class ShallowWaterParameters(EquationParameters):
    """Physical parameters for the shallow-water equations."""

    g = 9.80616
    Omega = 7.292e-5  # rotation rate
    H = None  # mean depth
    # Factor that multiplies the vapour in the equivalent buoyancy
    # formulation of the thermal shallow water equations
    beta2 = None
    # Scaling factor for the saturation function in the equivalent buoyancy
    # formulation of the thermal shallow water equations
    nu = None
    # Scaling factor for the saturation function in the equivalent buoyancy
    # formulation of the thermal shallow water equations
    q0 = None


class SpongeLayerParameters(EquationParameters):
    """Specifies parameters describing a 'sponge' (damping) layer."""

    H = None
    z_level = None
    mubar = None

class PMLParameters(EquationParameters):
    """Specifies parameters describing a PML damping layer."""

    c_max = 350 # Fastest wave speed in the medium
    delta_frac = 0.1 # Fraction of domain that is the PML
    tol = 1e-3 # Tolerance for the PML error
    gamma0 = 0.1 # Stretching parameter


class DiffusionParameters(EquationParameters):
    """Parameters for a diffusion term with an interior penalty method."""

    kappa = None
    mu = None


class BoundaryLayerParameters(EquationParameters):
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


class HeldSuarezParameters(EquationParameters):
    """
    Parameters used in the default configuration for the Held Suarez test case.
    """
    T0stra = 200               # Stratosphere temp
    T0surf = 315               # Surface temperature at equator
    T0horiz = 60               # Equator to pole temperature difference
    T0vert = 10                # Stability parameter
    sigmab = 0.7               # Height of the boundary layer
    tau_d = 40 * 24 * 60 * 60  # 40 day time scale
    tau_u = 4 * 24 * 60 * 60   # 4 day timescale
