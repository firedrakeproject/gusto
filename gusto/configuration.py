"""
Some simple tools for making model configuration nicer.
"""

from firedrake import sqrt, PeriodicIntervalMesh, PeriodicRectangleMesh,\
    ExtrudedMesh, IcosahedralSphereMesh, SpatialCoordinate,\
    CellNormal, inner, interpolate, Constant, as_vector, cross


__all__ = ["TimesteppingParameters", "OutputParameters", "IncompressibleParameters", "CompressibleParameters", "ShallowWaterParameters", "EadyParameters", "CompressibleEadyParameters", "Sphere", "VerticalSlice"]


class Configuration(object):

    def __init__(self, **kwargs):

        for name, value in kwargs.items():
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


class PhysicalParameters(Configuration):
    g = 9.810616
    Omega = 7.292e-5  # rotation rate


class StratificationParameters(Configuration):
    N = 0.01  # Brunt-Vaisala frequency (1/s)


class IncompressibleParameters(PhysicalParameters, StratificationParameters):
    pass


class CompressibleParameters(PhysicalParameters):

    """
    Physical parameters for Compressible Euler
    """
    cp = 1004.5  # SHC of dry air at const. pressure (J/kg/K)
    R_d = 287.  # Gas constant for dry air (J/kg/K)
    kappa = 2.0/7.0  # R_d/c_p
    p_0 = 1000.0*100.0  # reference pressure (Pa, not hPa)
    cv = 717.  # SHC of dry air at const. volume (J/kg/K)


class MoistureParameters(Configuration):
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


class ShallowWaterParameters(PhysicalParameters):

    """
    Physical parameters for shallow water simulations
    """
    H = None  # mean depth


class EadyParameters(Configuration):

    """
    Physical parameters for Incompressible Eady
    """
    Nsq = 2.5e-05  # squared Brunt-Vaisala frequency (1/s)
    dbdy = -1.0e-07
    H = None
    L = None
    f = None
    deltax = None
    deltaz = None
    fourthorder = False


class CompressibleEadyParameters(CompressibleParameters, EadyParameters):

    """
    Physical parameters for Compressible Eady
    """
    g = 10.
    N = sqrt(EadyParameters.Nsq)
    theta_surf = 300.
    dthetady = theta_surf/g*EadyParameters.dbdy
    Pi0 = 0.0


class SphericalParameters(Configuration):
    R = 6371220.
    H = None


class VerticalSliceParameters(Configuration):
    L = None
    H = None
    deltax = None
    deltaz = None


class PhysicalDomain(object):

    def __init__(self, mesh, domain_parameters, vertical_normal, *,
                 perp=None, is_3d=False, on_sphere=True, is_extruded=True):
        self.mesh = mesh
        self.domain_parameters = domain_parameters
        self.vertical_normal = vertical_normal
        self.is_3d = is_3d
        self.on_sphere = on_sphere
        self.is_extruded = is_extruded
        if perp is not None:
            self.perp = perp
        self.is_rotating = None


def Sphere(mesh=None, *, radius=None, ref_level=None, nlayers=None, H=None):

    if mesh is None:
        if ref_level is None or radius is None:
            raise ValueError("Either provide a mesh, or supply the spherical radius and a reference level so that we can generate one for you.")
        m = IcosahedralSphereMesh(radius=radius, refinement_level=ref_level,
                                  degree=3)
        if all([nlayers, H]):
            mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers,
                                extrusion_type="radial")
            is_3d = True
        elif any([nlayers, H]):
            raise ValueError("Must supply both nlayers and H, or neither.")
        else:
            mesh = m
            mesh.init_cell_orientations(SpatialCoordinate(mesh))
            is_3d = False
    else:
        is_3d = (mesh.topological_dimension() == 3)

    domain_parameters = SphericalParameters(H=H)

    if is_3d:
        perp = None
        is_3d = True
        is_extruded = True
    else:
        outward_normals = CellNormal(mesh)
        perp = lambda u: cross(outward_normals, u)
        is_3d = False
        is_extruded = False

    x = SpatialCoordinate(mesh)
    R = sqrt(inner(x, x))
    k = interpolate(x/R, mesh.coordinates.function_space())

    return PhysicalDomain(mesh, domain_parameters, k, perp=perp, is_3d=is_3d, is_extruded=is_extruded)


def VerticalSlice(mesh=None, *, H=None, L=None, ncolumns=None, nlayers=None,
                  is_3d=False):
    if mesh is None:
        if all([H, L, ncolumns, nlayers]):
            if is_3d:
                m = PeriodicRectangleMesh(ncolumns, 1, L, 1.e5,
                                          quadrilateral=True)
            else:
                m = PeriodicIntervalMesh(ncolumns, L)
            mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
        else:
            raise ValueError("Either provide a mesh, or supply the parameters so that we can generate one for you.")
    else:
        is_3d = (mesh.topological_dimension() == 3)

    domain_parameters = VerticalSliceParameters(H=H, L=L,
                                                deltax=L/ncolumns,
                                                deltaz=H/nlayers)

    if is_3d:
        k = Constant([0.0, 0.0, 1.0])
        perp = None
    else:
        k = Constant([0.0, 1.0])
        perp = lambda u: as_vector([-u[1], u[0]])

    return PhysicalDomain(mesh, domain_parameters, k, perp=perp, is_3d=is_3d, on_sphere=False)
