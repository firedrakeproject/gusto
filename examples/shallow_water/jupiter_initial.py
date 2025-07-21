from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, as_vector, pi, interpolate, exp, cos, sin, sqrt, PeriodicRectangleMesh, atan2, conditional
import scipy
import numpy as np
import time

# setup shallow water parameters
Bu = 10
# H = 5e4
g = 24.79
Omega = 1.74e-4
R = 71.4e6

nx = 256
ny = nx
Lx = 7e7
Ly = Lx

b = 1.5               # Steepness parameter
Ro = 0.23             # Rossby Number
f0 = 2 * Omega        # Planetary vorticity
rm = 1e6              # Radius of vortex (m)
vm = Ro * f0 * rm     # Calculate speed with Ro

phi0 = Bu * (f0*rm)**2
H = phi0/g

t_day = 2*pi/Omega
dt = 250.          # timestep (in seconds)
tmax = 50*t_day       # duration of the simulation (in seconds)

folder_name = 'jupiter_initial_multi'

# Set up the mesh
mesh = PeriodicRectangleMesh(nx=nx, ny=ny, Lx=Lx, Ly=Ly, quadrilateral=True)
x, y = SpatialCoordinate(mesh)
# x *= Lx
# y *= Ly

parameters = ShallowWaterParameters(mesh, H=H, g=g, Omega=Omega)

domain = Domain(mesh, dt, "RTCF", 1)

def rtheta_from_xy(x, y, angle_units='rad'):
    """
    Returns the r, theta coordinates (where theta is measured anticlockwise from horizontal) from the planar
    Cartesian x, y coordinates.

    Args:
        x (:class:`np.ndarray` or :class:`ufl.Expr`): x-coordinate.
        y (:class:`np.ndarray` or :class:`ufl.Expr`): y-coordinate.
        angle_units (str, optional): the units to use for the angle. Valid
            options are 'rad' (radians) or 'deg' (degrees). Defaults to 'rad'.

    Returns:
        tuple of :class`np.ndarray` or tuple of :class:`ufl.Expr`: the tuple
            of (r, theta) coordinates in the appropriate form given the
            provided arguments.
    """

    if angle_units not in ['rad', 'deg']:
        raise ValueError(f'angle_units arg {angle_units} not valid')

    if angle_units == 'deg':
        unit_factor = 180./pi
    if angle_units == 'rad':
        unit_factor = 1.0
    
    x_shift = x-Lx/2
    y_shift = y-Ly/2

    theta = atan2(y_shift, x_shift)
    r = sqrt(x_shift**2 + y_shift**2)

    return r, theta*unit_factor

def lonlat_from_rtheta(r, theta, angle_units='rad', pole='north'):
    """
    Returns the lon lat coordinates from the polar r theta coordinates

    Args:
        r (:class:`np.ndarray` or :class:`ufl.Expr`): r-coordinate.
        theta (:class:`np.ndarray` or :class:`ufl.Expr`): theta-coordinate.
        angle_units (str, optional): the units to use for the angle. Valid
            options are 'rad' (radians) or 'deg' (degrees). Defaults to 'rad'.

    Returns:
        tuple of :class`np.ndarray` or tuple of :class:`ufl.Expr`: the tuple
            of (lon, lat) coordinates in the appropriate form given the
            provided arguments.
    """

    if angle_units not in ['rad', 'deg']:
        raise ValueError(f'angle_units arg {angle_units} not valid')

    if angle_units == 'deg':
        unit_factor = 180./pi
    if angle_units == 'rad':
        unit_factor = 1.0

    theta_scaled = theta/unit_factor
    
    lon = pi/2-theta_scaled
    lat = pi/2 - r/R
    if pole == 'south':
        lat*=-1

    return lon*unit_factor, lat*unit_factor

def rtheta_from_lonlat(lon, lat, angle_units='rad', pole='north'):
    """
    Returns the polar r theta coordinates from the lon lat coordinates

    Args:
        lon (:class:`np.ndarray` or :class:`ufl.Expr`): lon-coordinate.
        lat (:class:`np.ndarray` or :class:`ufl.Expr`): lat-coordinate.
        angle_units (str, optional): the units to use for the angle. Valid
            options are 'rad' (radians) or 'deg' (degrees). Defaults to 'rad'.

    Returns:
        tuple of :class`np.ndarray` or tuple of :class:`ufl.Expr`: the tuple
            of (r, theta) coordinates in the appropriate form given the
            provided arguments.
    """

    if angle_units not in ['rad', 'deg']:
        raise ValueError(f'angle_units arg {angle_units} not valid')

    if angle_units == 'deg':
        unit_factor = 180./pi
    if angle_units == 'rad':
        unit_factor = 1.0
    
    if pole == 'south':
        lat*=-1

    lon_scaled = lon/unit_factor
    lat_scaled = lat/unit_factor

    theta = pi/2-lon_scaled
    r = R*(pi/2-lat_scaled)
    
    return r, theta*unit_factor

def xy_from_rtheta(r, theta, angle_units='rad'):
    """
    Returns the planar cartsian x, y coordinates from the
    r, theta coordinates

    Args:
        r (:class:`np.ndarray` or :class:`ufl.Expr`): r-coordinate.
        theta (:class:`np.ndarray` or :class:`ufl.Expr`): theta-coordinate.
        angle_units (str, optional): the units to use for the angle. Valid
            options are 'rad' (radians) or 'deg' (degrees). Defaults to 'rad'.

    Returns:
        tuple of :class`np.ndarray` or tuple of :class:`ufl.Expr`: the tuple
            of (x, y) coordinates in the appropriate form given the
            provided arguments.
    """

    if angle_units not in ['rad', 'deg']:
        raise ValueError(f'angle_units arg {angle_units} not valid')

    if angle_units == 'deg':
        unit_factor = 180./pi
    if angle_units == 'rad':
        unit_factor = 1.0
    
    theta = theta/unit_factor

    x = r * cos(theta)
    y = r * sin(theta)

    x_shift = x+Lx/2
    y_shift = y+Ly/2

    return x_shift, y_shift


r, theta = rtheta_from_xy(x, y)

# lon, lat = lonlat_from_rtheta(r, theta)

# Create a spatially varying function for the Coriolis force:
Omega = parameters.Omega
fexpr = 2*Omega*(1-0.5*r**2/R**2)
rstar = Lx/2
ftrap = conditional(r < rstar, fexpr, 2*Omega)
eqns = ShallowWaterEquations(domain, parameters, fexpr=ftrap)
logger.info(f'Estimated number of cores = {eqns.X.function_space().dim() / 50000} \n mpiexec -n nprocs python script.py')

output = OutputParameters(dirname=f'/data/home/sh1293/results/jupiter_sw/{folder_name}', dumpfreq=10)

# diagnostic_fields = [SteadyStateError('u'), SteadyStateError('D'),
#                      RelativeVorticity(), PotentialVorticity(),
#                      ShallowWaterKineticEnergy(),
#                      ShallowWaterPotentialEnergy(parameters),
#                      ShallowWaterPotentialEnstrophy(),
#                      CourantNumber()]#, MeridionalComponent('u'), ZonalComponent('u')]

diagnostic_fields = [RelativeVorticity(), PotentialVorticity()]

io = IO(domain, output=output, diagnostic_fields=diagnostic_fields)

subcycling_options = SubcyclingOptions(subcycle_by_courant=0.33)
transported_fields = [TrapeziumRule(domain, "u"),
                      SSPRK3(domain, "D", subcycling_options=subcycling_options)]
transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                  transport_methods)

u0 = stepper.fields("u")
D0 = stepper.fields("D")

# south_lat_deg = [90., 85., 85., 85., 85., 75.]
south_lat_deg = [90., 83., 83., 83., 83., 83., 70.]
south_lon_deg = [0., 72., 144., 216., 288., 0., 0.]

# south_lat = [deg*pi/180. for deg in south_lat_deg]
# south_lon = [deg*pi/180. for deg in south_lon_deg]

south_lat = np.deg2rad(south_lat_deg)
south_lon = np.deg2rad(south_lon_deg)

def initialise_D(X, idx):
    # computes the initial depth perturbation corresponding to vortex
    # idx, given coordinates X

    # print('getting coords from list')
    # get the lon, lat coordinates of the centre of this vortex
    lamda_c = south_lon[idx]
    phi_c = south_lat[idx]

    # now want to convert this to x_c and y_c, to make distance calculations easier, as doing it on a plane
    # print('convert to r theta')
    r_c, theta_c = rtheta_from_lonlat(lamda_c, phi_c)
    # print('convert to x y')
    x_c, y_c = xy_from_rtheta(r_c, theta_c)

    # make an empty list of D values to append to
    D_values = []

    # print('beginning looping')
    # loop over X coordinate values
    for Xval in X:
        # print('getting xy')
        x, y = Xval
        # calculate distance from centre to this point
        # print('calc r')
        dr = sqrt((x-x_c)**2 + (y-y_c)**2)
        # print('calc phi')
        phi_perturb = (Ro/Bu)*exp(1./b)*(b**(-1.+(2./b)))*scipy.special.gammaincc(2./b, (1./b)*(dr/rm)**b)
        # print('append')
        D_values.append(-1 * H * phi_perturb)

    # print('loop done')
    # return list of D values in correct order
    return D_values

logger.info('Setting initial depth field')
# get the function space from the field we would like to initialise (D0)
VD = D0.function_space()
# create a vector function space for the coordinate values with the
# same mesh and finite element as D0
W = VectorFunctionSpace(VD.mesh(), VD.ufl_element())
# set up a function, X, with the value of the coordinates
# corresponding to the DOFs of D0
X = assemble(interpolate(VD.mesh().coordinates, W))
# set up a temporary function with the same structure as D to hold the
# values from each vortex
Dtmp = Function(D0.function_space())
Dfinal = Function(D0.function_space())
# loop over vortices
for idx in range(len(south_lat)):
    # calculate depth perturbation for each vortex
    # print('beginning initialise D')
    Dtmp.dat.data[:] = initialise_D(X.dat.data_ro, idx)
    # print('ending initialise D')
    # add on to D0
    Dfinal += Dtmp

u_veloc = 0.*x
v_veloc = 0.*y
Dfinal += H

logger.info('Setting initial velocity field')
for i in range(len(south_lat)):
    r_c, theta_c = rtheta_from_lonlat(south_lon[i], south_lat[i])
    # logger.info(f'r is {r_c}, theta is {theta_c}')
    x_c, y_c = xy_from_rtheta(r_c, theta_c)
    # logger.info(f'x is {x_c}, y is {y_c}')
    # print(f'(x, y) is ({x_c}, {y_c})')
    dr = sqrt((x-x_c)**2 + (y-y_c)**2)

    # Overide u,v components in velocity field
    # u_veloc += - vm * ( r / rm ) * exp( (1/b) * ( 1 - ( r / rm )**b ) ) * ( (y_grid-yy) / ( r + 1e-16 ) )
    # v_veloc += vm * ( r / rm ) * exp( (1/b) * ( 1 - ( r / rm )**b ) ) * ( (x_grid-xx) / ( r + 1e-16 ) )

    mag_veloc = vm * ( dr / rm ) * exp( (1/b) * ( 1 - ( dr / rm )**b ) )
    # logger.info(f'mag_veloc is {mag_veloc}')
    dx = x - x_c
    dy = y - y_c
    dl = sqrt(dx**2 + dy**2)

    u_veloc += - mag_veloc * (dy / dl)
    v_veloc += mag_veloc * (dx / dl)

uexpr = as_vector([u_veloc, v_veloc])

u0.project(uexpr)
D0.interpolate(Dfinal)

Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

start_time = time.time()

# Run the timestepper and generate the output.
stepper.run(t=0, tmax=tmax)

end_time = time.time()

logger.info(f'Total time taken {(end_time-start_time):.2f} seconds')
logger.info(f'File produced:\njupiter_sw/{folder_name}')