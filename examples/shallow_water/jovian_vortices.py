import scipy
import numpy as np

from gusto import *
from firedrake import PeriodicRectangleMesh, SpatialCoordinate, as_vector, pi, interpolate, exp, cos, sin, sqrt, atan2, conditional

dt = 250.          # timestep (in seconds)
tmax = 5*dt        # duration of the simulation (in seconds)

# Set up the mesh and choose the refinement level
Lx = 7e7
Ly = Lx
nx = 256
ny = nx
mesh = PeriodicRectangleMesh(nx, ny, Lx, Ly, quadrilateral=True)
x, y = SpatialCoordinate(mesh)

# setup shallow water parameters
# specify g and Omega here if different from Earth values
H = 5e4
g = 24.79
Omega = 1.74e-4
R = 71.4e6
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

def smooth_f_profile(delta, rstar, Omega=parameters.Omega, R=R, Lx=Lx, nx=nx):
    """
    Returns the coefficients of a 5th order polynomial to smooth the transition from a polar gamma plane to the trap.

    Args:
      delta: half-width of smoothed region in number of cells
      rstar: radius of edge of trap in x-y grid coordinates
    """
    import sympy as sp
    delta *= Lx/nx
    Omega=Omega.dat.data[0]
    r = sp.symbols('r')

    print("Omega:", Omega, type(Omega))
    print("R:", R, type(R))
    print("r:", r, type(r))
    print("rstar:", rstar, type(rstar))
    print("delta:", delta, type(delta))
    print("float:", float)

    fexpr = 2*Omega*(1-0.5*r**2/R**2)
    left_val = fexpr.subs(r, rstar-delta)
    right_val = 2*Omega
    left_diff_val = sp.diff(fexpr, r).subs(r, rstar-delta)
    left_diff2_val = sp.diff(fexpr, r, 2).subs(r, rstar-delta)

    a = sp.symbols(f'a_0:6')
    P = a[0]
    for i in range(1, 6):
        P += a[i]*r**i
    eqns = [
        P.subs(r, rstar-delta) - left_val,
        P.subs(r, rstar+delta) - right_val,
        sp.diff(P, r).subs(r, rstar-delta) - left_diff_val,
        sp.diff(P, r).subs(r, rstar+delta),
        sp.diff(P, r, 2).subs(r, rstar-delta) - left_diff2_val,
        sp.diff(P, r, 2).subs(r, rstar+delta)
    ]

    sol = sp.solve(eqns, a)
    coeffs = [sol[sp.Symbol(f'a_{i}')] for i in range(6)]
    return coeffs

r, theta = rtheta_from_xy(x, y)

Omega = parameters.Omega
# define the expression for the gamma plane
fexpr = 2*Omega*(1-0.5*r**2/R**2)
# extract the coefficients of the smooth region. Here the trap radius is 3 cells from the edge of the domain, and the smoothing takes 2 cells either side of this trap radius
delta = 2
rstar = Lx/2-3*Lx/nx
coeffs = smooth_f_profile(delta, rstar)
# define a firedrake function using the polynomial coefficients
fsmooth = float(coeffs[0]) + float(coeffs[1])*r + float(coeffs[2])*r**2 + float(coeffs[3])*r**3 + float(coeffs[4])*r**4 + float(coeffs[5])*r**5
# use the gamma plane expression (fexpr), the smoothing (fsmooth) and the trap value (2*Omega) to build a full Coriolis expression
ftrap1 = conditional(r<rstar-delta*Lx/nx, fexpr, fsmooth)
ftrap = conditional(r<rstar+delta*Lx/nx, ftrap1, 2*Omega)

eqns = ShallowWaterEquations(domain, parameters, fexpr=ftrap)

output = OutputParameters(dirname="singlevortextrial7", dumpfreq=5)

diagnostic_fields = [SteadyStateError('u'), SteadyStateError('D'),
                     RelativeVorticity(), PotentialVorticity(),
                     ShallowWaterKineticEnergy(),
                     ShallowWaterPotentialEnergy(parameters),
                     ShallowWaterPotentialEnstrophy(),
                     CourantNumber()
                     ]

io = IO(domain, output=output, diagnostic_fields=diagnostic_fields)

subcycling_options = SubcyclingOptions(subcycle_by_courant=0.33)
transported_fields = [TrapeziumRule(domain, "u"),
                      SSPRK3(domain, "D", subcycling_options=subcycling_options)]
transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                  transport_methods)

u0 = stepper.fields("u")
D0 = stepper.fields("D")

south_lat_deg = [85.]#, 85., 85., 85., 85., 85.]#, 75.]
south_lon_deg = [0.]#, 0., 72., 144., 216., 288.]#, 0.]

b = 1.5               # Steepness parameter
Ro = 0.23             # Rossby Number
f0 = 2 * Omega.dat.data[0]        # Planetary vorticity
rm = 1e6              # Radius of vortex (m)
vm = Ro * f0 * rm     # Calculate speed with Ro
g = parameters.g.dat.data[0]
phi0 = g*H
Bu = phi0 / (f0 * rm)**2

south_lat = np.deg2rad(south_lat_deg)
south_lon = np.deg2rad(south_lon_deg)

def initialise_D(X, idx):
    # computes the initial depth perturbation corresponding to vortex
    # idx, given coordinates X

    # get the lon, lat coordinates of the centre of this vortex
    lamda_c = south_lon[idx]
    phi_c = south_lat[idx]

    # find the x-y (grid) coordinates of the centre of the vortex
    r_c, theta_c = rtheta_from_lonlat(lamda_c, phi_c)
    x_c, y_c = xy_from_rtheta(r_c, theta_c)

    # make an empty list of D values to append to
    D_values = []

    # loop over X coordinate values
    for Xval in X:
        x, y = Xval
        # calculate distance from centre of vortex to point
        dr = sqrt((x-x_c)**2 + (y-y_c)**2)
        phi_perturb = (Ro/Bu)*exp(1./b)*(b**(-1.+(2./b)))*scipy.special.gammaincc(2./b, (1./b)*(dr/rm)**b)*scipy.special.gamma(2./b)
        D_values.append(-1 * H * phi_perturb)

    # return list of D values in correct order
    return D_values

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
    Dtmp.dat.data[:] = initialise_D(X.dat.data_ro, idx)
    # add on to D0
    Dfinal += Dtmp
Dfinal += H

u_veloc = 0.*x
v_veloc = 0.*y

for i in range(len(south_lat)):
    r_c, theta_c = rtheta_from_lonlat(south_lon[i], south_lat[i])
    x_c, y_c = xy_from_rtheta(r_c, theta_c)
    dx = x - x_c
    dy = y - y_c
    dr = sqrt(dx**2 + dy**2)

    # Overide u,v components in velocity field
    mag_veloc = vm * ( dr / rm ) * exp( (1/b) * ( 1 - ( dr / rm )**b ) )

    u_veloc += - mag_veloc * (dy / dr)
    v_veloc += mag_veloc * (dx / dr)

uexpr = as_vector([u_veloc, v_veloc])

u0.project(uexpr)
D0.interpolate(Dfinal)

Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

# Run the timestepper and generate the output.
stepper.run(t=0, tmax=tmax)

