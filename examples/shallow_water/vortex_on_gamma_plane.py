from gusto import (
    OutputParameters, pick_up_mesh, ShallowWaterParameters, Domain,
    ShallowWaterEquations, logger, RelativeVorticity, PotentialVorticity,
    ShallowWaterKineticEnergy, ShallowWaterPotentialEnergy,
    ShallowWaterPotentialEnstrophy, ShallowWaterAvailablePotentialEnergy,
    SteadyStateError, IO, SubcyclingOptions, TrapeziumRule, SSPRK3,
    DGUpwind, SemiImplicitQuasiNewton, VectorFunctionSpace, assemble,
    Function, FunctionSpace, WaterVapour, CloudWater, DG1Limiter,
    MoistConvectiveSWSolver, SWSaturationAdjustment, ForwardEuler,
    ZeroLimiter, MixedFSLimiter, MoistConvectiveSWRelativeHumidity,
    SWHeightRelax
)
from firedrake import (
    SpatialCoordinate, VertexOnlyMesh, as_vector, pi, interpolate, exp, cos, sin,
    sqrt, PeriodicRectangleMesh, atan2, conditional, RandomGenerator, PCG64,
    MinCellEdgeLength, MaxCellEdgeLength
)

import scipy
import numpy as np
import time
import os
import shutil
import pdb
from decimal import Decimal, ROUND_HALF_UP
import sympy as sp

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

def smooth_f_profile(degree, delta, style, rstar, Omega, R, Lx, nx):
    import sympy as sp

    delta *= Lx/nx
    r = sp.symbols('r')
    if style == 'polar':
        fexpr = 2*Omega*(1-0.5*r**2/R**2)
        left_val = fexpr.subs(r, rstar-delta)
        right_val = 2*Omega
        left_diff_val = sp.diff(fexpr, r).subs(r, rstar-delta)
        left_diff2_val = sp.diff(fexpr, r, 2).subs(r, rstar-delta)
    elif style == 'flat':
        left_val = 2*Omega*(1-0.5*(rstar-delta)**2/R**2)
        right_val = 2*Omega
        left_diff_val = 0
        left_diff2_val = 0

    a = sp.symbols(f'a_0:{degree+1}')
    P = a[0]
    for i in range(1, degree+1):
        P += a[i]*r**i

    if degree == 3:
        eqns = [
            P.subs(r, rstar-delta) - left_val,
            P.subs(r, rstar+delta) - right_val,
            sp.diff(P, r).subs(r, rstar-delta) - left_diff_val,
            sp.diff(P, r).subs(r, rstar+delta)
        ]
    elif degree == 5:
        eqns = [
            P.subs(r, rstar-delta) - left_val,
            P.subs(r, rstar+delta) - right_val,
            sp.diff(P, r).subs(r, rstar-delta) - left_diff_val,
            sp.diff(P, r).subs(r, rstar+delta),
            sp.diff(P, r, 2).subs(r, rstar-delta) - left_diff2_val,
            sp.diff(P, r, 2).subs(r, rstar+delta)
        ]
    else:
        print('do not have BCs for this degree')

    sol = sp.solve(eqns, a)
    coeffs = [sol[sp.Symbol(f'a_{i}')] for i in range(degree+1)]
    # P_smooth = P.subs(sol)
    # f_smooth = sp.Piecewise(
    #     (fexpr, r<rstar-delta),
    #     (P_smooth, (rstar-delta<=r) & (r<=rstar+delta)),
    #     (right_val, rstar+delta<r)
    # )
    return coeffs

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
        phi_perturb = (Ro/Bu)*exp(1./b)*(b**(-1.+(2./b)))*scipy.special.gammaincc(2./b, (1./b)*(dr/rm)**b)*scipy.special.gamma(2./b)
        # print('append')
        if not ac:
            D_values.append(-1 * H * phi_perturb)
        ### I think this would turn it into an anticyclone???
        elif ac:
            D_values.append(H * phi_perturb)

    # print('loop done')
    # return list of D values in correct order
    return D_values

### setup shallow water parameters
g = 24.79
Omega = 1.74e-4
R = 71.4e6  # radius of sphere that the gamma plane is calculated for
# mean depth H is calculated based on vortex parameters, H=phi0/g, phi0=Bu(f0*rm)**2
f0 = 2*Omega

### setup vortex parameters
Bu = 10
b = 1.5
Ro = 0.2
rm = 1e6 
vm = Ro * f0 * rm
phi0 = Bu * (f0*rm)**2
H = phi0/g
t_day = 2*pi/Omega

### setup timing options
dump_freq = 1    # how often data is output
dt = 250
tmax = 5*t_day

### setup grid parameters
nx = 256    # number of cells in x direction
ny = nx
Lx = 7e7    # Length of grid in x direction
Ly = Lx
rstar = Lx/2-3*Lx/nx    # location of edge of trap (middle of the smoothed region) in grid (x-y) coordinates
rstarr, rstartheta = rtheta_from_xy(rstar, 0)
_, rstarlat_rad = lonlat_from_rtheta(rstarr, rstartheta)
rstarlat_deg = 180./pi * rstarlat_rad
logger.info(f'Edge of trap is at {rstarlat_deg:.2f}deg')

### vortex location(s)
lat_deg = [80.]
lon_deg = [0.]

### directory to save output
dirname=f'/data/home/sh1293/results/jupiter_sw/testing'

### setup the mesh
mesh = PeriodicRectangleMesh(nx=nx, ny=ny, Lx=Lx, Ly=Ly, quadrilateral=True)