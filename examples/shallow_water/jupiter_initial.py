from gusto import *
from firedrake import (SpatialCoordinate, VertexOnlyMesh,
                        as_vector, pi, interpolate, exp, cos, sin,
                        sqrt, PeriodicRectangleMesh, atan2,
                        conditional, RandomGenerator, PCG64)
import scipy
import numpy as np
import time
import os
import shutil

### options changed in Cheng Li 2020
Bu = 2
b = 1.5
Ro = 0.2

### overwrite H for Laura's setup
# H = 46e3

### setup grid parameters
nx = 256
ny = nx
Lx = 7e7
Ly = Lx
rstar = Lx/2-3*Lx/nx

### setup smoothing parameters
smooth_degree = 5
smooth_delta = 2

# setup shallow water parameters
# Bu = 10
# H = 5e4
g = 24.79
Omega = 1.74e-4
R = 71.4e6

# b = 1.5               # Steepness parameter
# Ro = 0.23             # Rossby Number
f0 = 2 * Omega        # Planetary vorticity
rm = 1e6              # Radius of vortex (m)
vm = Ro * f0 * rm     # Calculate speed with Ro

phi0 = Bu * (f0*rm)**2
H = phi0/g
t_day = 2*pi/Omega

### timing options
dump_freq = 1    # dump frequency of output
dt = 250          # timestep (in seconds)
tmax = 1*dt       # duration of the simulation (in seconds)

restart = False
restart_name = 'new_single_halfoffset_fplane_Bu2b1p5Rop2_l1dt250df1'
t0 = 200*t_day

### vortex locations
south_lat_deg = [90.]#, 83., 83., 83., 83., 83.]#, 70.]
south_lon_deg = [0.]#, 72., 144., 216., 288., 0.]#, 0.]

### name
setup = 'new_single'

### add noise to initial depth profile?
noise = False

### include available potential energy diagnostic or no?
avlpe_diag = True

#### if true then the experiment is done on an f-plane, if not then it's the PV trap as usual
fplane = True

### extract points - if True it extracts transect of y=ypole, x=xpole±40 'grid points'
extract_points = True

##########################################################################

tmin = np.ceil(t0/dump_freq)*dump_freq
tmax = np.ceil(tmax/dump_freq)*dump_freq

def split_number(x):
    x = float(x)
    xint, xdec = str(x).split('.')
    if xint == '0':
        xint = ''
    if xdec == '0':
        xdec = ''
    else:
        xdec = f'p{xdec}'
    return xint, xdec

bint, bdec = split_number(b)
Roint, Rodec = split_number(Ro)
Buint, Budec = split_number(Bu)

if setup != '':
    setup = f'{setup}_'
if fplane:
    setup =f'{setup}fplane_'
if noise:
    noise_name = f'_n'
else:
    noise_name = ''
folder_name = f'{setup}Bu{Buint}{Budec}b{bint}{bdec}Ro{Roint}{Rodec}_l{round(tmax/t_day)}dt{int(dt)}df{dump_freq}{noise_name}'

dirname=f'/data/home/sh1293/results/jupiter_sw/{folder_name}'
dirnameold=f'/data/home/sh1293/results/jupiter_sw/{restart_name}'

def create_restart_nc(dirname, dirnameold):#, groups):
    if not os.path.exists(f'{dirname}/'):
        os.makedirs(f'{dirname}')
    shutil.copy(f'{dirnameold}/field_output.nc', f'{dirname}/field_output.nc')
    # Paths to the original and target files
    input_file = f'{dirnameold}/field_output.nc'
    output_file = f'{dirname}/field_output.nc'
    # new_groups(input_file, output_file, groups, 'D')

# Set up the mesh
if not restart:
    mesh = PeriodicRectangleMesh(nx=nx, ny=ny, Lx=Lx, Ly=Ly, quadrilateral=True)
    output = OutputParameters(dirname=f'/data/home/sh1293/results/jupiter_sw/{folder_name}', dumpfreq=dump_freq, dump_nc=True, checkpoint=True)
elif restart:
    create_restart_nc(dirname=dirname, dirnameold=dirnameold)
    output = OutputParameters(dirname=dirname, dump_nc=True, dumpfreq=dump_freq, checkpoint=True, checkpoint_pickup_filename=f'{dirnameold}/chkpt.h5')
    chkpt_mesh = pick_up_mesh(output, 'firedrake_default')
    mesh = chkpt_mesh


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

def smooth_f_profile(degree, delta, rstar=rstar, Omega=parameters.Omega, R=R, Lx=Lx, nx=nx):
    import sympy as sp

    delta *= Lx/nx
    r = sp.symbols('r')
    fexpr = 2*Omega*(1-0.5*r**2/R**2)
    left_val = fexpr.subs(r, rstar-delta)
    right_val = 2*Omega
    left_diff_val = sp.diff(fexpr, r).subs(r, rstar-delta)
    left_diff2_val = sp.diff(fexpr, r, 2).subs(r, rstar-delta)

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

r, theta = rtheta_from_xy(x, y)

# lon, lat = lonlat_from_rtheta(r, theta)

# Create a spatially varying function for the Coriolis force:
Omega = parameters.Omega
fexpr = 2*Omega*(1-0.5*r**2/R**2)
# ftrap = conditional(r < rstar, fexpr, 2*Omega)
coeffs = smooth_f_profile(degree=smooth_degree, delta=smooth_delta)
fsmooth = float(coeffs[0]) + float(coeffs[1])*r + float(coeffs[2])*r**2 + float(coeffs[3])*r**3
if smooth_degree == 5:
    fsmooth += float(coeffs[4])*r**4 + float(coeffs[5])*r**5

ftrap1 = conditional(r<rstar-smooth_delta*Lx/nx, fexpr, fsmooth)
ftrap = conditional(r<rstar+smooth_delta*Lx/nx, ftrap1, 2*Omega)

if fplane:
    ftrap = 2*Omega

eqns = ShallowWaterEquations(domain, parameters, fexpr=ftrap)
logger.info(f'Estimated number of cores = {eqns.X.function_space().dim() / 50000} \n mpiexec -n nprocs python script.py')

# diagnostic_fields = [SteadyStateError('u'), SteadyStateError('D'),
#                      RelativeVorticity(), PotentialVorticity(),
#                      ShallowWaterKineticEnergy(),
#                      ShallowWaterPotentialEnergy(parameters),
#                      ShallowWaterPotentialEnstrophy(),
#                      CourantNumber()]#, MeridionalComponent('u'), ZonalComponent('u')]

diagnostic_fields = [RelativeVorticity(), PotentialVorticity(),
                    ShallowWaterKineticEnergy(), 
                    ShallowWaterPotentialEnergy(parameters),
                    ShallowWaterPotentialEnstrophy()]#,
if avlpe_diag:
    diagnostic_fields.append(ShallowWaterAvailablePotentialEnergy(parameters))
                    #ShallowWaterAvailablePotentialEnergy(parameters)]

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
        phi_perturb = (Ro/Bu)*exp(1./b)*(b**(-1.+(2./b)))*scipy.special.gammaincc(2./b, (1./b)*(dr/rm)**b)*scipy.special.gamma(2./b)
        # print('append')
        D_values.append(-1 * H * phi_perturb)

    # print('loop done')
    # return list of D values in correct order
    return D_values


if not restart:
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

    if noise:
        pcg = PCG64()
        rg = RandomGenerator(pcg)
        f_normal = rg.normal(VD, 0.0, 1.5e-3*H)
        D0 += f_normal

Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

start_time = time.time()

# Run the timestepper and generate the output.
if not restart:
    stepper.run(t=0, tmax=tmax)
elif restart:
    stepper.run(t=tmin, tmax=tmax, pick_up=True)

if extract_points:
    points = [(136.71875*i, 136.71875*256) for i in range(217, 296)]
    vom = VertexOnlyMesh(mesh, points)
    P0DG = FunctionSpace(vom.input_ordering, "DG", 0)
    PV = stepper.fields('PotentialVorticity')
    PV_at_points = assemble(interpolate(PV, P0DG))

end_time = time.time()

t_start = tmin if restart else 0

logger.info((f'Start time {t_start}'))
logger.info(f'Total time taken {(end_time-start_time):.2f} seconds, {((end_time-start_time)/60**2):.2f} hours')
logger.info(f'File produced:\n{folder_name}')