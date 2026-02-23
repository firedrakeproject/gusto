from firedrake import (
    pi, atan2, sqrt, cos, sin, logger, OutputParameters, Domain,
    DiffusionParameters, BackwardEuler, CGDiffusion, IO,
    Timestepper, PCG64, RandomGenerator, DiffusionEquation
)
import sympy as sp
import os
import shutil


def split_number(x):
    x = float(abs(x))
    xint, xdec = str(x).split('.')
    if xint == '0':
        xint = ''
    if xdec == '0':
        xdec = ''
    else:
        xdec = f'p{xdec}'
    return xint, xdec

def create_restart_nc(dirname, dirnameold):#, groups):
    if not os.path.exists(f'{dirname}/'):
        os.makedirs(f'{dirname}')
    shutil.copy(f'{dirnameold}/field_output.nc', f'{dirname}/field_output.nc')
    # Paths to the original and target files
    input_file = f'{dirnameold}/field_output.nc'
    output_file = f'{dirname}/field_output.nc'
    # new_groups(input_file, output_file, groups, 'D')

def rtheta_from_xy(x, y, Lx, Ly, angle_units='rad'):
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

def lonlat_from_rtheta(r, theta, R, angle_units='rad', pole='north'):
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

def rtheta_from_lonlat(lon, lat, R, angle_units='rad', pole='north'):
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

def xy_from_rtheta(r, theta, Lx, Ly, angle_units='rad'):
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

def smooth_tophat(degree, delta, rstar, Lx, nx):
    delta *= Lx/nx
    r = sp.symbols('r')
    left_val = 1
    right_val = 0
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
    #     (1, r<rstar-delta),
    #     (P_smooth, (rstar-delta<=r) & (r<=rstar+delta)),
    #     (0, rstar+delta<r)
    # )
    return coeffs

def diffusion_noise_generation(mesh, Lx):

    mesh = mesh
    Lx = Lx
    factor = Lx/10

    dt = 0.01*factor
    tmax = 0.2*factor

    kappa = 1.*factor
    # mu = 5.
    logger.info('Generating noise')
    output = OutputParameters(dump_vtus=False, dump_diagnostics=False)
    domain = Domain(mesh, dt, "RTCF", 1)

    V = domain.spaces("H1")

    diffusion_params = DiffusionParameters(domain.mesh, kappa=kappa)
    eqn = DiffusionEquation(domain, V, "f", diffusion_parameters=diffusion_params)
    diffusion_scheme = BackwardEuler(domain)
    diffusion_methods = [CGDiffusion(eqn, "f", diffusion_params)]
    io = IO(domain, output=output)
    timestepper = Timestepper(eqn, diffusion_scheme, io, spatial_methods=diffusion_methods)

    f0 = timestepper.fields("f")
    pcg = PCG64()
    rg = RandomGenerator(pcg)
    noise_init = rg.normal(V, 0.0, 1.)

    # x = SpatialCoordinate(mesh)
    # noise_init = exp(-((x[0]-0.5*L)**2 + (x[1]-0.5*L)**2))

    f0.interpolate(noise_init)
    timestepper.run(0., tmax)

    return timestepper.fields("f")