from gusto import (
    OutputParameters, pick_up_mesh, ShallowWaterParameters, Domain,
    ShallowWaterEquations, logger, RelativeVorticity, PotentialVorticity,
    ShallowWaterKineticEnergy, ShallowWaterPotentialEnergy,
    ShallowWaterPotentialEnstrophy, ShallowWaterAvailablePotentialEnergy,
    SteadyStateError, IO, SubcyclingOptions, TrapeziumRule, SSPRK3,
    DGUpwind, SemiImplicitQuasiNewton, VectorFunctionSpace, assemble,
    Function, WaterVapour, Rain, DG1Limiter, ForwardEuler, ZeroLimiter,
    MixedFSLimiter, SWHeightRelax, MoistConvectiveSWRelativeHumidity,
    xy_from_rtheta, rtheta_from_xy, rtheta_from_lonlat,
    lonlat_from_rtheta, CoriolisOptions, InstantRain, Evaporation
)
from firedrake import (
    SpatialCoordinate, as_vector, pi, interpolate, exp, sqrt,
    PeriodicRectangleMesh, conditional, RandomGenerator, PCG64,
    MoistConvectiveSWRelativeHumidity
)
from stephen_functions import (
    split_number, create_restart_nc, smooth_f_profile, smooth_tophat,
    diffusion_noise_generation
)
import scipy
import numpy as np
import time
import os
import shutil
import pdb
from decimal import Decimal, ROUND_HALF_UP
import sympy as sp

def initialise_D(X, idx):
    # computes the initial depth perturbation corresponding to vortex
    # idx, given coordinates X

    # print('getting coords from list')
    # get the lon, lat coordinates of the centre of this vortex
    lamda_c = south_lon[idx]
    phi_c = south_lat[idx]

    # now want to convert this to x_c and y_c, to make distance calculations easier, as doing it on a plane
    # print('convert to r theta')
    r_c, theta_c = rtheta_from_lonlat(lamda_c, phi_c, R)
    # print('convert to x y')
    x_c, y_c = xy_from_rtheta(r_c, theta_c, Lx/2, Ly/2)

    # make an empty list of D values to append to
    D_values = []

    # loop over X coordinate values
    for Xval in X:
        x, y = Xval
        # calculate distance from centre to this point
        dr = sqrt((x-x_c)**2 + (y-y_c)**2)
        phi_perturb = (Ro/Bu)*exp(1./b)*(b**(-1.+(2./b)))*scipy.special.gammaincc(2./b, (1./b)*(dr/rm)**b)*scipy.special.gamma(2./b)
        D_values.append(-1 * H * phi_perturb)

    # return list of D values in correct order
    return D_values

def sat_func_phys(x_in):
    D = x_in.subfunctions[1]
    return q0*exp(-alpha*D/H)

def sat_func(D):
    return q0*exp(-alpha*D/H)

### options changed in Li 2020
Bu = 1
b = 1.5
Ro = 0.2

### grid parameters
nx = 256
ny = nx
Lx = 7e7
Ly = Lx
rstar = Lx/2 - 3 * Lx/nx

### smoothing parameters
smooth_degree = 5
smooth_delta = 2

### shallow water parameters
g = 24.79
Omega = 1.74e-4
R = 71.4e6

f0 = 2*  Omega
rm = 1e6
vm = Ro * f0 * rm

phi0 = Bu * (f0*rm)**2
H = phi0/g
t_day = 2*pi/Omega

### timing options
dump_freq = 1
dt = 250
tmax = 1*t_day

restart = False
restart_name = ''
t0 = 200*t_day

### vortex locations
south_lat_deg = [90.]#, 83., 83., 83., 83., 83.]#, 70.]
south_lon_deg = [0.]#, 72., 144., 216., 288., 0.]#, 0.]

### noise on depth
noise = False
large_noise = True
noise_amp = 0.05

### noise on moisture
moist_noise = False
moist_noise_amp = 0.01

### coriolis form
coriolisform = 'fulltrap'

### moist variables
L = 2.4e6    # latent heat (V+P)
R = 3745    # gas constant (V+P)
T0 = 165   # reference temperature
alpha = L/(R*T0)   # V+P saturation function constant
gamma = 3900   # V+P condensation feedback term (=Nell's beta1)
xi = 0.1   # how far below saturation we start
cD = 1    # scaling term for evaporation
q0 = 0.5   # scaling such that q0*exp(-alpha)=atmospheric specific humidity in kg/kg=1e-2

### radiative damping
raddamp = True
tau_r = 5 # timescale in Jovian days

### name
setup = 'single' 

###############################################################
if coriolisform == 'fplane':
    fplane = True
    flattrap = False
elif coriolisform == 'flattrap':
    fplane = False
    flattrap = True
elif coriolisform == 'fulltrap':
    fplane = False
    flattrap = False
else:
    logger.info('Incorrect coriolisform option')

tmin = np.ceil(t0/dump_freq)*dump_freq
tmax = np.ceil(tmax/dump_freq)*dump_freq

bint, bdec = split_number(b)
Roint, Rodec = split_number(Ro)
Buint, Budec = split_number(Bu)
taurint, taurdec = split_number(tau_r)
cDint, cDdec = split_number(cD)
if xi < 0:
    xiprefix = 'm'
else:
    xiprefix = ''
q01, q02 = f'{q0:.2e}'.split('e')
q01f = float(q01)
q02i = int(q02)
if q02i < 0:
    q02i = f'm{abs(q02i)}'
q01fint, q01fdec = split_number(q01f)
xi1, xi2 = f'{abs(xi):.2e}'.split('e')
xi1f = float(xi1)
xi2i = int(xi2)
if xi2i < 0:
    xi2i = f'm{abs(xi2i)}'
xi1fint, xi1fdec = split_number(xi1f)

if setup != '':
    setup = f'{setup}_'
if fplane:
    setup = f'{setup}fplane_'
elif flattrap:
    setup = f'{setup}flattrap_'
if noise:
    noiseint, noisedec = split_number(noise_amp)
    if large_noise:
        noisetype = 'lg'
    else:
        noisetype = ''
    noise_name = f'_{noisetype}n{noiseint}{noisedec}'
else:
    noise_name = ''
if moist_noise:
    moist_noiseint, moist_noisedec = split_number(moist_noise_amp)
    moist_noise_name = f'_qn{moist_noiseint}{moist_noisedec}'
else:
    moist_noise_name = ''
moist_name = f'cD{cDint}{cDdec}gamma{gamma}q0{q01fint}{q01fdec}e{q02i}xi{xiprefix}{xi1fint}{xi1fdec}e{xi2i}_'
if raddamp:
    rad_name = f'radt{taurint}{taurdec}'
else:
    rad_name = ''
folder_name = f'{setup}{rad_name}{moist_name}Bu{Buint}{Budec}b{bint}{bdec}Ro{Roint}{Rodec}_l{round(tmax/t_day)}dt{int(dt)}df{dump_freq}{noise_name}{moist_noise_name}'

dirname=f'/data/home/sh1293/results/vp20_moist_jupiter/{folder_name}'
dirnameold=f'/data/home/sh1293/results/vp20_moist_jupiter/{restart_name}'

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

parameters = ShallowWaterParameters(mesh, H=H, g=g, Omega=Omega, cD=cD, R=R,
rotation=CoriolisOptions.gammaplane, x0=Lx/2, y0=Ly/2)

domain = Domain(mesh, dt, "RTCF", 1)

r, theta_coord = rtheta_from_xy(x, y, Lx/2, Ly/2)

_, lat = lonlat_from_rtheta(r, theta_coord, R)

# Create a spatially varying function for the Coriolis force:
Omega_num = Omega
Omega = parameters.Omega
fexpr = 2*Omega*(1-0.5*r**2/R**2)
if flattrap:
    fexpr = 2*Omega*(1-0.5*(rstar-smooth_delta*Lx/nx)**2/R**2)
# ftrap = conditional(r < rstar, fexpr, 2*Omega)
coeffs = smooth_f_profile(degree=smooth_degree, delta=smooth_delta, style='flat' if flattrap else 'polar', rstar=rstar, Omega=Omega_num, R=R, Lx=Lx, nx=nx)
fsmooth = float(coeffs[0]) + float(coeffs[1])*r + float(coeffs[2])*r**2 + float(coeffs[3])*r**3
if smooth_degree == 5:
    fsmooth += float(coeffs[4])*r**4 + float(coeffs[5])*r**5

ftrap = conditional(r<rstar+smooth_delta*Lx/nx, fsmooth, 2*Omega)

if fplane:
    ftrap = 2*Omega

tracers = [WaterVapour(space='DG')]

eqns = ShallowWaterEquations(domain, parameters, coriolis_trap=(rstar-smooth_delta*Lx/nx, ftrap), active_tracers=tracers)

logger.info(f'Estimated number of cores = {eqns.X.function_space().dim() / 50000} \n mpiexec -n nprocs python script.py')

Ld = sqrt(H*g)/f0
logger.info(f'Ld={Ld/1e3:.2f} km')

diagnostic_fields = [RelativeVorticity(), PotentialVorticity(),
                    ShallowWaterKineticEnergy(), 
                    ShallowWaterPotentialEnergy(parameters),
                    ShallowWaterPotentialEnstrophy(),
                    ShallowWaterAvailablePotentialEnergy(parameters),
                    SteadyStateError('D'),
                    MoistConvectiveSWRelativeHumidity(sat_func)
                    ]

io = IO(domain, output=output, diagnostic_fields=diagnostic_fields)

subcycling_options = SubcyclingOptions(subcycle_by_courant=0.33)

transport_methods = [
    DGUpwind(eqns, field_name) for field_name in eqns.field_names
]

DG1limiter = DG1Limiter(domain.spaces('DG'))
zerolimiter = ZeroLimiter(domain.spaces('DG'))
physics_sublimiters = {'water_vapour': zerolimiter,
                        'cloud_water': zerolimiter}
transport_sublimiters = {'water_vapour': DG1limiter,
                        'cloud_water': DG1limiter}
physics_limiter = MixedFSLimiter(eqns, physics_sublimiters)
transport_limiter = MixedFSLimiter(eqns, transport_sublimiters)
transported_fields = [TrapeziumRule(domain, "u"),
                        SSPRK3(domain, "D", subcycling_options=subcycling_options),
                        SSPRK3(domain, "water_vapour", limiter=DG1limiter)]

### physics schemes
height_relax = SWHeightRelax(eqns, H_rel=H, tau_r=tau_r*t_day)

# work: put in condensation and evaporation physics schemes

physics_schemes = [
    (height_relax, ForwardEuler(domain)),
    # work: put in condensation and evaporation
]

stepper = SemiImplicitQuasiNewton(
    eqns, io,
    transport_schemes=transported_fields,
    spatial_methods=transport_methods,
    final_physics_schemes=physics_schemes
)

u0 = stepper.fields("u")
D0 = stepper.fields("D")
wv0 = stepper.fields("water_vapour")
if moist_noise:
    qnoise = Function(wv0.function_space())

south_lat = np.deg2rad(south_lat_deg)
south_lon = np.deg2rad(south_lon_deg)

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
    Dnoise = Function(D0.function_space())
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
        r_c, theta_c = rtheta_from_lonlat(south_lon[i], south_lat[i], R)
        x_c, y_c = xy_from_rtheta(r_c, theta_c, Lx/2, Ly/2)
        dr = sqrt((x-x_c)**2 + (y-y_c)**2)
        mag_veloc = vm * ( dr / rm ) * exp( (1/b) * ( 1 - ( dr / rm )**b ) )
        dx = x - x_c
        dy = y - y_c
        dl = sqrt(dx**2 + dy**2)

        u_veloc += - mag_veloc * (dy / dl)
        v_veloc += mag_veloc * (dx / dl)

    uexpr = as_vector([u_veloc, v_veloc])

    u0.project(uexpr)
    D0.interpolate(Dfinal)
    if noise:
        if not large_noise:
            pcg = PCG64()
            rg = RandomGenerator(pcg)
            f_normal = rg.normal(VD, 0.0, noise_amp*H)
            D0 += f_normal
        elif large_noise:
            large_noise = diffusion_noise_generation(mesh, Lx)
            scaled_noise = large_noise*noise_amp*H/np.max(abs(large_noise.dat.data))
            Dnoise.interpolate(scaled_noise)
            D0 += Dnoise

    initial_msat = sat_func(Dfinal)
    coeffs = smooth_tophat(degree=smooth_degree, delta=smooth_delta, rstar=rstar, Lx=Lx, nx=nx)
    hatsmooth = float(coeffs[0]) + float(coeffs[1])*r + float(coeffs[2])*r**2 + float(coeffs[3])*r**3
    if smooth_degree == 5:
        hatsmooth += float(coeffs[4])*r**4 + float(coeffs[5])*r**5
    tophat1 = conditional(r<rstar-smooth_delta*Lx/nx, 1, hatsmooth)
    tophat = conditional(r<rstar+smooth_delta*Lx/nx, tophat1, 0)
    wvexpr = (1-xi) * initial_msat * tophat
    wv0.interpolate(wvexpr)
    if moist_noise:
        noise = diffusion_noise_generation(mesh, Lx)
        scaled_noise = noise*moist_noise_amp*10/np.max(abs(noise.dat.data)) * tophat
        qnoise.interpolate(scaled_noise)
        wv0 += qnoise
    
Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

start_time = time.time()

if not restart:
    stepper.run(t=0, tmax=tmax)
elif restart:
    stepper.run(t=tmin, tmax=tmax, pick_up=True)

end_time = time.time()

t_start = tmin if restart else 0

logger.info((f'Start time {t_start}'))
logger.info(f'Total time taken {(end_time-start_time):.2f} seconds, {((end_time-start_time)/60**2):.2f} hours')
logger.info(f'File produced:\n{folder_name}')