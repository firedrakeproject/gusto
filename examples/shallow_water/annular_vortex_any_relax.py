"""
Set up Martian annular vortex experiment!
"""

from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, pi, sqrt, min_value, sin, cos,
                       interpolate, PCG64, RandomGenerator)
import numpy as np
from netCDF4 import Dataset
#import matplotlib.pyplot as plt
#import xarray as xr

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

day = 88774.

# set inner and outer latitude limits of annulus   
phis = 60
phin = 70
phimp = phis

# False means initial vortex is annular, True means it's monopolar
monopolar = False

# scaling for topography term. A0scal = 0 means no topography
A0scal = 0

# scaling factor for PV at pole in annular relaxation profile (defaults 1.6 and 1.0)
pvmax = 1.6
pvpole = 1.0

# tau_r is radiative relaxation time constant
# tau_c is CO2 condensation relaxation time constant
tau_r_ratio = 2
tau_c_ratio = 0.01

# beta is scaling factor for h_th
beta = 1.0

# relaxation schemes can be rad, co2, both, none
rel_sch = 'none'
include_co2 = 'yes'

extra_name = ''
if include_co2 == 'no':
    extra_name = f'{extra_name}_no-co2'
if phimp != phis:
    extra_name = f'{extra_name}_phimp--{phimp}'

# if toporel:
#     toponame = f'A0-{A0scal}-rel'
# else:
toponame = f'A0-{A0scal}-norel'

### max runtime currently 1 day
rundays = 10
tmax = rundays * day
### timestep
dt = 450.

pvpoleint = str(pvpole).split('.')[0]
pvpoledec = str(pvpole).split('.')[1]

pvmaxint = str(pvmax).split('.')[0]
pvmaxdec = str(pvmax).split('.')[1]


betaint = str(beta).split('.')[0]
betadec = str(beta).split('.')[1]

if rel_sch == 'both':
    rel_sch_name = f'tau_r--{tau_r_ratio}sol_tau_c--{tau_c_ratio}sol_beta--{betaint}-{betadec}'
    rel_sch_folder = 'Relax_to_pole_and_CO2'
elif rel_sch == 'co2':
    rel_sch_name = f'tau_c--{tau_c_ratio}sol_alpha--{betaint}-{betadec}'
    rel_sch_folder = 'CO2'
elif rel_sch == 'rad':
    rel_sch_name = f'PVmax--{pvmaxint}-{pvmaxdec}_PVpole--{pvpoleint}-{pvpoledec}_tau_r--{tau_r_ratio}sol'
    rel_sch_folder = 'Relax_to_annulus'
elif rel_sch == 'none':
    rel_sch_name = 'free'
    rel_sch_folder = 'Free_run'

tau_r = tau_r_ratio * day
tau_c = tau_c_ratio * day

# setup shallow water parameters
R = 3396000.    # Mars value (3389500)
H = 17000.      # Will's Mars value
Omega = 2*np.pi/88774.
g = 3.71

### setup shallow water parameters - can also change g and Omega as required
parameters = ShallowWaterParameters(g=g, H=H, Omega=Omega)
# ------------------------------------------------------------------------ #
# Set up model objects
# ------------------------------------------------------------------------ #

def initial_profiles(omega, radius, phiss, phinn, annulus, **kwargs):

    pvpole = kwargs.pop('pvpole', 1)
    pvmax = kwargs.pop('pvmax', 1.6)

    # set numerical method parameters
    ny = 1000000
    tol = 1e-10
    alpha = 0.5

    # target hbart and phibar - set to 1 to match Richard Scott?
    hbart = 1
    phibar = 17000


    # set up latitude array, and sin and cos of lat
    rlat = np.linspace(-np.pi/2, np.pi/2, num=ny)[1:-1]
    sinlat = np.sin(rlat)
    coslat = np.cos(rlat)
    da = coslat * np.pi/ny
    hn = np.ones(len(rlat))
    f = 2 * omega * sinlat

    #setup different initial PV profiles
    smoothing = True
    rlat1 = np.radians(phiss)
    rlat2 = np.radians(phinn)
    qp = 2 * omega / hbart
    qt0 = 2 * omega * sinlat / hbart
    qt = qt0

    # Will setup - annulus
    qt = np.where(rlat > 0., 0.3 * qp, qt)
    qt = np.where(rlat > rlat1, pvmax * qp, qt)
    if annulus:
        qt = np.where(rlat > rlat2, pvpole * qp, qt)
    
    if smoothing:
        # annulus smoothing - linearly for +-1.5deg around each boundary
        def lat_in_rlat(val, rlat):
            x = np.where(rlat > val, rlat, np.nan)
            x = x[~np.isnan(x)]
            return x[0]

        def q_at_lat(val, q):
            x = np.where(rlat == val, q, np.nan)
            x = x[~np.isnan(x)]
            return x[0]

        lim = np.radians(1.5)
        rlat0l = lat_in_rlat(-lim, rlat)
        rlat0u = lat_in_rlat(lim, rlat)
        rlat1l = lat_in_rlat(rlat1 - lim, rlat)
        rlat1u = lat_in_rlat(rlat1 + lim, rlat)
        if annulus:
            rlat2l = lat_in_rlat(rlat2 - lim, rlat)
            rlat2u = lat_in_rlat(rlat2 + lim, rlat)

        q_smooth0 = np.interp(rlat, [rlat0l, rlat0u], [q_at_lat(rlat0l, qt), q_at_lat(rlat0u, qt)])
        q_smooth1 = np.interp(rlat, [rlat1l, rlat1u], [q_at_lat(rlat1l, qt), q_at_lat(rlat1u, qt)])
        if annulus:
            q_smooth2 = np.interp(rlat, [rlat2l, rlat2u], [q_at_lat(rlat2l, qt), q_at_lat(rlat2u, qt)])

        qt = np.where((rlat0l <= rlat) & (rlat <= rlat0u), q_smooth0, qt)
        qt = np.where((rlat1l <= rlat) & (rlat <= rlat1u), q_smooth1, qt)
        if annulus:
            qt = np.where((rlat2l <= rlat) & (rlat <= rlat2u), q_smooth2, qt)

    # Scott Liu setup
    #qt = np.where(rlat > 0., 0.3*qp, qt)
    #qt = np.where(rlat > 60 * np.pi/180, 2.3*qp, qt)

    count = 0
    error = 1
    while error > tol:

        # constant for global zeta integral to be 0
        ctop = np.sum(qt * hn * da)
        cbot = np.sum(hn * da)

        coff = -ctop / cbot

        # calculate zeta
        if count >= 1:
            zn0 = zn
        zn = (coff + qt) * hn - f
        if count >= 1:
            zn = alpha * zn + (1 - alpha) * zn0

        # u as an integral of zeta
        un = - np.cumsum(zn * da) * radius / coslat

        # dh/dmu (mu = sinlat) from calculated u
        dhdmu = - (un / coslat + 2 * omega * radius) * un * sinlat / (coslat * phibar) * 1 / g

        # h as an integral of dh/dhmu
        hn = np.cumsum(dhdmu * da)

        # change average value of h to match the target average value
        havgn = np.sum(hn * da) / (np.sum(da))
        hn = hn - havgn + hbart

        # numerical PV value from newly calculated values, with correction constant
        qn = (f + zn)/hn - coff

        # compare PV profiles and compare to tolerance threshold
        error = np.sqrt(np.sum((qn - qt)**2))
        count += 1
        # print(count, error)

    # final values and constants corrected
    thini = (hn - hbart) * phibar
    vorini = zn + f
    uini = un


    # make random noise
    #sd = 1.5e-3 * H
    #noise = np.random.normal(loc=0, scale=sd, size=np.size(sd))
    #thini += noise


    #fig, axs = plt.subplots(3, 1, sharex=True, figsize = (6,9))
    #axs[0].plot(rlat, qt0, '--', color = 'black', alpha = 0.5)
    #axs[0].plot(rlat, qt, color = 'blue', label = 'target')
    #axs[0].plot(rlat, qn, '--', color='red', label = 'numerical')
    #axs[0].set_ylabel('q')
    #axs[1].plot(rlat, [0]*len(rlat), '--', color = 'black', alpha = 0.5)
    #axs[0].legend()
    #axs[1].plot(rlat, thini/phibar, color = 'blue')
    #axs[1].set_ylabel('h/H-1')
    #axs[2].plot(rlat, [0]*len(rlat), '--', color = 'black', alpha = 0.5)
    #axs[2].plot(rlat, un, color = 'blue')
    #axs[2].set_ylabel('u')
    #fig.tight_layout()
    #plt.show()
    #plt.savefig('/data/home/sh1293/firedrake-real-opt/src/gusto/examples/shallow_water/results/%s.pdf' %(dirname))

    return rlat, uini, thini

rlat, uini, hini = initial_profiles(Omega, R, phis, phin, annulus=True, pvpole=pvpole, pvmax=pvmax)
rlat_mp, uini_mp, hini_mp = initial_profiles(Omega, R, phimp, phin, annulus=False)
h_th = min(hini)*beta+H

if monopolar:
    rlat, uini, hini = rlat_mp, uini_mp, hini_mp
    phin = 90


# Domain
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=4, degree=2)
x = SpatialCoordinate(mesh)
domain = Domain(mesh, dt, 'BDM', degree=1)

# Equation, including mountain given by bexpr
fexpr = 2*Omega*x[2]/R
lamda, theta, _ = lonlatr_from_xyz(x[0], x[1], x[2])
bexpr = A0scal * H * (cos(theta))**2 * cos(2*lamda)
eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, bexpr=bexpr)
tracer_eqn = AdvectionEquation(domain, domain.spaces("DG"), "tracer")

# estimate core count for Pileus
print(f'Estimated number of cores = {eqns.X.function_space().dim() / 50000} ')

H_rel = Function(domain.spaces('L2'))

# I/O (input/output)
homepath = '/data/home/sh1293/results'
dirname = f'{rel_sch_folder}/annular_vortex_mars_{phis}-{phin}_{rel_sch_name}_{toponame}_len-{rundays}sols{extra_name}'
print(f'directory name is {dirname}')
dirpath = f'{homepath}/{dirname}'
output = OutputParameters(dirname=dirpath, dump_nc=True, dumpfreq=10, checkpoint=True)
diagnostic_fields = [PotentialVorticity(), ZonalComponent('u'), MeridionalComponent('u'), Heaviside_flag_less('D', h_th), TracerDensity('tracer', 'tracer')]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
transported_fields = [TrapeziumRule(domain, "u"),
                      SSPRK3(domain, "D")]
tracer_transport = [(tracer_eqn, SSPRK3(domain))]
transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D"), DGUpwind(tracer_eqn, "tracer")]

H_rel = Function(domain.spaces('L2'))
height_relax = SWHeightRelax(eqns, H_rel, tau_r=tau_r)

co2_cond = SWCO2cond(eqns, h_th=h_th, tau_c=tau_c)

if rel_sch == 'both':
    physics_schemes = [(height_relax, ForwardEuler(domain)),
                        (co2_cond, ForwardEuler(domain))]
    if include_co2 == 'no':
        physics_schemes = [(height_relax, ForwardEuler(domain))]
elif rel_sch == 'co2':
    physics_schemes = [(co2_cond, ForwardEuler(domain))]
elif rel_sch == 'rad':
    physics_schemes = [(height_relax, ForwardEuler(domain))]
elif rel_sch == 'none':
    physics_schemes = []

# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields, transport_methods, physics_schemes=physics_schemes, 
                                    auxiliary_equations_and_schemes=tracer_transport,
                                    num_outer=2, num_inner=2)

# ------------------------------------------------------------------------ #
# Initial conditions - these need changing!
# ------------------------------------------------------------------------ #

u0 = stepper.fields('u')
D0 = stepper.fields('D')
D0_mp = Function(domain.spaces('L2'))
D0_mp.assign(D0)
tracer0 = stepper.fields('tracer')



#ic = xr.Dataset(data_vars=dict(u=(['rlat'], uini), h=(['rlat'], hini)), coords=dict(lat=rlat))
#ic.to_netcdf('/data/home/sh1293/firedrake-real-opt/src/gusto/examples/shallow_water/results/%s.nc' %(dirname))

def initial_u(X):
    lats = []
    for X0 in X:
        x, y, z = X0
        _, lat, _ = lonlatr_from_xyz(x, y, z)
        lats.append(lat)
    return np.interp(np.array(lats), rlat, uini)

def initial_D(X, h):
    lats = []
    for X0 in X:
        x, y, z = X0
        _, lat, _ = lonlatr_from_xyz(x, y, z)
        lats.append(lat)
    return np.interp(np.array(lats), rlat, h)


Vu = FunctionSpace(mesh, "DG", 2)
uzonal = Function(Vu)
umesh = Vu.mesh()
Wu = VectorFunctionSpace(umesh, Vu.ufl_element())
Xu = interpolate(umesh.coordinates, Wu)
uzonal.dat.data[:] = initial_u(Xu.dat.data_ro)
X = SpatialCoordinate(mesh)
u0.project(xyz_vector_from_lonlatr(uzonal, Constant(0), Constant(0), X))

uspace = u0.function_space()
pcg = PCG64()
rg = RandomGenerator(pcg)
#f_normal = rg.normal(uspace, 0.0, 1.5)
#u0 += f_normal

tracer0.interpolate(1)



VD = D0.function_space()
Dmesh = VD.mesh()
WD = VectorFunctionSpace(umesh, VD.ufl_element())
XD = interpolate(Dmesh.coordinates, WD)
D0.dat.data[:] = initial_D(XD.dat.data_ro, hini)
D0_mp.dat.data[:] = initial_D(XD.dat.data_ro, hini_mp)
D0 += H
D0_mp += H

# from firedrake import File
# mp_ic = File("mp_ic.pvd")
# mp_ic.write(D0_mp)


#hinit = Function(D0.function_space()).interpolate(D0/H -1)
#from firedrake import File
#of = File(f'{dirname}_H/out.pvd')
#of.write(hinit)
pcg = PCG64()
rg = RandomGenerator(pcg)
f_normal = rg.normal(VD, 0.0, 1.5e-3*H)
if rel_sch == 'both':
    H_rel.assign(D0_mp)
elif rel_sch == 'rad':
    H_rel.assign(D0)
    # H_rel.assign(H)
D0 += f_normal


#print(max(f_normal.dat.data))
#print(min(f_normal.dat.data))

Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)


# results_file_name = f'{dirname}/field_output.nc'
# output_file_name = f'{dirname}/regrid_output.nc'
# data_file = Dataset(results_file_name, 'r')
# for field_name in ['D', 'D_minus_H_rel_flag_less', 'u_meridional', 'u_zonal', 'PotentialVorticity']:
#     field_data = extract_gusto_field(data_file, field_name)
#     coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
#     times = np.arange(np.shape(field_data)[1])
#     lats = np.arange(-90, 91, 3)
#     lons = np.arange(-180, 181, 3)
#     X, Y = np.meshgrid(lons, lats)
#     new_data = regrid_horizontal_slice(X, Y,
#                                         coords_X, coords_Y, field_data)
#     da = xr.DataArray(data=new_data.astype('float32'),
#                     dims=['lat', 'lon', 'time'],
#                     coords=dict(lat=lats.astype('float32'), lon=lons.astype('float32'), time=times.astype('float32')),
#                     name=field_name)
#     ds1 = da.to_dataset()
#     if field_name == 'D':
#         ds = ds1
#     else:
#         ds = xr.merge([ds, ds1])

# ds.to_netcdf(output_file_name)


print(f'directory name is {dirname}')