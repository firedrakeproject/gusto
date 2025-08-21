"""
Set up Martian annular vortex experiment!
"""

from gusto import (
    ShallowWaterParameters, lonlatr_from_xyz, ShallowWaterEquations,
    ContinuityEquation, logger, TrapeziumRule, SSPRK3, DGUpwind,
    PotentialVorticity, ZonalComponent, MeridionalComponent,
    Heaviside_flag_less, Sum, SWCO2cond_flag, CumulativeSum,
    Domain, OutputParameters, pick_up_mesh, Function, IO,
    SWHeightRelax, SWCO2cond, ForwardEuler, SemiImplicitQuasiNewton,
    FunctionSpace, VectorFunctionSpace, xyz_vector_from_lonlatr,
    Constant
)
from firedrake import (
    IcosahedralSphereMesh, SpatialCoordinate, pi, cos,
    interpolate, PCG64, RandomGenerator
)
import numpy as np
from netCDF4 import Dataset
import os
import shutil
import subprocess
import pdb

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

# set inner and outer latitude limits of annulus
phis = 57
phin = 62
phimp = phis

# False means initial vortex is annular, True means it's monopolar
monopolar = False

# scaling for topography term. A0scal = 0 means no topography
A0scal = 0

# scaling factor for PV at pole in annular relaxation profile (defaults 1.6 and 1.0)
pvmax = 2.2
pvpole = 1.05

# tau_r is radiative relaxation time constant
# tau_c is CO2 condensation relaxation time constant
tau_r_ratio = 2
tau_c_ratio = 0.01

# beta is scaling factor for h_th
beta = 1.0

# relaxation schemes can be rad, co2, both, none
rel_sch = 'rad'
include_co2 = 'yes'

# refinement level
ref_lev = 5

# do you want to run from a restart file (True) or not (False). If yes, input the name of the restart file e.g. Free_run/...
restart = True
restart_name = 'Relax_to_annulus/annular_vortex_mars_57-62_PVmax--2-2_PVpole--1-05_tau_r--2sol_A0-0-norel_len-100sols_tracer_tophat-80_ref-5'

# length of this run, time to start from (only relevant if doing a restart)
rundays = 200
start_time = 100
dt = (0.5)**(ref_lev-4) * 450.

# do you want a tracer or not. Edge of tophat function for tracer, north of this the tracer is intialised as 1, south is 0
# if running a restart, True introduces a new tracer whilst False still maintains the old one
tracer = True

hat = False
hat_edge = 80

strip = True
strip_eq_edge = 20
strip_pole_edge = 40

# standard_start is 60-70, 1.6, 1 (i.e. the standard setup I've been playing with). If true then every run starts with this
standard_start = True

# normal dump_freq = 10
dump_freq = 10

# any extra info to include in the directory name
extra_name = ''

# if true then automatically regrid file
regrid = True

#####################################################################################

day = 88774.

if include_co2 == 'no':
    extra_name = f'{extra_name}_no-co2'
if phimp != phis:
    extra_name = f'{extra_name}_phimp--{phimp}'

if tracer:
    if hat and not strip:
        tracername = f'tracer_tophat-{hat_edge}'
    elif strip and not hat:
        tracername = f'tracer_strip-{strip_eq_edge}-{strip_pole_edge}'
elif not tracer:
    tracername = ''

toponame = f'A0-{A0scal}-norel'

dt_dump = dt*dump_freq

# ## max runtime currently 1 day
if restart:
    tmax1 = (rundays + start_time) * day
    tmax = np.ceil(tmax1/dt_dump)*dt_dump
    tmin1 = start_time*day
    tmin = np.ceil(tmin1/dt_dump)*dt_dump
    lenname = f'len-{start_time}-{start_time+rundays}sols'
else:
    tmax1 = rundays * day
    tmax = np.ceil(tmax1/dt_dump)*dt_dump
    lenname = f'len-{rundays}sols'
# ## timestep

refname = f'ref-{ref_lev}'

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

# ## setup shallow water parameters - can also change g and Omega as required
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

    # setup different initial PV profiles
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
    # qt = np.where(rlat > 0., 0.3*qp, qt)
    # qt = np.where(rlat > 60 * np.pi/180, 2.3*qp, qt)

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
    # vorini = zn + f
    uini = un

    return rlat, uini, thini


def new_groups(input_file, output_file, names, base):
    with Dataset(input_file, 'r') as src:
        with Dataset(output_file, 'a') as dst:
            PV_group = src.groups[base]
            for name in names:
                if name not in dst.groups:
                    new_group = dst.createGroup(f'{name}')
                    for var_name, variable in PV_group.variables.items():
                        var_dims = variable.dimensions
                        new_var = new_group.createVariable(var_name, variable.datatype, var_dims)
                        new_var[:] = np.zeros_like(variable[:])
                        for attr in variable.ncattrs():
                            new_var.setncattr(attr, variable.getncattr(attr))


def create_restart_nc(dirpath, dirnameold, groups):
    if not os.path.exists(f'{dirpath}/'):
        os.makedirs(f'{dirpath}')
    shutil.copy(f'{dirnameold}/field_output.nc', f'{dirpath}/field_output.nc')
    # Paths to the original and target files
    input_file = f'{dirnameold}/field_output.nc'
    output_file = f'{dirpath}/field_output.nc'
    new_groups(input_file, output_file, groups, 'D')


def equation_setup(restart, domain, mesh, parameters, Omega, R, A0scal, H):
    x = SpatialCoordinate(mesh)
    fexpr = 2*Omega*x[2]/R
    lamda, theta, _ = lonlatr_from_xyz(x[0], x[1], x[2])
    bexpr = A0scal * H * (cos(theta))**2 * cos(2*lamda)
    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, topog_expr=bexpr)
    tracer_eqn = ContinuityEquation(domain, domain.spaces("DG"), "tracer")
    if restart:
        rs_tracer_eqn = ContinuityEquation(domain, domain.spaces("DG"), "tracer_rs")
    else:
        rs_tracer_eqn = None
    logger.info(f'Estimated number of cores = {eqns.X.function_space().dim() / 50000} \n mpiexec -n nprocs python script.py')

    return eqns, tracer_eqn, rs_tracer_eqn


def transport_setup(restart, domain, eqns, tracer_eqn, rs_tracer_eqn):
    transported_fields = [TrapeziumRule(domain, "u"), SSPRK3(domain, "D")]
    tracer_transport = [(tracer_eqn, SSPRK3(domain))]
    transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D"), DGUpwind(tracer_eqn, "tracer")]
    if restart:
        tracer_transport.append((rs_tracer_eqn, SSPRK3(domain)))
        transport_methods.append(DGUpwind(rs_tracer_eqn, "tracer_rs"))

    return transported_fields, tracer_transport, transport_methods


rlat_an, uini_an, hini_an = initial_profiles(Omega, R, phis, phin, annulus=True, pvpole=pvpole, pvmax=pvmax)
rlat_mp, uini_mp, hini_mp = initial_profiles(Omega, R, phimp, phin, annulus=False, pvmax=pvmax)
rlat, ust, hst = initial_profiles(Omega, R, phiss=60, phinn=70, annulus=True)

if tracer and not restart:
    if hat and not strip:
        Tini = np.where(rlat >= hat_edge*pi/180, 1, 0)
    elif strip and not hat:
        Tini = np.where((strip_eq_edge*pi/180 <= rlat) & (rlat <=strip_pole_edge*pi/180), 1, 0)
elif tracer and restart:
    if hat and not strip:
        Tini_rs = np.where(rlat >= hat_edge*pi/180, 1, 0)
    elif strip and not hat:
        Tini_rs = np.where((strip_eq_edge*pi/180 <= rlat) & (rlat <=strip_pole_edge*pi/180), 1, 0)
elif not tracer:
    Tini = 0
    Tini_rs = 0

if standard_start:
    uini, hini = ust, hst
else:
    if monopolar:
        uini, hini = uini_mp, hini_mp
        phin = 90
    else:
        uini, hini = uini_an, hini_an

if standard_start:
    h_th = min(hst)*beta+H
else:
    h_th = min(hini)*beta+H

# if not restart:
#     # Domain
#     mesh = IcosahedralSphereMesh(radius=R, refinement_level=ref_lev, degree=2)
#     domain = Domain(mesh, dt, 'BDM', degree=1)

#     # Equation, including mountain given by bexpr
#     equation_setup(restart=False)

diagnostic_fields = [PotentialVorticity(), ZonalComponent('u'), MeridionalComponent('u'), Heaviside_flag_less('D', h_th), Sum('D', 'topography'), SWCO2cond_flag('D', h_th), CumulativeSum('CO2cond_flag')]
dumplist = ['D', 'topography', 'tracer']
groups = ['PotentialVorticity', 'u_zonal', 'u_meridional', 'D_minus_H_rel_flag_less', 'tracer', 'D', 'topography', 'D_plus_topography', 'CO2cond_flag', 'CO2cond_flag_cumulative']

# I/O (input/output)
homepath = '/data/home/sh1293/results'
dirnameold = f'{homepath}/{restart_name}'
dirname = f'{rel_sch_folder}/annular_vortex_mars_{phis}-{phin}_{rel_sch_name}_{toponame}_{lenname}_{tracername}_{refname}{extra_name}'
# print(f'directory name is {dirname}')
dirpath = f'{homepath}/{dirname}'

# if restart:
#     if os.path.exists(f'{dirpath}/field_output.nc'):
#         logger.info('File already exists, delete if necessary')
#         exit()

if not restart:
    mesh = IcosahedralSphereMesh(radius=R, refinement_level=ref_lev, degree=2)
    domain = Domain(mesh, dt, 'BDM', degree=1)

    # Equation, including mountain given by bexpr
    eqns, tracer_eqn, rs_tracer_eqn = equation_setup(restart=False, domain=domain, mesh=mesh, parameters=parameters, Omega=Omega, R=R, A0scal=A0scal, H=H)

    output = OutputParameters(dirname=dirpath, dump_nc=True, dumpfreq=dump_freq, checkpoint=True, dumplist=dumplist)

    # Transport schemes
    transported_fields, tracer_transport, transport_methods = transport_setup(restart=False, domain=domain, eqns=eqns, tracer_eqn=tracer_eqn, rs_tracer_eqn=rs_tracer_eqn)
elif restart:
    if tracer:
        dumplist.append('tracer_rs')
        groups.append('tracer_rs')
    if os.path.exists(f'{dirpath}/field_output.nc'):
        logger.info('not making output netcdf')
    else:
        logger.info('making output netcdf')
        create_restart_nc(dirpath=dirpath, dirnameold=dirnameold, groups=groups)
    logger.info('escaped that bit')
    output = OutputParameters(dirname=dirpath, dump_nc=True, dumpfreq=dump_freq, checkpoint=True, checkpoint_pickup_filename=f'{dirnameold}/chkpt.h5', dumplist=dumplist)

    chkpt_mesh = pick_up_mesh(output, 'firedrake_default')
    mesh = chkpt_mesh
    domain = Domain(mesh, dt, 'BDM', degree=1)

    # Equation, including mountain given by bexpr
    eqns, tracer_eqn, rs_tracer_eqn = equation_setup(restart=True, domain=domain, mesh=mesh, parameters=parameters, Omega=Omega, R=R, A0scal=A0scal, H=H)

    # Transport schemes
    transported_fields, tracer_transport, transport_methods = transport_setup(restart=True, domain=domain, eqns=eqns, tracer_eqn=tracer_eqn, rs_tracer_eqn=rs_tracer_eqn)

io = IO(domain, output, diagnostic_fields=diagnostic_fields)

H_rel = Function(domain.spaces('L2'))
height_relax = SWHeightRelax(eqns, H_rel, tau_r=tau_r)

# pdb.set_trace()
co2_cond = SWCO2cond(eqns, h_th=h_th, tau_c=tau_c)

if rel_sch == 'both':
    physics_schemes = [(height_relax, ForwardEuler(domain)), (co2_cond, ForwardEuler(domain))]
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


def initial_u(X, u):
    lats = []
    for X0 in X:
        x, y, z = X0
        _, lat, _ = lonlatr_from_xyz(x, y, z)
        lats.append(lat)
    return np.interp(np.array(lats), rlat, u)


def initial_D(X, h):
    lats = []
    for X0 in X:
        x, y, z = X0
        _, lat, _ = lonlatr_from_xyz(x, y, z)
        lats.append(lat)
    return np.interp(np.array(lats), rlat, h)


def initial_T(X, Tini):
    lats = []
    for X0 in X:
        x, y, z = X0
        _, lat, _ = lonlatr_from_xyz(x, y, z)
        lats.append(lat)
    return np.interp(np.array(lats), rlat, Tini)


D0_mp = Function(domain.spaces('L2'))
# D0_mp.assign(D0)
D0_an = Function(domain.spaces('L2'))
if not restart:
    u0 = stepper.fields('u')
    D0 = stepper.fields('D')
    T0 = stepper.fields('tracer')

    Vu = FunctionSpace(mesh, "DG", 2)
    uzonal = Function(Vu)
    umesh = Vu.mesh()
    Wu = VectorFunctionSpace(umesh, Vu.ufl_element())
    Xu = interpolate(umesh.coordinates, Wu)
    uzonal.dat.data[:] = initial_u(Xu.dat.data_ro, uini)
    X = SpatialCoordinate(mesh)
    u0.project(xyz_vector_from_lonlatr(uzonal, Constant(0), Constant(0), X))

    VT = T0.function_space()
    Tmesh = VT.mesh()
    WT = VectorFunctionSpace(Tmesh, VT.ufl_element())
    XT = interpolate(Tmesh.coordinates, WT)
    T0.dat.data[:] = initial_T(XT.dat.data_ro, Tini)

    # f_init = exp(-(x[1]/1e6)**2-(x[0]/1e6)**2)
    # f_init = 1
    # T0.interpolate(f_init)

    VD = D0.function_space()
    Dmesh = VD.mesh()
    WD = VectorFunctionSpace(Dmesh, VD.ufl_element())
    XD = interpolate(Dmesh.coordinates, WD)
    D0.dat.data[:] = initial_D(XD.dat.data_ro, hini)
    D0 += H

elif restart:
    T0 = stepper.fields('tracer_rs')
    VT = T0.function_space()
    Tmesh = VT.mesh()
    WT = VectorFunctionSpace(Tmesh, VT.ufl_element())
    XT = interpolate(Tmesh.coordinates, WT)
    T0.dat.data[:] = initial_T(XT.dat.data_ro, Tini_rs)

    VD = D0_mp.function_space()
    Dmesh = VD.mesh()
    WD = VectorFunctionSpace(Dmesh, VD.ufl_element())
    XD = interpolate(Dmesh.coordinates, WD)

D0_mp.dat.data[:] = initial_D(XD.dat.data_ro, hini_mp)
D0_mp += H
D0_an.dat.data[:] = initial_D(XD.dat.data_ro, hini_an)
D0_an += H

# from firedrake import File
# mp_ic = File("mp_ic.pvd")
# mp_ic.write(D0_mp)


# hinit = Function(D0.function_space()).interpolate(D0/H -1)
# from firedrake import File
# of = File(f'{dirname}_H/out.pvd')
# of.write(hinit)
if not restart:
    pcg = PCG64()
    rg = RandomGenerator(pcg)
    f_normal = rg.normal(VD, 0.0, 1.5e-3*H)
    D0 += f_normal

if rel_sch == 'both':
    H_rel.assign(D0_mp)
elif rel_sch == 'rad':
    H_rel.assign(D0_an)
    # H_rel.assign(H)

# print(max(f_normal.dat.data))
# print(min(f_normal.dat.data))

Dbar = Function(D0_mp.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

# # confirm the dirname is correct
# if not restart:
#     print(f'Directory name is {dirname}\n\n Input \'y\' to continue')
#     confirm = input()
# elif restart:
#     print(f'Directory name is {dirname},\n old run is {dirnameold}\n\n Input \'y\' to continue')
#     confirm = input()

# # ------------------------------------------------------------------------ #
# # Run
# # ------------------------------------------------------------------------ #

# if confirm == 'y':
#     if not restart:
#         stepper.run(t=0, tmax=tmax)
#     elif restart:
#         print('restart')
#         stepper.run(t=start_time*day, tmax=tmax, pick_up=True)
# else:
#     print('Confirmation not given')


if not restart:
    logger.info(f'Directory name is {dirname}')
    stepper.run(t=0, tmax=tmax)
elif restart:
    logger.info(f'Restart from {dirnameold}')
    logger.info(f'Directory name is {dirname}')
    stepper.run(t=tmin, tmax=tmax, pick_up=True)


print(f'directory name is {dirname}')

if regrid:
    print(f'regridding, {dirname}, {ref_lev}')
    subprocess.run(['python', '/data/home/sh1293/firedrake-real-opt_oct24/src/gusto/plotting/my_plots/regrid.py', dirname, str(ref_lev)])