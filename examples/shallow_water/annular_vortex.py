"""
Set up Martian annular vortex experiment!
"""

from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, pi, sqrt, min_value, sin, cos,
                       interpolate)
import numpy as np
import matplotlib.pyplot as plt
#import xarray as xr

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

day = 88774.

# set inner and outer latitude limits of annulus   
phis = 80
phin = 85


### max runtime currently 1 day
tmax = 20 * day
### timestep
dt = 450.

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

# Domain
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=4, degree=2)
x = SpatialCoordinate(mesh)
domain = Domain(mesh, dt, 'BDM', 1)

# Equation, including mountain given by bexpr
fexpr = 2*Omega*x[2]/R
eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr)

# I/O (input/output)
dirname = f'annular_vortex_mars_{phis}-{phin}_noise'
output = OutputParameters(dirname=dirname, dump_nc=True)
diagnostic_fields = [PotentialVorticity(), ZonalComponent('u'), MeridionalComponent('u')]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
transported_fields = [TrapeziumRule(domain, "u"),
                      SSPRK3(domain, "D")]
transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]

# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields, transport_methods)

# ------------------------------------------------------------------------ #
# Initial conditions - these need changing!
# ------------------------------------------------------------------------ #

u0 = stepper.fields('u')
D0 = stepper.fields('D')

def initial_profiles(omega, radius):

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
    rlat1 = np.radians(phis)
    rlat2 = np.radians(phin)
    qp = 2 * omega / hbart
    qt0 = 2 * omega * sinlat / hbart
    qt = qt0

    # Will setup - annulus
    qt = np.where(rlat > 0., 0.3 * qp, qt)
    qt = np.where(rlat > rlat1, 1.6 * qp, qt)
    qt = np.where(rlat > rlat2, qp, qt)
    
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
        rlat2l = lat_in_rlat(rlat2 - lim, rlat)
        rlat2u = lat_in_rlat(rlat2 + lim, rlat)

        q_smooth0 = np.interp(rlat, [rlat0l, rlat0u], [q_at_lat(rlat0l, qt), q_at_lat(rlat0u, qt)])
        q_smooth1 = np.interp(rlat, [rlat1l, rlat1u], [q_at_lat(rlat1l, qt), q_at_lat(rlat1u, qt)])
        q_smooth2 = np.interp(rlat, [rlat2l, rlat2u], [q_at_lat(rlat2l, qt), q_at_lat(rlat2u, qt)])

        qt = np.where((rlat0l <= rlat) & (rlat <= rlat0u), q_smooth0, qt)
        qt = np.where((rlat1l <= rlat) & (rlat <= rlat1u), q_smooth1, qt)
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
        error = np.sum(np.sqrt((qn - qt)**2))
        count += 1
        print(count, error)

    # final values and constants corrected
    thini = (hn - hbart) * phibar
    vorini = zn + f
    uini = un


    # make random noise
    sd = 1.5e-3 * H
    noise = np.random.normal(loc=0, scale=sd, size=np.size(sd))
    thini += noise


    fig, axs = plt.subplots(3, 1, sharex=True, figsize = (6,9))
    axs[0].plot(rlat, qt0, '--', color = 'black', alpha = 0.5)
    axs[0].plot(rlat, qt, color = 'blue', label = 'target')
    axs[0].plot(rlat, qn, '--', color='red', label = 'numerical')
    axs[0].legend()
    axs[0].set_ylabel('q')
    axs[1].plot(rlat, [0]*len(rlat), '--', color = 'black', alpha = 0.5)
    axs[1].plot(rlat, thini/phibar, color = 'blue')
    axs[1].set_ylabel('h/H-1')
    axs[2].plot(rlat, [0]*len(rlat), '--', color = 'black', alpha = 0.5)
    axs[2].plot(rlat, un, color = 'blue')
    axs[2].set_ylabel('u')
    fig.tight_layout()
    plt.show()
    plt.savefig('/data/home/sh1293/firedrake-real-opt/src/gusto/examples/shallow_water/results/%s.pdf' %(dirname))

    return rlat, uini, thini

rlat, uini, hini = initial_profiles(Omega, R)

#ic = xr.Dataset(data_vars=dict(u=(['rlat'], uini), h=(['rlat'], hini)), coords=dict(lat=rlat))
#ic.to_netcdf('/data/home/sh1293/firedrake-real-opt/src/gusto/examples/shallow_water/results/%s.nc' %(dirname))

def initial_u(X):
    lats = []
    for X0 in X:
        x, y, z = X0
        _, lat, _ = lonlatr_from_xyz(x, y, z)
        lats.append(lat)
    return np.interp(np.array(lats), rlat, uini)

def initial_D(X):
    lats = []
    for X0 in X:
        x, y, z = X0
        _, lat, _ = lonlatr_from_xyz(x, y, z)
        lats.append(lat)
    return np.interp(np.array(lats), rlat, hini)


Vu = FunctionSpace(mesh, "DG", 2)
uzonal = Function(Vu)
umesh = Vu.mesh()
Wu = VectorFunctionSpace(umesh, Vu.ufl_element())
Xu = interpolate(umesh.coordinates, Wu)
uzonal.dat.data[:] = initial_u(Xu.dat.data_ro)
X = SpatialCoordinate(mesh)
u0.project(xyz_vector_from_lonlatr(uzonal, Constant(0), Constant(0), X))

VD = D0.function_space()
Dmesh = VD.mesh()
WD = VectorFunctionSpace(umesh, VD.ufl_element())
XD = interpolate(Dmesh.coordinates, WD)
D0.dat.data[:] = initial_D(XD.dat.data_ro)

D0 += H
#hinit = Function(D0.function_space()).interpolate(D0/H -1)
#from firedrake import File
#of = File(f'{dirname}_H/out.pvd')
#of.write(hinit)

Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)
