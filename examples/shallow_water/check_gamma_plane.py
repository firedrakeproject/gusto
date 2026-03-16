from gusto import(
    OutputParameters, ShallowWaterParameters, Domain, ShallowWaterEquations,
    logger, PotentialVorticity, IO, SubcyclingOptions, TrapeziumRule,
    SSPRK3, DGUpwind, SemiImplicitQuasiNewton, Function,
    xy_from_rtheta, rtheta_from_xy, rtheta_from_lonlat, lonlat_from_rtheta,
    CoriolisOptions
)
from firedrake import (
    SpatialCoordinate, VertexOnlyMesh, as_vector, pi, interpolate, exp, sqrt,
    PeriodicRectangleMesh, conditional
)
import scipy
import numpy as np
import time
import os
import shutil
import sympy as sp

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
    return coeffs

folder_name_suffix = 'smooth-trap-232dg'

nx = 256
ny = nx
Lx = 7e7
Ly = Lx

rstar = Lx/2-3*Lx/nx
smooth_delta = 2

Bu = 1
g=24.79
Omega = 1.74e-4
R = 71.4e6
f0 = 2 * Omega
rm = 1e6
phi0 = Bu * (f0*rm)**2
H = phi0/g

smooth_degree = 5
smooth_delta = 2

dt = 250
tmax = 100*dt

dirname=f'/data/home/sh1293/results/jupiter_sw/check_gamma_plane_{folder_name_suffix}'

mesh = PeriodicRectangleMesh(nx=nx, ny=ny, Lx=Lx, Ly=Ly, quadrilateral=True)
output = OutputParameters(dirname=f'{dirname}', dumpfreq=1, dump_nc=True)

parameters = ShallowWaterParameters(mesh, H=H, Omega=Omega, R=R,
                                    rotation=CoriolisOptions.gammaplane)

domain = Domain(mesh, dt, "RTCF", 1)


x, y = SpatialCoordinate(mesh)
Lxmesh = mesh.coordinates.dat.data[:, 0].max()
Lymesh = mesh.coordinates.dat.data[:, 1].max()
rmesh, _ = rtheta_from_xy(x, y, Lxmesh/2, Lymesh/2)
Rsq = parameters.R**2
fexpr = 2*parameters.Omega * (1 - 0.5 * rmesh**2 / Rsq)
# from firedrake import FunctionSpace
# Vcg = FunctionSpace(domain.mesh, "DG", 1)
# rfunc = Function(Vcg).interpolate(fexpr)
# from firedrake.output import VTKFile
# outfile = VTKFile(f'{dirname}/r_output.pvd')
# outfile.write(rfunc)


### rstar and smooth_delta are so the trap can be in the same place as my big script

Omega_num = Omega
Omega = parameters.Omega
fexpr = 2*Omega*(1-0.5*rmesh**2/R**2)
# ftrap = conditional(r < rstar, fexpr, 2*Omega)
coeffs = smooth_f_profile(degree=smooth_degree, delta=smooth_delta, style='polar', rstar=rstar, Omega=Omega_num, R=R, Lx=Lx, nx=nx)
fsmooth = float(coeffs[0]) + float(coeffs[1])*rmesh + float(coeffs[2])*rmesh**2 + float(coeffs[3])*rmesh**3
if smooth_degree == 5:
    fsmooth += float(coeffs[4])*rmesh**4 + float(coeffs[5])*rmesh**5

# ftrap1 = conditional(r<rstar-smooth_delta*Lx/nx, fexpr, fsmooth)
# ftrap = conditional(r<rstar+smooth_delta*Lx/nx, ftrap1, 2*Omega)#-2*Omega

ftrap = conditional(rmesh<rstar+smooth_delta*Lx/nx, fsmooth, 2*Omega)

# eqns = ShallowWaterEquations(domain, parameters, coriolis_trap=(rstar-smooth_delta*Lx/nx, 2*Omega))
eqns = ShallowWaterEquations(domain, parameters, coriolis_trap=(rstar-smooth_delta*Lx/nx, ftrap))

diagnostic_fields = [PotentialVorticity()]

io = IO(domain, output=output, diagnostic_fields=diagnostic_fields)

subcycling_options = SubcyclingOptions(subcycle_by_courant=0.33)

transport_methods = [
    DGUpwind(eqns, field_name) for field_name in eqns.field_names
]
transported_fields = [TrapeziumRule(domain, "u"),
                        SSPRK3(domain, "D", subcycling_options=subcycling_options)]
stepper = SemiImplicitQuasiNewton(
    eqns, io, transported_fields, transport_methods
)

u0 = stepper.fields("u")
# Dexpr = H * (1-rmesh**2/(5e7)**2)
D0 = stepper.fields("D")
coriolis = stepper.fields("coriolis")

u0.assign(0.)
# D0.interpolate(coriolis)
D0.assign(H)
Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

stepper.run(t=0, tmax=tmax)

logger.info(f'File produced:\ncheck_gamma_plane_{folder_name_suffix}')