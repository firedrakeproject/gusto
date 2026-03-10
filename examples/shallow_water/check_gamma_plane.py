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

folder_name_suffix = 'step_trap_dginternal'

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

dt = 250
tmax = dt

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

eqns = ShallowWaterEquations(domain, parameters, coriolis_trap=(rstar-smooth_delta*Lx/nx, 2*Omega))

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