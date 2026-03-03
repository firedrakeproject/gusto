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

folder_name_suffix = 'no_trap'

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
tmax = 5*dt

dirname=f'/data/home/sh1293/results/jupiter_sw/check_gamma_plane_{folder_name_suffix}'

mesh = PeriodicRectangleMesh(nx=nx, ny=ny, Lx=Lx, Ly=Ly, quadrilateral=True)
output = OutputParameters(dirname=f'{dirname}', dumpfreq=1, dump_nc=True)

parameters = ShallowWaterParameters(mesh, H=H, Omega=Omega, R=R,
                                    rotation=CoriolisOptions.gammaplane)

domain = Domain(mesh, dt, "RTCF", 1)


### rstar and smooth_delta are so the trap can be in the same place as my big script

eqns = ShallowWaterEquations(domain, parameters)#, coriolis_trap=(0.5*rstar-smooth_delta*Lx/nx, 2*Omega))

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
D0 = stepper.fields("D")

u0.assign(0.)
D0.assign(H)
Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

stepper.run(t=0, tmax=tmax)

logger.info(f'File produced:\ncheck_gamma_plane_{folder_name_suffix}')