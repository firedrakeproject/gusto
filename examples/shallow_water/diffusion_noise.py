from gusto import (
    DiffusionEquation, OutputParameters, Domain, DiffusionParameters, BackwardEuler,
    InteriorPenaltyDiffusion, Timestepper, IO, SpatialCoordinate
)

from firedrake import (
    PeriodicRectangleMesh, PCG64, RandomGenerator, exp
)

nx = 256
# nx=10
ny = nx
Lx = 7e7
# Lx=10
Ly = Lx
L = Lx

dt = 0.01
tmax = 100.

kappa = 1.
mu = 5.

mesh = PeriodicRectangleMesh(nx=nx, ny=ny, Lx=Lx, Ly=Ly, quadrilateral=True)
output = OutputParameters(dirname=f'/data/home/sh1293/results/jupiter_sw/sample_noise', dumpfreq=1, dump_nc=True, checkpoint=True)
domain = Domain(mesh, dt, "RTCF", 1)

V = domain.spaces("DG")

diffusion_params = DiffusionParameters(domain.mesh, kappa=kappa, mu=mu)
eqn = DiffusionEquation(domain, V, "f", diffusion_parameters=diffusion_params)
diffusion_scheme = BackwardEuler(domain)
diffusion_methods = [InteriorPenaltyDiffusion(eqn, "f", diffusion_params)]
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
