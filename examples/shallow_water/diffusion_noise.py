from gusto import (
    DiffusionEquation, OutputParameters, Domain, DiffusionParameters, BackwardEuler,
    CGDiffusion, Timestepper, IO, SpatialCoordinate
)

from firedrake import (
    PeriodicRectangleMesh, PCG64, RandomGenerator, exp, FunctionSpace, assemble,
    interpolate, Function
)

nx = 256
# nx=100
ny = nx
Lx = 7e7
# Lx=10
Ly = Lx
L = Lx

factor = Lx/10

dt = 0.01*factor
tmax = 0.2*factor

kappa = 1.*factor
# mu = 5.

mesh = PeriodicRectangleMesh(nx=nx, ny=ny, Lx=Lx, Ly=Ly, quadrilateral=True)
breakpoint()
output = OutputParameters(dirname=f'/data/home/sh1293/results/jupiter_sw/sample_noise', dumpfreq=1, dump_nc=True, checkpoint=True)
# output = OutputParameters(dump_vtus=False, dump_diagnostics=False)
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


dest_mesh = PeriodicRectangleMesh(nx=256, ny=256, Lx=7e7, Ly=7e7, quadrilateral=True)
V_dest = FunctionSpace(dest_mesh, "DG", 2) ## probably change this to just be whatever's in the script
f_dest = Function(V_dest).interpolate(timestepper.fields("f"), allow_missing_dofs=True)
# breakpoint()