from gusto import (
    DiffusionEquation, OutputParameters, Domain, DiffusionParameters, BackwardEuler,
    CGDiffusion, Timestepper, IO, SpatialCoordinate
)

from firedrake import (
    PCG64, RandomGenerator, sqrt
)

def diffusion_noise_generation(mesh, Lx):

    mesh = mesh
    Lx = Lx
    factor = Lx/10

    dt = 0.01*factor
    tmax = 0.2*factor

    kappa = 1.*factor
    # mu = 5.
    
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

    return timestepper.fields("f")