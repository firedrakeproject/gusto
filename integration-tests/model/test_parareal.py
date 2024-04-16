from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate,
                       exp, errornorm, Function)
from gusto import (Domain, IO, OutputParameters,
                   DiffusionParameters, DiffusionEquation,
                   InteriorPenaltyDiffusion,
                   Heun, Parareal, Timestepper)


def compute_reference(tmpdir, domain, diffusion_params, f_init, tmax):

    # ------------------------------------------------------------------------ #
    # Compute the reference solution using the fine timestep
    # ------------------------------------------------------------------------ #

    V = domain.spaces("DG")
    eqn = DiffusionEquation(domain, V, "f",
                            diffusion_parameters=diffusion_params)

    output = OutputParameters(dirname=tmpdir+"diffusion")
    io = IO(domain, output)
    scheme = Heun(domain)
    diffusion_methods = [InteriorPenaltyDiffusion(eqn, "f", diffusion_params)]
    timestepper = Timestepper(eqn, scheme, io,
                              spatial_methods=diffusion_methods)
    timestepper.fields("f").interpolate(f_init)
    timestepper.run(0, tmax)

    # return answer
    ans = Function(V).assign(timestepper.fields("f"))
    return ans


def compute_parareal(tmpdir, domain, diffusion_params, f_init, tmax,
                     n_intervals, n_iterations):

    # ------------------------------------------------------------------------ #
    # Compute parareal solution
    # ------------------------------------------------------------------------ #

    V = domain.spaces("DG")
    eqn = DiffusionEquation(domain, V, "f",
                            diffusion_parameters=diffusion_params)
    output = OutputParameters(dirname=tmpdir+"parareal_diffusion")
    io = IO(domain, output)
    coarse_scheme = Heun(domain)
    fine_scheme = Heun(domain)
    diffusion_methods = [InteriorPenaltyDiffusion(eqn, "f", diffusion_params)]
    timestepper = Timestepper(eqn,
                              Parareal(domain, coarse_scheme, fine_scheme,
                                       1, 10, n_intervals, n_iterations),
                              io, spatial_methods=diffusion_methods)
    timestepper.fields("f").interpolate(f_init)
    timestepper.run(0, tmax)

    # return answer
    ans = Function(V).assign(timestepper.fields("f"))
    return ans


def test_parareal(tmpdir):

    dirname = str(tmpdir)
    L = 10.
    m = PeriodicIntervalMesh(50, L)
    mesh = ExtrudedMesh(m, layers=50, layer_height=L/50.)

    x = SpatialCoordinate(mesh)
    f_init = exp(-((x[0]-0.5*L)**2 + (x[1]-0.5*L)**2))

    kappa = 1.
    mu = 5.
    diffusion_params = DiffusionParameters(kappa=kappa, mu=mu)

    # compute reference solution with fine timestep
    dt = 0.001
    domain = Domain(mesh, dt, family="CG", degree=1)
    tmax = 0.1
    ref_sol = compute_reference(dirname, domain, diffusion_params, f_init,
                                tmax)

    # compute parareal solution
    domain = Domain(mesh, tmax, family="CG", degree=1)
    n_intervals = 10
    n_iterations = 10
    parareal_sol = compute_parareal(dirname, domain, diffusion_params, f_init,
                                    tmax, n_intervals, n_iterations)

    assert errornorm(parareal_sol, ref_sol) < 1e-15
