from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate,
                       exp, errornorm, norm, tricontourf, File)
from gusto import *
import matplotlib.pyplot as plt

dt = 0.001
tmax = 0.1
L = 10.
m = PeriodicIntervalMesh(50, L)
mesh = ExtrudedMesh(m, layers=50, layer_height=L/50.)

x = SpatialCoordinate(mesh)
f_init = exp(-((x[0]-0.5*L)**2 + (x[1]-0.5*L)**2))

kappa = 1.
mu = 5.
dirname = "parareal_diffusion/"

# compute fine solution
domain = Domain(mesh, dt, family="CG", degree=1)
V = domain.spaces("DG")
diffusion_params = DiffusionParameters(kappa=kappa, mu=mu)
eqn = DiffusionEquation(domain, V, "f",
                        diffusion_parameters=diffusion_params)

output = OutputParameters(dirname=dirname+"diffusion")
io = IO(domain, output)
scheme = Heun(domain)
diffusion_methods = [InteriorPenaltyDiffusion(eqn, "f", diffusion_params)]
timestepper = Timestepper(eqn, scheme, io,
                          spatial_methods=diffusion_methods)
timestepper.fields("f").interpolate(f_init)
timestepper.run(0, tmax)

# store answer and write to file for comparison with parareal answer
ans = Function(V).assign(timestepper.fields("f"))
outfile = File('out.pvd')
outf = Function(V).assign(timestepper.fields("f"))
outfile.write(outf)

# compute parareal solution
n_intervals = 10
n_iterations = 10
domain = Domain(mesh, tmax, family="CG", degree=1)
V = domain.spaces("DG")
diffusion_params = DiffusionParameters(kappa=kappa, mu=mu)
eqn = DiffusionEquation(domain, V, "f",
                        diffusion_parameters=diffusion_params)
output = OutputParameters(dirname=dirname+"parareal_diffusion")
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

p_ans = Function(V).assign(timestepper.fields("f"))
outf.assign(timestepper.fields("f"))
outfile.write(outf)
outf.assign(p_ans-ans)
outfile.write(outf)
print("JEMMA: in test file")
print(ans.dat.data.min(), ans.dat.data.max())
print(p_ans.dat.data.min(), p_ans.dat.data.max())
print("error: ", errornorm(p_ans, ans))
