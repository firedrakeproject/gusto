from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate,
                       exp, errornorm, norm, tricontourf, File)
from gusto import *
import matplotlib.pyplot as plt

dt = 0.001
L = 10.
m = PeriodicIntervalMesh(50, L)
mesh = ExtrudedMesh(m, layers=50, layer_height=L/50.)

output = OutputParameters(dirname="parareal_diffusion", dumpfreq=1)

x = SpatialCoordinate(mesh)
f_init = exp(-((x[0]-0.5*L)**2 + (x[1]-0.5*L)**2))

kappa = 1.
mu = 5.
diffusion_params = DiffusionParameters(kappa=kappa, mu=mu)
dt_ratio = 10
for nints in range(1,3):
    state = State(mesh, dt=dt, output=output)
    V = state.spaces("DG", "DG", 1)
    eqn = DiffusionEquation(state, V, "f",
                            diffusion_parameters=diffusion_params)
    state.fields("f").interpolate(f_init)
    state.fields("u", V)

    outfile = File("out%i.pvd" % nints)
    tmax = nints * dt
    scheme = Heun(state, subcycles=dt_ratio)
    timestepper = Timestepper(eqn, scheme, state)
    timestepper.run(0, tmax)
    ans = Function(V).assign(state.fields("f"))
    outf = Function(V)
    outf.assign(state.fields("f"))
    outfile.write(outf)

    state.fields("f").interpolate(f_init)
    coarse_scheme = Heun(state)
    fine_scheme = Heun(state, subcycles=dt_ratio)

    timestepper = Parareal(eqn, coarse_scheme, fine_scheme, state, nints, nints)
    timestepper.run(0, tmax)

    p_ans = Function(V).assign(state.fields("f"))
    outf.assign(state.fields("f"))
    outfile.write(outf)
    outf.assign(p_ans-ans)
    outfile.write(outf)
    print("JEMMA: in test file")
    print(ans.dat.data.min(), ans.dat.data.max())
    print(p_ans.dat.data.min(), p_ans.dat.data.max())
    print(nints, errornorm(p_ans, ans))
