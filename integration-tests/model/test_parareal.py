from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate,
                       exp, errornorm, norm, tricontourf, File)
from gusto import *
import pytest

def test_parareal(tmpdir):
    dt = 0.001
    L = 10.
    m = PeriodicIntervalMesh(50, L)
    mesh = ExtrudedMesh(m, layers=50, layer_height=L/50.)

    x = SpatialCoordinate(mesh)
    f_init = exp(-((x[0]-0.5*L)**2 + (x[1]-0.5*L)**2))

    kappa = 1.
    mu = 5.
    diffusion_params = DiffusionParameters(kappa=kappa, mu=mu)
    dt_ratio = 10
    for nints in range(1,3):
        output = OutputParameters(dirname=str(tmpdir)+"_%s_fine" % str(nints), dumpfreq=1)
        state = State(mesh, dt=dt, output=output)
        V = state.spaces("DG", "DG", 1)
        eqn = DiffusionEquation(state, V, "f",
                                diffusion_parameters=diffusion_params)
        state.fields("f").interpolate(f_init)
        state.fields("u", V)

        tmax = nints * dt
        scheme = Heun(state, subcycles=dt_ratio)
        timestepper = Timestepper(eqn, scheme, state)
        timestepper.run(0, tmax)
        ans = Function(V).assign(state.fields("f"))

        output = OutputParameters(dirname=str(tmpdir) + str(nints), dumpfreq=1)
        state = State(mesh, dt=dt, output=output)
        V = state.spaces("DG", "DG", 1)
        eqn = DiffusionEquation(state, V, "f",
                                diffusion_parameters=diffusion_params)
        state.fields("f").interpolate(f_init)
        state.fields("u", V)
        coarse_scheme = Heun(state)
        fine_scheme = Heun(state, subcycles=dt_ratio)

        timestepper = Parareal(eqn, coarse_scheme, fine_scheme, state, nints, nints)
        timestepper.run(0, tmax)

        parareal_ans = Function(V).assign(state.fields("f"))

        assert abs(ans.dat.data.min() - parareal_ans.dat.data.min()) < 1e-22
        assert abs(ans.dat.data.max() - parareal_ans.dat.data.max()) < 1e-22
        assert errornorm(parareal_ans, ans) < 1e-15
