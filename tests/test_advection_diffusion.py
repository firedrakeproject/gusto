from firedrake import as_vector, exp, norm
from gusto import *


def run(setup):

    state = setup.state
    tmax = setup.tmax
    f_init = setup.f_init

    fspace = state.spaces("DG")
    state.fields("f_exact", space=fspace)

    x = SpatialCoordinate(state.mesh)
    L = 10.

    u = state.fields("u", space=state.spaces("HDiv"))
    u.project(as_vector([10., 0.]))

    equation = AdvectionDiffusionEquation(state, fspace, "f", kappa=1., mu=5)
    schemes = [SSPRK3(state, equation, advection),
               BackwardEuler(state, equation, diffusion)]
    f = state.fields("f")
    f.interpolate(f_init)

    timestepper = PrescribedAdvectionTimestepper(state, schemes)

    timestepper.run(0, tmax=tmax)

    d = 5.
    xs = x[0] - 0.5*L
    f_exact_expr = (1/(1+4*tmax))*((exp(-d*(2*xs+d)) + exp(d*(2*xs-d)))*f_init)**(1/(1+4*tmax))
    f_exact = Function(f.function_space()).interpolate(f_exact_expr)
    ferr = Function(f.function_space()).assign(f-f_exact)

    return ferr


def test_advection_diffusion(tmpdir, tracer_setup):

    setup = tracer_setup(tmpdir, geometry="slice", blob=True)
    err = run(setup)
    assert norm(err) < 5e-2
