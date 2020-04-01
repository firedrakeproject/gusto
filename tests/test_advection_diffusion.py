from firedrake import as_vector
from gusto import *


def run(setup):

    state = setup.state
    tmax = setup.tmax
    f_init = setup.f_init
    fspace = state.spaces("DG")

    u = state.fields("u", space=state.spaces("HDiv"))
    u0 = 10.
    u.project(as_vector([u0, 0.]))

    equation = AdvectionDiffusionEquation(state, fspace, "f", kappa=1., mu=5)
    problem = [(equation, ((SSPRK3(state), advection),
                           (BackwardEuler(state), diffusion)))]
    f = state.fields("f")
    f.interpolate(f_init)

    timestepper = PrescribedAdvection(state, problem)

    timestepper.run(0, tmax=tmax)


def test_advection_diffusion(tmpdir, tracer_setup):

    setup = tracer_setup(tmpdir, geometry="slice", blob=True)
    run(setup)
