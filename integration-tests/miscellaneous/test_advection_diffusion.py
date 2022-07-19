"""
Tests discretisations of the advection-diffusion equation. This checks the
errornorm for the resulting field to ensure that the result is reasonable.
"""

from firedrake import as_vector
from gusto import *


def run(setup):

    state = setup.state
    tmax = setup.tmax
    f_init = setup.f_init
    V = state.spaces("DG", "DG", 1)

    diffusion_params = DiffusionParameters(kappa=1., mu=5)
    equation = AdvectionDiffusionEquation(state, V, "f", ufamily=setup.family,
                                          udegree=setup.degree,
                                          diffusion_parameters=diffusion_params)
    problem = [(equation, ((SSPRK3(state), transport),
                           (BackwardEuler(state), diffusion)))]
    state.fields("f").interpolate(f_init)
    state.fields("u").project(as_vector([10, 0.]))

    timestepper = PrescribedTransport(state, problem)

    timestepper.run(0, tmax=tmax)


def test_advection_diffusion(tmpdir, tracer_setup):

    setup = tracer_setup(tmpdir, geometry="slice", blob=True)
    run(setup)
