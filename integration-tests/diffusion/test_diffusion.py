"""
Tests discretisations of the diffusion equation. This checks the errornorm for
the resulting field to ensure that the result is reasonable.
"""

from gusto import *
from firedrake import (VectorFunctionSpace, Constant, as_vector, errornorm)
import pytest


def run(timestepper, tmax):
    timestepper.run(0., tmax)
    return timestepper.fields("f")


@pytest.mark.parametrize("DG", [True, False])
def test_scalar_diffusion(tmpdir, DG, tracer_setup):

    setup = tracer_setup(tmpdir, geometry="slice", blob=True)
    domain = setup.domain
    f_init = setup.f_init
    tmax = setup.tmax
    tol = 5.e-2
    kappa = 1.

    f_end_expr = (1/(1+4*tmax))*f_init**(1/(1+4*tmax))

    if DG:
        V = domain.spaces("DG", "DG", degree=1)
    else:
        V = domain.spaces("theta", degree=1)

    mu = 5.

    diffusion_params = DiffusionParameters(kappa=kappa, mu=mu)
    eqn = DiffusionEquation(domain, V, "f", diffusion_parameters=diffusion_params)
    diffusion_scheme = BackwardEuler(domain)
    diffusion_methods = [InteriorPenaltyDiffusion(eqn, "f", diffusion_params)]
    timestepper = Timestepper(eqn, diffusion_scheme, setup.io, spatial_methods=diffusion_methods)

    # Initial conditions
    timestepper.fields("f").interpolate(f_init)
    f_end = run(timestepper, tmax)
    assert errornorm(f_end_expr, f_end) < tol


@pytest.mark.parametrize("DG", [True, False])
def test_vector_diffusion(tmpdir, DG, tracer_setup):

    setup = tracer_setup(tmpdir, geometry="slice", blob=True)
    domain = setup.domain
    f_init = setup.f_init
    tmax = setup.tmax
    tol = 3.e-2
    kappa = 1.

    f_end_expr = (1/(1+4*tmax))*f_init**(1/(1+4*tmax))

    kappa = Constant([[kappa, 0.], [0., kappa]])
    if DG:
        V = VectorFunctionSpace(domain.mesh, "DG", 1)
    else:
        V = domain.spaces("HDiv", "CG", 1)
    f_init = as_vector([f_init, 0.])
    f_end_expr = as_vector([f_end_expr, 0.])

    mu = 5.

    diffusion_params = DiffusionParameters(kappa=kappa, mu=mu)
    eqn = DiffusionEquation(domain, V, "f", diffusion_parameters=diffusion_params)
    diffusion_scheme = BackwardEuler(domain)
    diffusion_methods = [InteriorPenaltyDiffusion(eqn, "f", diffusion_params)]
    timestepper = Timestepper(eqn, diffusion_scheme, setup.io, spatial_methods=diffusion_methods)

    # Initial conditions
    if DG:
        timestepper.fields("f").interpolate(f_init)
    else:
        timestepper.fields("f").project(f_init)

    f_end = run(timestepper, tmax)
    assert errornorm(f_end_expr, f_end) < tol
