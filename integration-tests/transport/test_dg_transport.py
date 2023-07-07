"""
Tests the DG upwind transport scheme, with various options. This tests that the
field is transported to the correct place.
"""

from firedrake import norm, VectorFunctionSpace, as_vector
from gusto import *
import pytest


def run(timestepper, tmax, f_end):
    timestepper.run(0, tmax)
    return norm(timestepper.fields("f") - f_end) / norm(f_end)


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
def test_dg_transport_scalar(tmpdir, geometry, equation_form, tracer_setup):
    setup = tracer_setup(tmpdir, geometry)
    domain = setup.domain
    V = domain.spaces("DG")

    if equation_form == "advective":
        eqn = AdvectionEquation(domain, V, "f")
    else:
        eqn = ContinuityEquation(domain, V, "f")

    transport_scheme = SSPRK3(domain)
    transport_method = DGUpwind(eqn, "f")

    timestepper = PrescribedTransport(eqn, transport_scheme, transport_method, setup.io)

    # Initial conditions
    timestepper.fields("f").interpolate(setup.f_init)
    timestepper.fields("u").project(setup.uexpr)

    error = run(timestepper, setup.tmax, setup.f_end)
    assert error < setup.tol, \
        'The transport error is greater than the permitted tolerance'


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
def test_dg_transport_vector(tmpdir, geometry, equation_form, tracer_setup):
    setup = tracer_setup(tmpdir, geometry)
    domain = setup.domain
    gdim = domain.mesh.geometric_dimension()
    f_init = as_vector([setup.f_init]*gdim)
    V = VectorFunctionSpace(domain.mesh, "DG", 1)
    if equation_form == "advective":
        eqn = AdvectionEquation(domain, V, "f")
    else:
        eqn = ContinuityEquation(domain, V, "f")

    transport_scheme = SSPRK3(domain)
    transport_method = DGUpwind(eqn, "f")

    timestepper = PrescribedTransport(eqn, transport_scheme, transport_method, setup.io)

    # Initial conditions
    timestepper.fields("f").interpolate(f_init)
    timestepper.fields("u").project(setup.uexpr)
    f_end = as_vector([setup.f_end]*gdim)
    error = run(timestepper, setup.tmax, f_end)
    assert error < setup.tol, \
        'The transport error is greater than the permitted tolerance'
