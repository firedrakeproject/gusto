"""
Tests the DG upwind transport scheme, with various options. This tests that the
field is transported to the correct place.
"""

from firedrake import norm, VectorFunctionSpace, as_vector
from gusto import *
import pytest


def run(eqn, transport_scheme, io, tmax, f_end):
    timestepper = PrescribedTransport(eqn, transport_scheme, io)
    timestepper.run(0, tmax)
    return norm(eqn.fields("f") - f_end) / norm(f_end)


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

    io = IO(domain, eqn, dt=setup.dt, output=setup.output)
    eqn.fields("f").interpolate(setup.f_init)
    eqn.fields("u").project(setup.uexpr)

    transport_scheme = SSPRK3(domain, io)
    error = run(eqn, transport_scheme, io, setup.tmax, setup.f_end)
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

    io = IO(domain, eqn, dt=setup.dt, output=setup.output)
    eqn.fields("f").interpolate(f_init)
    eqn.fields("u").project(setup.uexpr)
    transport_schemes = SSPRK3(domain, io)
    f_end = as_vector([setup.f_end]*gdim)
    error = run(eqn, transport_schemes, io, setup.tmax, f_end)
    assert error < setup.tol, \
        'The transport error is greater than the permitted tolerance'
