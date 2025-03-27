"""
Tests the Split horizontal and vertical DG upwind transport scheme for
advective form transport equation. This tests that the
field is transported to the correct place.
"""

from firedrake import norm, VectorFunctionSpace, as_vector
from gusto import *


def run(timestepper, tmax, f_end):
    timestepper.run(0, tmax)
    return norm(timestepper.fields("f") - f_end) / norm(f_end)


def test_split_dg_transport_scalar(tmpdir, tracer_setup):
    setup = tracer_setup(tmpdir, "slice")
    domain = setup.domain
    V = domain.spaces("DG")

    eqn = AdvectionEquation(domain, V, "f")
    eqn = split_hv_advective_form(eqn, "f")

    transport_method = SplitDGUpwind(eqn, "f")
    transport_scheme = SSPRK3(domain)

    time_varying_velocity = False
    timestepper = PrescribedTransport(
        eqn, transport_scheme, setup.io, time_varying_velocity, transport_method
    )

    # Initial conditions
    timestepper.fields("f").interpolate(setup.f_init)
    timestepper.fields("u").project(setup.uexpr)

    error = run(timestepper, setup.tmax, setup.f_end)
    assert error < setup.tol, \
        'The transport error is greater than the permitted tolerance'


def test_split_dg_transport_vector(tmpdir, tracer_setup):
    setup = tracer_setup(tmpdir, "slice")
    domain = setup.domain
    gdim = domain.mesh.geometric_dimension()
    f_init = as_vector([setup.f_init]*gdim)
    V = VectorFunctionSpace(domain.mesh, "DG", 1)
    eqn = AdvectionEquation(domain, V, "f")
    eqn = split_hv_advective_form(eqn, "f")

    transport_scheme = SSPRK3(domain)
    transport_method = SplitDGUpwind(eqn, "f")

    time_varying_velocity = False
    timestepper = PrescribedTransport(
        eqn, transport_scheme, setup.io, time_varying_velocity, transport_method
    )

    # Initial conditions
    timestepper.fields("f").interpolate(f_init)
    timestepper.fields("u").project(setup.uexpr)
    f_end = as_vector([setup.f_end]*gdim)
    error = run(timestepper, setup.tmax, f_end)
    assert error < setup.tol, \
        'The transport error is greater than the permitted tolerance'
