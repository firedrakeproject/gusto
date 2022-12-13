"""
Tests the embedded DG transport scheme, checking that the field is transported
to the right place.
"""

from firedrake import norm
from gusto import *
import pytest


def run(eqn, transport_schemes, io, tmax, f_end):
    timestepper = PrescribedTransport(eqn, transport_schemes, io)
    timestepper.run(0, tmax)
    return norm(eqn.fields("f") - f_end) / norm(f_end)


@pytest.mark.parametrize("ibp", [IntegrateByParts.ONCE, IntegrateByParts.TWICE])
@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
@pytest.mark.parametrize("space", ["broken", "dg"])
def test_embedded_dg_advection_scalar(tmpdir, ibp, equation_form, space,
                                      tracer_setup):
    setup = tracer_setup(tmpdir, geometry="slice")
    domain = setup.domain
    V = domain.spaces("theta")

    if space == "broken":
        opts = EmbeddedDGOptions()
    elif space == "dg":
        opts = EmbeddedDGOptions(embedding_space=domain.spaces("DG"))

    if equation_form == "advective":
        eqn = AdvectionEquation(domain, V, "f", ibp=ibp)
    else:
        eqn = ContinuityEquation(domain, V, "f", ibp=ibp)

    io = IO(domain, eqn, output=setup.output)
    eqn.fields("f").interpolate(setup.f_init)
    eqn.fields("u").project(setup.uexpr)

    transport_schemes = SSPRK3(domain, options=opts)

    error = run(eqn, transport_schemes, io, setup.tmax, setup.f_end)
    assert error < setup.tol, \
        'The transport error is greater than the permitted tolerance'
