"""
Tests transport with the SUPG options. This checks that the field is transported
to the correct position.
"""

from firedrake import norm, FunctionSpace, VectorFunctionSpace, as_vector
from gusto import *
import pytest


def run(eqn, transport_scheme, io, tmax, f_end):
    timestepper = PrescribedTransport(eqn, transport_scheme, io)
    timestepper.run(0, tmax)
    return norm(eqn.fields("f") - f_end) / norm(f_end)


@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
@pytest.mark.parametrize("scheme", ["ssprk", "implicit_midpoint"])
@pytest.mark.parametrize("space", ["CG", "theta"])
def test_supg_transport_scalar(tmpdir, equation_form, scheme, space,
                               tracer_setup):

    setup = tracer_setup(tmpdir, geometry="slice")
    domain = setup.domain

    if space == "CG":
        V = FunctionSpace(domain.mesh, "CG", 1)
        ibp = IntegrateByParts.NEVER
    else:
        V = domain.spaces("theta")
        ibp = IntegrateByParts.TWICE

    opts = SUPGOptions(ibp=ibp)

    if equation_form == "advective":
        eqn = AdvectionEquation(domain, V, "f")
    else:
        eqn = ContinuityEquation(domain, V, "f")

    io = IO(domain, eqn, dt=setup.dt, output=setup.output)

    eqn.fields("f").interpolate(setup.f_init)
    eqn.fields("u").project(setup.uexpr)

    if scheme == "ssprk":
        transport_scheme = SSPRK3(domain, io, options=opts)
    elif scheme == "implicit_midpoint":
        transport_scheme = ImplicitMidpoint(domain, io, options=opts)

    error = run(eqn, transport_scheme, io, setup.tmax, setup.f_end)
    assert error < setup.tol, \
        'The transport error is greater than the permitted tolerance'


@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
@pytest.mark.parametrize("scheme", ["ssprk", "implicit_midpoint"])
@pytest.mark.parametrize("space", ["CG", "HDiv"])
def test_supg_transport_vector(tmpdir, equation_form, scheme, space,
                               tracer_setup):

    setup = tracer_setup(tmpdir, geometry="slice")
    domain = setup.domain

    gdim = domain.mesh.geometric_dimension()
    f_init = as_vector([setup.f_init]*gdim)
    if space == "CG":
        V = VectorFunctionSpace(domain.mesh, "CG", 1)
        ibp = IntegrateByParts.NEVER
    else:
        V = domain.spaces("HDiv")
        ibp = IntegrateByParts.TWICE

    opts = SUPGOptions(ibp=ibp)

    if equation_form == "advective":
        eqn = AdvectionEquation(domain, V, "f")
    else:
        eqn = ContinuityEquation(domain, V, "f")

    io = IO(domain, eqn, dt=setup.dt, output=setup.output)

    f = eqn.fields("f")
    if space == "CG":
        f.interpolate(f_init)
    else:
        f.project(f_init)
    eqn.fields("u").project(setup.uexpr)
    if scheme == "ssprk":
        transport_scheme = SSPRK3(domain, io, options=opts)
    elif scheme == "implicit_midpoint":
        transport_scheme = ImplicitMidpoint(domain, io, options=opts)

    f_end = as_vector([setup.f_end]*gdim)
    error = run(eqn, transport_scheme, io, setup.tmax, f_end)
    assert error < setup.tol, \
        'The transport error is greater than the permitted tolerance'
