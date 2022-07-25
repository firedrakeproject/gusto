"""
Tests transport with the SUPG options. This checks that the field is transported
to the correct position.
"""

from firedrake import norm, VectorFunctionSpace, as_vector
from gusto import *
import pytest


def run(state, transport_scheme, tmax, f_end):
    timestepper = PrescribedTransport(state, transport_scheme)
    timestepper.run(0, tmax)
    return norm(state.fields("f") - f_end)


@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
@pytest.mark.parametrize("scheme", ["ssprk", "implicit_midpoint"])
@pytest.mark.parametrize("space", ["CG", "theta"])
def test_supg_transport_scalar(tmpdir, equation_form, scheme, space,
                               tracer_setup):

    setup = tracer_setup(tmpdir, geometry="slice")
    state = setup.state

    if space == "CG":
        V = state.spaces("CG1", "CG", 1)
        ibp = IntegrateByParts.NEVER
    else:
        V = state.spaces("theta", degree=1)
        ibp = IntegrateByParts.TWICE

    opts = SUPGOptions(ibp=ibp)

    if equation_form == "advective":
        eqn = AdvectionEquation(state, V, "f", ufamily=setup.family,
                                udegree=setup.degree)
    else:
        eqn = ContinuityEquation(state, V, "f", ufamily=setup.family,
                                 udegree=setup.degree)
    state.fields("f").interpolate(setup.f_init)
    state.fields("u").project(setup.uexpr)
    if scheme == "ssprk":
        transport_scheme = [(eqn, SSPRK3(state, options=opts))]
    elif scheme == "implicit_midpoint":
        transport_scheme = [(eqn, ImplicitMidpoint(state, options=opts))]

    assert run(state, transport_scheme, setup.tmax, setup.f_end) < setup.tol


@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
@pytest.mark.parametrize("scheme", ["ssprk", "implicit_midpoint"])
@pytest.mark.parametrize("space", ["CG", "HDiv"])
def test_supg_transport_vector(tmpdir, equation_form, scheme, space,
                               tracer_setup):

    setup = tracer_setup(tmpdir, geometry="slice")
    state = setup.state

    gdim = state.mesh.geometric_dimension()
    f_init = as_vector((setup.f_init, *[0.]*(gdim-1)))
    if space == "CG":
        V = VectorFunctionSpace(state.mesh, "CG", 1)
        ibp = IntegrateByParts.NEVER
    else:
        V = state.spaces("HDiv", setup.family, setup.degree)
        ibp = IntegrateByParts.TWICE

    opts = SUPGOptions(ibp=ibp)

    if equation_form == "advective":
        eqn = AdvectionEquation(state, V, "f", ufamily=setup.family,
                                udegree=setup.degree)
    else:
        eqn = ContinuityEquation(state, V, "f", ufamily=setup.family,
                                 udegree=setup.degree)
    f = state.fields("f")
    if space == "CG":
        f.interpolate(f_init)
    else:
        f.project(f_init)
    state.fields("u").project(setup.uexpr)
    if scheme == "ssprk":
        transport_scheme = [(eqn, SSPRK3(state, options=opts))]
    elif scheme == "implicit_midpoint":
        transport_scheme = [(eqn, ImplicitMidpoint(state, options=opts))]

    f_end = as_vector((setup.f_end, *[0.]*(gdim-1)))
    assert run(state, transport_scheme, setup.tmax, f_end) < setup.tol
