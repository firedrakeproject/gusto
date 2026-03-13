"""
Tests transport with the SUPG options. This checks that the field is transported
to the correct position.
"""

from firedrake import norm, FunctionSpace, VectorFunctionSpace, as_vector
from gusto import *
import pytest


def run(timestepper, tmax, f_end):
    timestepper.run(0, tmax)
    return norm(timestepper.fields("f") - f_end) / norm(f_end)


def run_coupled(timestepper, tmax, f_end):
    timestepper.run(0, tmax)
    norm1 = norm(timestepper.fields("f1") - f_end) / norm(f_end)
    norm2 = norm(timestepper.fields("f2") - f_end) / norm(f_end)
    return norm1, norm2


@pytest.mark.parametrize("scheme", ["ssprk", "implicit_midpoint"])
def test_supg_transport_mixed_scalar(tmpdir, scheme, tracer_setup):
    setup = tracer_setup(tmpdir, geometry="slice")
    domain = setup.domain

    ibp = IntegrateByParts.TWICE

    opts = SUPGOptions(ibp=ibp)

    tracer1 = ActiveTracer(name='f1', space="theta",
                           variable_type=TracerVariableType.mixing_ratio,
                           transport_eqn=TransportEquationType.advective)
    tracer2 = ActiveTracer(name='f2', space="theta",
                           variable_type=TracerVariableType.mixing_ratio,
                           transport_eqn=TransportEquationType.conservative)
    tracers = [tracer1, tracer2]
    Vu = domain.spaces("HDiv")
    eqn = CoupledTransportEquation(domain, active_tracers=tracers, Vu=Vu)
    suboptions = {}
    suboptions.update({'f1': [time_derivative, transport]})
    suboptions.update({'f2': None})
    opts = SUPGOptions(suboptions=suboptions)
    transport_method = [DGUpwind(eqn, "f1", ibp=ibp), DGUpwind(eqn, "f2", ibp=ibp)]

    if scheme == "ssprk":
        transport_scheme = SSPRK3(domain, options=opts)
    elif scheme == "implicit_midpoint":
        transport_scheme = TrapeziumRule(domain, options=opts)

    time_varying_velocity = False
    timestepper = PrescribedTransport(
        eqn, transport_scheme, setup.io, time_varying_velocity, transport_method
    )

    # Initial conditions
    timestepper.fields("f1").interpolate(setup.f_init)
    timestepper.fields("f2").interpolate(setup.f_init)
    timestepper.fields("u").project(setup.uexpr)

    error1, error2 = run_coupled(timestepper, setup.tmax, setup.f_end)
    assert error1 < setup.tol, \
        'The transport error for f1 is greater than the permitted tolerance'
    assert error2 < setup.tol, \
        'The transport error for f2 is greater than the permitted tolerance'


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

    if equation_form == "advective":
        eqn = AdvectionEquation(domain, V, "f")
    else:
        eqn = ContinuityEquation(domain, V, "f")

    opts = SUPGOptions(ibp=ibp)
    transport_method = DGUpwind(eqn, "f", ibp=ibp)

    if scheme == "ssprk":
        transport_scheme = SSPRK3(domain, options=opts)
    elif scheme == "implicit_midpoint":
        transport_scheme = TrapeziumRule(domain, options=opts)

    time_varying_velocity = False
    timestepper = PrescribedTransport(
        eqn, transport_scheme, setup.io, time_varying_velocity, transport_method
    )

    timestepper.fields("f").interpolate(setup.f_init)
    timestepper.fields("u").project(setup.uexpr)
    error = run(timestepper, setup.tmax, setup.f_end)
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

    opts = SUPGOptions(ibp=ibp)
    transport_method = DGUpwind(eqn, "f", ibp=ibp)

    if scheme == "ssprk":
        transport_scheme = SSPRK3(domain, options=opts)
    elif scheme == "implicit_midpoint":
        transport_scheme = TrapeziumRule(domain, options=opts)

    time_varying_velocity = False
    timestepper = PrescribedTransport(
        eqn, transport_scheme, setup.io, time_varying_velocity, transport_method
    )

    # Initial conditions
    f = timestepper.fields("f")
    if space == "CG":
        f.interpolate(f_init)
    else:
        f.project(f_init)
    timestepper.fields("u").project(setup.uexpr)

    f_end = as_vector([setup.f_end]*gdim)
    error = run(timestepper, setup.tmax, f_end)
    assert error < setup.tol, \
        'The transport error is greater than the permitted tolerance'
