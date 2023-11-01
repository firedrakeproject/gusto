from firedrake import norm
from gusto import *
import pytest


def run(timestepper, tmax, f_end):
    timestepper.run(0, tmax)
    return norm(timestepper.fields("f") - f_end) / norm(f_end)

@pytest.mark.parametrize("scheme", ["ssp3","ark2","ars3", "trap2", "euler"])
def test_time_discretisation(tmpdir, scheme, tracer_setup):

    setup = tracer_setup(tmpdir, "sphere")
    domain = setup.domain
    V = domain.spaces("DG")
    eqn = ContinuityEquation(domain, V, "f")
    # Split continuity term
    eqn = split_continuity_form(eqn)
    #Label terms are implicit and explicit
    eqn.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
    eqn.label_terms(lambda t: t.has_label(transport), explicit)

    if scheme == "ssp3":
        transport_scheme = SSP3(domain)
    elif scheme == "ark2":
        transport_scheme = ARK2(domain)
    elif scheme == "ars3":
        transport_scheme = ARS3(domain)
    elif scheme == "trap2":
        transport_scheme = Trap2(domain)
    elif scheme == "euler":
        transport_scheme = IMEX_Euler(domain)

    transport_method = DGUpwind(eqn, "f")

    timestepper = PrescribedTransport(eqn, transport_scheme, setup.io, transport_method)

    # Initial conditions
    timestepper.fields("f").interpolate(setup.f_init)
    timestepper.fields("u").project(setup.uexpr)
    assert run(timestepper, setup.tmax, setup.f_end) < setup.tol
