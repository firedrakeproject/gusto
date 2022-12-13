from firedrake import norm
from gusto import *
import pytest


def run(eqn, transport_scheme, io, tmax, f_end):
    timestepper = PrescribedTransport(eqn, transport_scheme, io)
    timestepper.run(0, tmax)
    return norm(eqn.fields("f") - f_end) / norm(f_end)


@pytest.mark.parametrize("scheme", ["ssprk", "implicit_midpoint",
                                    "RK4", "Heun", "BDF2"])
def test_time_discretisation(tmpdir, scheme, tracer_setup):
    geometry = "sphere"
    setup = tracer_setup(tmpdir, geometry)
    domain = setup.domain
    V = domain.spaces("DG")

    eqn = AdvectionEquation(domain, V, "f")
    io = IO(domain, eqn, output=setup.output)

    eqn.fields("f").interpolate(setup.f_init)
    eqn.fields("u").project(setup.uexpr)

    if scheme == "ssprk":
        transport_scheme = SSPRK3(domain)
    elif scheme == "implicit_midpoint":
        transport_scheme = ImplicitMidpoint(domain)
    elif scheme == "RK4":
        transport_scheme = RK4(domain)
    elif scheme == "Heun":
        transport_scheme = Heun(domain)
    elif scheme == "BDF2":
        transport_scheme = BDF2(domain)
    assert run(eqn, transport_scheme, io, setup.tmax, setup.f_end) < setup.tol
