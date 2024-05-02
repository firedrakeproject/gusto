from firedrake import norm
from gusto import *
import pytest


def run(timestepper, tmax, f_end):
    timestepper.run(0, tmax)
    return norm(timestepper.fields("f") - f_end) / norm(f_end)


@pytest.mark.parametrize(
    "scheme", ["IMEX_SDC_Le(1,1)", "IMEX_SDC_R(2,2)", "BE_SDC_Lo(3,3)", "FE_SDC_Le(3,5)"])
def test_time_discretisationsdc(tmpdir, scheme, tracer_setup):
    geometry = "sphere"
    setup = tracer_setup(tmpdir, geometry)
    domain = setup.domain
    V = domain.spaces("DG")
    eqn = AdvectionEquation(domain, V, "f")
    node_dist = "LEGENDRE"
    qdelta_imp="BE"
    qdelta_exp="FE"

    if scheme == "IMEX_SDC_Le(1,1)":
        node_type="GAUSS"
        M = 1
        k = 1
        eqn = ContinuityEquation(domain, V, "f")
        # Split continuity term
        eqn = split_continuity_form(eqn)
        eqn.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
        eqn.label_terms(lambda t: t.has_label(transport), explicit)
        base_scheme = IMEX_Euler(domain)
        scheme = IMEX_SDC(base_scheme, domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp, final_update=True,initial_guess="base")
    elif scheme == "IMEX_SDC_R(2,2)":
        node_type="RADAU-RIGHT"
        M = 2
        k = 2
        eqn = ContinuityEquation(domain, V, "f")
        # Split continuity term
        eqn = split_continuity_form(eqn)
        eqn.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
        eqn.label_terms(lambda t: t.has_label(transport), explicit)
        base_scheme = IMEX_Euler(domain)
        scheme = IMEX_SDC(base_scheme, domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp, final_update=True, initial_guess="base")
    elif scheme == "BE_SDC_Lo(3,3)":
        node_type="LOBATTO"
        M = 3
        k = 3
        base_scheme = BackwardEuler(domain)
        scheme = BE_SDC(base_scheme, domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp, final_update=True, initial_guess="base")
    elif scheme == "FE_SDC_Le(3,5)":
        node_type="GAUSS"
        M = 3
        k = 4
        base_scheme = ForwardEuler(domain)
        scheme = FE_SDC(base_scheme, domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp, final_update=True, initial_guess="base")

    transport_method = DGUpwind(eqn, 'f')

    timestepper = PrescribedTransport(eqn, scheme, setup.io, transport_method)

    # Initial conditions
    timestepper.fields("f").interpolate(setup.f_init)
    timestepper.fields("u").project(setup.uexpr)
    assert run(timestepper, setup.tmax, setup.f_end) < setup.tol
