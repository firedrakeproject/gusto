"""
This runs a simple transport test on the sphere using the SDC time discretisations to
test whether the errors are within tolerance. The test is run for the following schemes:
- IMEX_SDC_Le(1,1) - IMEX SDC with 1 quadrature node of Gauss type (2nd order scheme)
- IMEX_SDC_R(2,2)  - IMEX SDC with 2 qaudrature nodes of Radau type (3rd order scheme) using
                     LU decomposition for the implicit update
- BE_SDC_Lo(3,3)   - Implicit SDC with 3 quadrature nodes of Lobatto type (4th order scheme).
- FE_SDC_Le(3,5)   - Explicit SDC with 3 quadrature nodes of Gauss type (6th order scheme).
"""

from firedrake import norm
from gusto import *
import pytest


def run(timestepper, tmax, f_end):
    timestepper.run(0, tmax)
    return norm(timestepper.fields("f") - f_end) / norm(f_end)


@pytest.mark.parametrize(
    "scheme", ["IMEX_SDC_Le(1,1)", "IMEX_SDC_R(2,2)", "BE_SDC_Lo(3,3)", "FE_SDC_Le(3,5)"])
def test_sdc(tmpdir, scheme, tracer_setup):
    geometry = "sphere"
    setup = tracer_setup(tmpdir, geometry)
    domain = setup.domain
    V = domain.spaces("DG")
    eqn = AdvectionEquation(domain, V, "f")
    node_type = "LEGENDRE"
    qdelta_imp = "BE"
    qdelta_exp = "FE"

    if scheme == "IMEX_SDC_Le(1,1)":
        quad_type = "GAUSS"
        M = 1
        k = 1
        eqn = ContinuityEquation(domain, V, "f")
        # Split continuity term
        eqn = split_continuity_form(eqn)
        eqn.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
        eqn.label_terms(lambda t: t.has_label(transport), explicit)
        base_scheme = IMEX_Euler(domain)
        scheme = SDC(base_scheme, domain, M, k, quad_type, node_type, qdelta_imp,
                     qdelta_exp, final_update=True, initial_guess="base")
    elif scheme == "IMEX_SDC_R(2,2)":
        quad_type = "RADAU-RIGHT"
        M = 2
        k = 2
        qdelta_imp = "LU"
        eqn = ContinuityEquation(domain, V, "f")
        # Split continuity term
        eqn = split_continuity_form(eqn)
        eqn.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
        eqn.label_terms(lambda t: t.has_label(transport), explicit)
        base_scheme = IMEX_Euler(domain)
        scheme = SDC(base_scheme, domain, M, k, quad_type, node_type, qdelta_imp,
                     qdelta_exp, formulation="Z2N", final_update=True, initial_guess="base")
    elif scheme == "BE_SDC_Lo(3,3)":
        quad_type = "LOBATTO"
        M = 3
        k = 3
        eqn.label_terms(lambda t: not t.has_label(time_derivative), implicit)
        base_scheme = BackwardEuler(domain)
        scheme = SDC(base_scheme, domain, M, k, quad_type, node_type, qdelta_imp,
                     qdelta_exp, final_update=True, initial_guess="base")
    elif scheme == "FE_SDC_Le(3,5)":
        quad_type = "GAUSS"
        M = 3
        k = 4
        eqn.label_terms(lambda t: not t.has_label(time_derivative), explicit)
        base_scheme = ForwardEuler(domain)
        scheme = SDC(base_scheme, domain, M, k, quad_type, node_type, qdelta_imp,
                     qdelta_exp, final_update=True, initial_guess="base")

    transport_method = DGUpwind(eqn, 'f')

    timestepper = PrescribedTransport(eqn, scheme, setup.io, transport_method)

    # Initial conditions
    timestepper.fields("f").interpolate(setup.f_init)
    timestepper.fields("u").project(setup.uexpr)
    assert run(timestepper, setup.tmax, setup.f_end) < setup.tol
