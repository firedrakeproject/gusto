"""
This runs a simple transport test on the sphere using the DC time discretisations to
test whether the errors are within tolerance. The test is run for the following schemes:
- IMEX_SDC_R(2,2)  - IMEX SDC with 2 qaudrature nodes of Radau type (3rd order scheme) using
- IMEX_RIDC_R(3)   - IMEX RIDC with 4 quadrature nodes of equidistant type, reduced stencils (3rd order scheme).
"""

from firedrake import norm, Ensemble, COMM_WORLD
from gusto import *
import pytest
from pytest_mpi.parallel_assert import parallel_assert


def run(timestepper, tmax, f_end):
    timestepper.run(0, tmax)
    print(norm(timestepper.fields("f") - f_end) / norm(f_end))
    return norm(timestepper.fields("f") - f_end) / norm(f_end)


@pytest.mark.parallel(nprocs=[2, 4])
@pytest.mark.parametrize(
    "scheme", ["IMEX_SDC_R(3,3)", "IMEX_RIDC_R(3)"])
def test_parallel_dc(tmpdir, scheme, tracer_setup):

    if scheme == "IMEX_SDC_R(3,3)":
        M = 2
        k = M
        ensemble = Ensemble(COMM_WORLD, COMM_WORLD.size//(M))
    elif scheme == "IMEX_RIDC_R(3)":
        k = 1
        ensemble = Ensemble(COMM_WORLD, COMM_WORLD.size//(k+1))
    geometry = "sphere"
    setup = tracer_setup(tmpdir, geometry, ensemble=ensemble)
    domain = setup.domain
    V = domain.spaces("DG")

    if scheme == "IMEX_SDC_R(3,3)":
        quad_type = "RADAU-RIGHT"
        node_type = "LEGENDRE"
        qdelta_imp = "MIN-SR-FLEX"
        qdelta_exp = "MIN-SR-NS"
        eqn = ContinuityEquation(domain, V, "f")
        # Split continuity term
        eqn = split_continuity_form(eqn)
        eqn.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
        eqn.label_terms(lambda t: t.has_label(transport), explicit)
        base_scheme = IMEX_Euler(domain)
        scheme = Parallel_SDC(base_scheme, domain, M, k, quad_type, node_type, qdelta_imp,
                              qdelta_exp, final_update=False, initial_guess="copy", communicator=ensemble)
    elif scheme == "IMEX_RIDC_R(3)":
        M = k*(k+1)//2 + 1
        eqn = ContinuityEquation(domain, V, "f")
        # Split continuity term
        eqn = split_continuity_form(eqn)
        eqn.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
        eqn.label_terms(lambda t: t.has_label(transport), explicit)
        base_scheme = IMEX_Euler(domain)
        scheme = Parallel_RIDC(base_scheme, domain, M, k, communicator=ensemble)

    transport_method = DGUpwind(eqn, 'f')

    time_varying_velocity = False
    timestepper = PrescribedTransport(
        eqn, scheme, setup.io, time_varying_velocity, transport_method
    )

    # Initial conditions
    timestepper.fields("f").interpolate(setup.f_init)
    timestepper.fields("u").project(setup.uexpr)
    parallel_assert(run(timestepper, setup.tmax, setup.f_end) < setup.tol, "Error too large")
