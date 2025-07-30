"""
This runs a simple transport test on the sphere using the parallel DC time discretisations to
test whether the errors are within tolerance. The test is run for the following schemes:
- IMEX_SDC(2,2)  - IMEX SDC with 2 qaudrature nodes of Radau type (3rd order scheme) using
- IMEX_RIDC(3)   - IMEX RIDC with 4 quadrature nodes of equidistant type, reduced stencils (3rd order scheme).
"""

from firedrake import (norm, Ensemble, COMM_WORLD, SpatialCoordinate,
                       as_vector, pi, exp, IcosahedralSphereMesh)

from gusto import *
import pytest
from pytest_mpi.parallel_assert import parallel_assert


def run(timestepper, tmax, f_end):
    timestepper.run(0, tmax)
    print(norm(timestepper.fields("f") - f_end) / norm(f_end))
    return norm(timestepper.fields("f") - f_end) / norm(f_end)


@pytest.mark.parallel(nprocs=[6])
@pytest.mark.parametrize(
    "scheme", ["IMEX_SDC(3,3)", "IMEX_RIDC(3)"])
def test_parallel_dc(tmpdir, scheme):

    if scheme == "IMEX_SDC(3,3)":
        M = 3
        k = M
        ensemble = Ensemble(COMM_WORLD, COMM_WORLD.size//(M))
    elif scheme == "IMEX_RIDC(3)":
        k = 2
        ensemble = Ensemble(COMM_WORLD, COMM_WORLD.size//(k+1))

    # Get the tracer setup
    radius = 1
    dirname = str(tmpdir)
    mesh = IcosahedralSphereMesh(
        radius=radius,
        refinement_level=3,
        degree=1,
        comm=ensemble.comm
    )
    x = SpatialCoordinate(mesh)

    # Parameters chosen so that dt != 1
    # Gaussian is translated from (lon=pi/2, lat=0) to (lon=0, lat=0)
    # to demonstrate that transport is working correctly

    dt = pi/3. * 0.02

    output = OutputParameters(dirname=dirname, dump_vtus=False, dump_nc=True, dumpfreq=15)
    domain = Domain(mesh, dt, family="BDM", degree=1)
    io = IO(domain, output)

    umax = 1.0
    uexpr = as_vector([- umax * x[1] / radius, umax * x[0] / radius, 0.0])

    tmax = pi/2
    f_init = exp(-x[2]**2 - x[0]**2)
    f_end = exp(-x[2]**2 - x[1]**2)

    tol = 0.05

    domain = domain
    V = domain.spaces("DG")
    eqn = ContinuityEquation(domain, V, "f")
    eqn = split_continuity_form(eqn)
    eqn.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
    eqn.label_terms(lambda t: t.has_label(transport), explicit)

    if scheme == "IMEX_SDC(3,3)":
        quad_type = "RADAU-RIGHT"
        node_type = "LEGENDRE"
        qdelta_imp = "MIN-SR-FLEX"
        qdelta_exp = "MIN-SR-NS"
        base_scheme = IMEX_Euler(domain)
        time_scheme = Parallel_SDC(base_scheme, domain, M, k, quad_type, node_type, qdelta_imp,
                                   qdelta_exp, final_update=True, initial_guess="copy", communicator=ensemble)
    elif scheme == "IMEX_RIDC(3)":
        M = k*(k+1)//2 + 4
        base_scheme = IMEX_Euler(domain)
        time_scheme = Parallel_RIDC(base_scheme, domain, M, k, communicator=ensemble)

    transport_method = DGUpwind(eqn, 'f')

    time_varying_velocity = False
    timestepper = PrescribedTransport(
        eqn, time_scheme, io, time_varying_velocity, transport_method
    )

    timestepper.fields("f").interpolate(f_init)
    timestepper.fields("u").project(uexpr)
    error = run(timestepper, tmax, f_end)
    parallel_assert(error < tol, f"Error too large, Error: {error}, tol: {tol}")
