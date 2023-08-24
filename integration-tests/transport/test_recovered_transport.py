"""
This tests the transport of a scalar-valued field using the recovery wrapper.
The computed solution is compared with a true one to check that the transport
is working correctly.
"""

from gusto import *
from firedrake import FunctionSpace, norm
import pytest


def run(timestepper, tmax, f_end):
    timestepper.run(0, tmax)
    return norm(timestepper.fields("f") - f_end) / norm(f_end)


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
def test_recovered_space_setup(tmpdir, geometry, tracer_setup):

    # Make domain using routine from conftest
    setup = tracer_setup(tmpdir, geometry, degree=0)
    domain = setup.domain
    mesh = domain.mesh

    # Spaces for recovery
    VDG0 = FunctionSpace(mesh, "DG", 0)
    VDG1 = domain.spaces("DG1_equispaced")
    VCG1 = FunctionSpace(mesh, "CG", 1)

    # Make equation
    eqn = ContinuityEquation(domain, VDG0, "f")

    # Declare transport scheme
    recovery_opts = RecoveryOptions(embedding_space=VDG1,
                                    recovered_space=VCG1,
                                    boundary_method=BoundaryMethod.taylor)

    transport_scheme = SSPRK3(domain, options=recovery_opts)
    transport_method = DGUpwind(eqn, "f")

    timestepper = PrescribedTransport(eqn, transport_scheme, setup.io, transport_method)

    # Initialise fields
    timestepper.fields("f").interpolate(setup.f_init)
    timestepper.fields("u").project(setup.uexpr)

    # Run and check error
    error = run(timestepper, setup.tmax, setup.f_end)
    assert error < setup.tol, \
        'The transport error is greater than the permitted tolerance'
