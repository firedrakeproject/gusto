"""
This tests the transport of a scalar-valued field using the recovery wrapper.
The computed solution is compared with a true one to check that the transport
is working correctly.
"""

from gusto import *
from firedrake import FunctionSpace, norm
import pytest


def run(eqn, transport_scheme, state, tmax, f_end):
    timestepper = PrescribedTransport(eqn, transport_scheme, state)
    timestepper.run(0, tmax)
    return norm(state.fields("f") - f_end) / norm(f_end)


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
def test_recovered_space_setup(tmpdir, geometry, tracer_setup):

    # Make mesh and state using routine from conftest
    setup = tracer_setup(tmpdir, geometry, degree=0)
    state = setup.state
    mesh = state.mesh

    # Spaces for recovery
    VDG0 = FunctionSpace(mesh, "DG", 0)
    VDG1 = state.spaces("DG1_equispaced")
    VCG1 = FunctionSpace(mesh, "CG", 1)

    # Make equation
    eqn = ContinuityEquation(state, VDG0, "f",
                             ufamily=setup.family, udegree=1)

    # Initialise fields
    state.fields("f").interpolate(setup.f_init)
    state.fields("u").project(setup.uexpr)

    # Declare transport scheme
    recovery_opts = RecoveryOptions(embedding_space=VDG1,
                                    recovered_space=VCG1,
                                    boundary_method=BoundaryMethod.taylor)

    transport_scheme = SSPRK3(state, options=recovery_opts)

    # Run and check error
    error = run(eqn, transport_scheme, state, setup.tmax, setup.f_end)
    assert error < setup.tol, \
        'The transport error is greater than the permitted tolerance'
