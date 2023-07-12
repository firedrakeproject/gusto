"""
This tests the transport of a vector-valued field using the recovery wrapper.
The computed solution is compared with a true one to check that the transport
is working correctly.
"""

from gusto import *
from firedrake import (as_vector, VectorFunctionSpace, norm)
import pytest


def run(timestepper, tmax, f_end):
    timestepper.run(0, tmax)

    return norm(timestepper.fields("f") - f_end) / norm(f_end)


@pytest.mark.parametrize("geometry", ["slice"])
def test_vector_recovered_space_setup(tmpdir, geometry, tracer_setup):

    # Make domain using routine from conftest
    setup = tracer_setup(tmpdir, geometry, degree=0)
    domain = setup.domain
    mesh = domain.mesh
    gdim = mesh.geometric_dimension()

    # Spaces for recovery
    Vu = domain.spaces("HDiv")
    if geometry == "slice":
        VDG1 = domain.spaces("DG1_equispaced")
        Vec_DG1 = VectorFunctionSpace(mesh, VDG1.ufl_element(), name='Vec_DG1')
        Vec_CG1 = VectorFunctionSpace(mesh, "CG", 1, name='Vec_CG1')

        rec_opts = RecoveryOptions(embedding_space=Vec_DG1,
                                   recovered_space=Vec_CG1,
                                   boundary_method=BoundaryMethod.taylor)
    else:
        raise NotImplementedError(
            f'Recovered spaces for geometry {geometry} have not been implemented')

    # Make equation
    eqn = AdvectionEquation(domain, Vu, "f")
    transport_scheme = SSPRK3(domain, options=rec_opts)
    transport_method = DGUpwind(eqn, "f")
    timestepper = PrescribedTransport(eqn, transport_scheme, setup.io, transport_method)

    # Initialise fields
    f_init = as_vector([setup.f_init]*gdim)
    timestepper.fields("f").project(f_init)
    timestepper.fields("u").project(setup.uexpr)

    f_end = as_vector([setup.f_end]*gdim)

    # Run and check error
    error = run(timestepper, setup.tmax, f_end)
    assert error < setup.tol, \
        'The transport error is greater than the permitted tolerance'
