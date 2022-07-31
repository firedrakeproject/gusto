"""
This tests the transport of a vector-valued field using the recovery wrapper.
The computed solution is compared with a true one to check that the transport
is working correctly.
"""

from gusto import *
from firedrake import (as_vector, FunctionSpace, VectorFunctionSpace,
                       BrokenElement, norm)
import pytest

def run(state, transport_scheme, tmax, f_end):
    timestepper = PrescribedTransport(state, transport_scheme)
    timestepper.run(0, tmax)
    return norm(state.fields("f") - f_end) / norm(f_end)


@pytest.mark.parametrize("geometry", ["slice"])
def test_vector_recovered_space_setup(tmpdir, geometry, tracer_setup):

    # Make mesh and state using routine from conftest
    setup = tracer_setup(tmpdir, geometry, degree=0)
    state = setup.state
    mesh = state.mesh
    gdim = state.mesh.geometric_dimension()

    # Spaces for recovery
    Vu = state.spaces("HDiv", family=setup.family, degree=setup.degree)
    if geometry == "slice":
        VDG1 = state.spaces("DG", "DG", 1)
        Vec_DG1 = VectorFunctionSpace(mesh, VDG1.ufl_element(), name='Vec_DG1')
        Vec_CG1 = VectorFunctionSpace(mesh, "CG", 1, name='Vec_CG1')
        Vu_brok = FunctionSpace(mesh, BrokenElement(Vu.ufl_element()))

        rec_opts = RecoveredOptions(embedding_space=Vec_DG1,
                                    recovered_space=Vec_CG1,
                                    broken_space=Vu_brok,
                                    boundary_method=Boundary_Method.dynamics)
    else:
        raise NotImplementedError(
            f'Recovered spaces for geometry {geometry} have not been implemented')

    # Make equation
    eqn = AdvectionEquation(state, Vu, "f",
                            ufamily=setup.family, udegree=1)

    # Initialise fields
    f_init = as_vector([setup.f_init]*gdim)
    state.fields("f").project(f_init)
    state.fields("u").project(setup.uexpr)

    transport_scheme = [(eqn, SSPRK3(state, options=rec_opts))]

    f_end = as_vector([setup.f_end]*gdim)

    # Run and check error
    error = run(state, transport_scheme, setup.tmax, f_end)
    assert error < setup.tol, \
        'The transport error is greater than the permitted tolerance'
