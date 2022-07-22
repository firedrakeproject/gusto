"""
This tests the transport of a vector-valued field using the recovery wrapper.

TODO: This needs to explicitly check that the vector has been transported to
the appropriate place. Use the tracer setup ...
"""

from gusto import *
from firedrake import (as_vector, Constant, PeriodicIntervalMesh,
                       SpatialCoordinate, ExtrudedMesh, FunctionSpace,
                       Function, conditional, sqrt, VectorFunctionSpace,
                       FiniteElement, TensorProductElement, HDiv, interval)

# This setup creates a sharp bubble of warm air in a vertical slice
# This bubble is then transported by a prescribed transport scheme


def run(state, transport_scheme, tmax):
    timestepper = PrescribedTransport(state, transport_scheme)
    timestepper.run(0, tmax)


def test_vector_recovered_space_setup(tmpdir):

    # declare grid shape, with length L and height H
    L = 400.
    H = 400.
    nlayers = int(H / 20.)
    ncolumns = int(L / 20.)

    # make mesh
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=(H / nlayers))

    dt = 1.0
    output = OutputParameters(dirname=str(tmpdir), dumpfreq=5)

    state = State(mesh,
                  dt=dt,
                  output=output)

    # horizontal base spaces
    cell = mesh._base_mesh.ufl_cell().cellname()
    u_hori = FiniteElement("CG", cell, 1, variant="equispaced")
    w_hori = FiniteElement("DG", cell, 0, variant="equispaced")

    # vertical base spaces
    u_vert = FiniteElement("DG", interval, 0, variant="equispaced")
    w_vert = FiniteElement("CG", interval, 1, variant="equispaced")

    # build elements
    u_element = HDiv(TensorProductElement(u_hori, u_vert))
    w_element = HDiv(TensorProductElement(w_hori, w_vert))
    v_element = u_element + w_element

    # spaces
    Vpsi = FunctionSpace(mesh, "CG", 2)
    VDG1 = state.spaces("DG", "DG", 1)
    Vu_DG1 = VectorFunctionSpace(mesh, VDG1.ufl_element(), name='Vec_DG1')
    Vu_CG1 = VectorFunctionSpace(mesh, "CG", 1, name='Vec_CG1')
    Vu = FunctionSpace(mesh, v_element)

    tracereqn = AdvectionEquation(state, Vu, "tracer", ufamily="CG",
                                  udegree=1)

    # initialise fields
    u0 = state.fields("u")

    x, z = SpatialCoordinate(mesh)

    # set up velocity field
    u_max = Constant(10.0)
    psi_expr = - u_max * z
    psi0 = Function(Vpsi).interpolate(psi_expr)

    # make a gradperp
    gradperp = lambda u: as_vector([-u.dx(1), u.dx(0)])

    u0.project(gradperp(psi0))

    # set up bubble
    xc = 200.
    zc = 200.
    rc = 100.
    tracer = state.fields("tracer")
    scalar_expr = conditional(sqrt((x - xc) ** 2.0) < rc,
                              conditional(sqrt((z - zc) ** 2.0) < rc,
                                          Constant(0.2),
                                          Constant(0.0)), Constant(0.0))
    tracer.project(as_vector([scalar_expr, scalar_expr]))

    # set up transport scheme
    recovered_opts = RecoveredOptions(embedding_space=Vu_DG1,
                                      recovered_space=Vu_CG1,
                                      broken_space=Vu,
                                      boundary_method=Boundary_Method.dynamics)

    transport_scheme = [(tracereqn, SSPRK3(state, options=recovered_opts))]

    run(state, transport_scheme, tmax=10)
