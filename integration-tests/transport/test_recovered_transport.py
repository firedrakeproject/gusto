"""
This tests the transport of a scalar-valued field using the recovery wrapper.

THOUGHTS: This needs to explicitly check that the field has been transported to
the appropriate place. Use the tracer setup ...
"""

from gusto import *
from firedrake import (as_vector, Constant, PeriodicIntervalMesh,
                       SpatialCoordinate, ExtrudedMesh, FunctionSpace,
                       Function, conditional, sqrt)

# This setup creates a sharp bubble of warm air in a vertical slice
# This bubble is then transported by a prescribed transport scheme


def run(state, transport_scheme, tmax):
    timestepper = PrescribedTransport(state, transport_scheme)
    timestepper.run(0, tmax)


def test_recovered_space_setup(tmpdir):

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

    # spaces
    Vpsi = FunctionSpace(mesh, "CG", 2)
    VDG0 = FunctionSpace(mesh, "DG", 0)
    VDG1 = state.spaces("DG", "DG", 1)
    VCG1 = FunctionSpace(mesh, "CG", 1)

    tracereqn = ContinuityEquation(state, VDG0, "tracer", ufamily="CG",
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
    tracer.interpolate(conditional(sqrt((x - xc) ** 2.0) < rc,
                                   conditional(sqrt((z - zc) ** 2.0) < rc,
                                               Constant(0.2),
                                               Constant(0.0)), Constant(0.0)))

    # set up transport scheme
    recovered_opts = RecoveredOptions(embedding_space=VDG1,
                                      recovered_space=VCG1,
                                      broken_space=VDG0,
                                      boundary_method=Boundary_Method.dynamics)

    transport_scheme = [(tracereqn, SSPRK3(state, options=recovered_opts))]

    run(state, transport_scheme, tmax=10)
