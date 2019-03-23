from gusto import *
from firedrake import (SpatialCoordinate, PeriodicRectangleMesh,
                       ExtrudedMesh, Function)


def setup_gw(dirname):
    nlayers = 10  # horizontal layers
    columns = 30  # number of columns
    L = 1.e5
    m = PeriodicRectangleMesh(columns, 1, L, 1.e4, quadrilateral=True)
    dt = 6.0

    # build volume mesh
    H = 1.0e4  # Height position of the model top
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

    output = OutputParameters(dirname=dirname+"/gw_incompressible", dumplist=['u'], dumpfreq=5)
    parameters = CompressibleParameters()

    state = State(mesh, dt=dt,
                  output=output,
                  parameters=parameters)

    eqns = IncompressibleBoussinesqEquations(state, "RTCF", 1, 1)

    # Initial conditions
    u0 = state.fields("u")
    p0 = state.fields("p")
    b0 = state.fields("b")

    # z.grad(bref) = N**2
    x, y, z = SpatialCoordinate(mesh)
    N = parameters.N
    bref = z*(N**2)

    b_b = Function(b0.function_space()).interpolate(bref)
    b0.interpolate(b_b)
    incompressible_hydrostatic_balance(state, b0, p0)
    state.initialise([('u', u0),
                      ('p', p0),
                      ('b', b0)])

    forcing = Forcing(eqns, dt, 0.5)
    return state, eqns, forcing


def run_gw_incompressible(dirname):

    state, eqns, forcing = setup_gw(dirname)
    print(eqns.field_name)
    xn = Function(eqns.function_space)
    xn.assign(state.fields(eqns.field_name))
    forcing.apply(xn, xn, xn, "explicit")
    u = xn.split()[0]
    w = Function(state.spaces("DG")).interpolate(u[2])
    return w


def test_gw(tmpdir):

    dirname = str(tmpdir)
    w = run_gw_incompressible(dirname)
    assert max(abs(w.dat.data.min()), w.dat.data.max()) < 3e-8
