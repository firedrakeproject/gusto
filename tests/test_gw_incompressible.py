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

    fieldlist = ['u', 'p', 'b']
    output = OutputParameters(dirname=dirname+"/gw_incompressible", dumplist=['u'], dumpfreq=5)
    parameters = CompressibleParameters()

    state = State(mesh, vertical_degree=1, horizontal_degree=1,
                  family="RTCF",
                  dt=dt,
                  output=output,
                  parameters=parameters,
                  fieldlist=fieldlist)

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

    # Set up forcing
    forcing = IncompressibleForcing(state)

    return state, forcing


def run_gw_incompressible(dirname):

    state, forcing = setup_gw(dirname)
    dt = state.dt
    x = TimeLevelFields(state)
    xn = x.n
    forcing.apply(dt, xn, xn, xn)
    u = xn('u')
    w = Function(state.spaces("DG")).interpolate(u[2])
    return w


def test_gw(tmpdir):

    dirname = str(tmpdir)
    w = run_gw_incompressible(dirname)
    assert max(abs(w.dat.data.min()), w.dat.data.max()) < 3e-8
