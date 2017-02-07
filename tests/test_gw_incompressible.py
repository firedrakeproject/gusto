from gusto import *
from firedrake import SpatialCoordinate, PeriodicRectangleMesh, ExtrudedMesh


def setup_gw(dirname):
    nlayers = 10  # horizontal layers
    columns = 30  # number of columns
    L = 1.e5
    m = PeriodicRectangleMesh(columns, 1, L, 1.e4, quadrilateral=True)
    dt = 6.0

    # build volume mesh
    H = 1.0e4  # Height position of the model top
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

    # vertical coordinate and normal
    x, y, z = SpatialCoordinate(mesh)
    k = Constant([0, 0, 1])

    fieldlist = ['u', 'p', 'b']
    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dirname=dirname+"/gw_incompressible", dumplist=['u'], dumpfreq=5)
    parameters = CompressibleParameters(geopotential=False)

    state = IncompressibleState(mesh, vertical_degree=1, horizontal_degree=1,
                                family="RTCF",
                                z=z, k=k,
                                timestepping=timestepping,
                                output=output,
                                parameters=parameters,
                                fieldlist=fieldlist,
                                on_sphere=False)

    # Initial conditions
    u0, p0, b0 = Function(state.V[0]), Function(state.V[1]), Function(state.V[2])

    # z.grad(bref) = N**2
    N = parameters.N
    bref = z*(N**2)

    b_b = Function(state.V[2]).interpolate(bref)
    b0.interpolate(b_b)
    incompressible_hydrostatic_balance(state, b0, p0)
    state.initialise([u0, p0, b0])

    # Set up forcing
    forcing = IncompressibleForcing(state)

    return state, forcing


def run_gw_incompressible(dirname):

    state, forcing = setup_gw(dirname)
    dt = state.timestepping.dt
    forcing.apply(dt, state.xn, state.xn, state.xn)
    u = state.xn.split()[0]
    w = Function(state.V[1]).interpolate(u[2])
    return w


def test_gw(tmpdir):

    dirname = str(tmpdir)
    w = run_gw_incompressible(dirname)
    print w.dat.data.min(), w.dat.data.max()
    assert max(abs(w.dat.data.min()), w.dat.data.max()) < 3e-8
