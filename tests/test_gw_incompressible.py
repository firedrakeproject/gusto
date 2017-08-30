from gusto import *
from firedrake import SpatialCoordinate, PeriodicRectangleMesh, ExtrudedMesh, \
    Function


def setup_gw(dirname):
    nlayers = 10  # horizontal layers
    ncolumns = 30  # number of columns
    L = 1.e5
    H = 1.0e4  # Height position of the model top
    physical_domain = VerticalSlice(H=H, L=L, ncolumns=ncolumns, nlayers=nlayers)

    timestepping = TimesteppingParameters(dt=6.0)
    output = OutputParameters(dirname=dirname+"/gw_incompressible", dumplist=['u'], dumpfreq=5)

    state = IncompressibleEulerState(physical_domain.mesh,
                                     output=output)

    model = IncompressibleEulerModel(state, physical_domain, is_rotating=False,
                                     timestepping=timestepping)

    # Initial conditions
    u0 = state.fields("u")
    p0 = state.fields("p")
    b0 = state.fields("b")

    # z.grad(bref) = N**2
    x, z = SpatialCoordinate(physical_domain.mesh)
    N = model.parameters.N
    bref = z*(N**2)

    b_b = Function(b0.function_space()).interpolate(bref)
    b0.interpolate(b_b)
    incompressible_hydrostatic_balance(state, physical_domain.vertical_normal, b0, p0)
    state.initialise([('u', u0),
                      ('p', p0),
                      ('b', b0)])

    return model


def run_gw_incompressible(dirname):

    model = setup_gw(dirname)
    dt = model.timestepping.dt
    xn = model.state.xn
    model.forcing.apply(dt, xn, xn, xn)
    u = xn.split()[0]
    w = Function(model.state.spaces("DG")).interpolate(u[1])
    return w


def test_gw(tmpdir):

    dirname = str(tmpdir)
    w = run_gw_incompressible(dirname)
    assert max(abs(w.dat.data.min()), w.dat.data.max()) < 3e-8
