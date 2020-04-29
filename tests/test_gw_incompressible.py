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

    state = State(mesh,
                  dt=dt,
                  output=output,
                  parameters=parameters)

    eqns = IncompressibleBoussinesqEquations(state, "RTCF", 1)

    # Initial conditions
    p0 = state.fields("p")
    b0 = state.fields("b")

    # z.grad(bref) = N**2
    x, y, z = SpatialCoordinate(mesh)
    N = parameters.N
    bref = z*(N**2)

    b_b = Function(b0.function_space()).interpolate(bref)
    b0.interpolate(b_b)
    incompressible_hydrostatic_balance(state, b0, p0)
    state.initialise([('p', p0),
                      ('b', b0)])

    return state, eqns


def run_gw_incompressible(dirname):

    state, eqns = setup_gw(dirname)
    x = TimeLevelFields(state, [eqns])
    xn = x.n
    forcing = Forcing(eqns, state.dt, alpha=1.)
    forcing.apply(xn, xn, xn(eqns.field_name), label="explicit")
    u = xn('u')
    w = Function(state.spaces("DG")).interpolate(u[2])
    return w


def test_gw(tmpdir):

    dirname = str(tmpdir)
    w = run_gw_incompressible(dirname)
    assert max(abs(w.dat.data.min()), w.dat.data.max()) < 3e-8
