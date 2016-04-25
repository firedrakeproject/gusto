from dcore import *
from firedrake import IcosahedralSphereMesh, Expression, SpatialCoordinate, \
    Constant, as_vector


def setup_sw(dirname):

    refinements = 3  # number of horizontal cells = 20*(4^refinements)

    R = 6371220.
    u_0 = 20.0  # Maximum amplitude of the zonal wind (m/s)

    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=refinements)
    global_normal = Expression(("x[0]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])",
                                "x[1]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])",
                                "x[2]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])"))
    mesh.init_cell_orientations(global_normal)

    fieldlist = ['u', 'D']
    timestepping = TimesteppingParameters()
    output = OutputParameters(dumplist=(True,True), dirname=dirname)
    parameters = ShallowWaterParameters()

    state = ShallowWaterState(mesh, vertical_degree=None, horizontal_degree=2,
                              family="BDM",
                              timestepping=timestepping,
                              output=output,
                              parameters=parameters,
                              fieldlist=fieldlist)

    # interpolate initial conditions
    u0, D0 = Function(state.V[0]), Function(state.V[1])
    x = SpatialCoordinate(mesh)
    u_max = Constant(u_0)
    R = Constant(R)
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    h0 = Constant(2940)
    Omega = Constant(parameters.Omega)
    g = Constant(parameters.g)
    Dexpr = h0 - ((R * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R*R)))/g

    u0.project(uexpr)
    D0.interpolate(Dexpr)

    state.initialise([u0, D0])


def run_sw(dirname):

    setup_sw(dirname)


def test_sw_setup(tmpdir):

    dirname = str(tmpdir)
    run_sw(dirname)
