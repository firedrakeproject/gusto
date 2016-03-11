from dcore import *
from firedrake import IcosahedralSphereMesh, Expression

def setup_sw():

    refinements = 3  # number of horizontal cells = 20*(4^refinements)

    R = 6371220.
    u_0 = 20.0  # Maximum amplitude of the zonal wind (m/s)

    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=refinements)
    global_normal = Expression(("x[0]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])",
                                "x[1]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])",
                                "x[2]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])"))
    mesh.init_cell_orientations(global_normal)

    timestepping = TimesteppingParameters()
    output = OutputParameters()
    parameters = ShallowWaterParameters()

    state = ShallowWaterState(mesh, vertical_degree=None, horizontal_degree=2,
                              family="BDM",
                              timestepping=timestepping,
                              output=output,
                              parameters=parameters)

    g = parameters.g
    Omega = parameters.Omega

    # interpolate initial conditions
    # Initial/current conditions
    u0, D0 = Function(state.V[0]), Function(state.V[1])
    uexpr = Expression(("-u_0*x[1]/R", "u_0*x[0]/R", "0.0"), u_0=u_0, R=R)
    Dexpr = Expression("h0 - ((R * Omega * u_0 + u_0*u_0/2.0)*(x[2]*x[2]/(R*R)))/g", h0=2940, R=R, Omega=Omega, u_0=u_0, g=g)

    u0.project(uexpr)
    D0.interpolate(Dexpr)

    state.initialise([u0, D0])

    # names of fields to dump
    state.fieldlist = ('u', 'D')

def run_sw():

    setup_sw()

def test_sw_setup():

    run_sw()
