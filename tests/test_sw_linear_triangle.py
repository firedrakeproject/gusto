from dcore import *
from firedrake import IcosahedralSphereMesh, Expression, SpatialCoordinate, \
    Constant, as_vector
from math import pi
import json


def setup_sw(dirname):
    refinements = 3  # number of horizontal cells = 20*(4^refinements)

    R = 6371220.
    H = 2000.
    day = 24.*60.*60.
    u_0 = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)

    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=refinements, degree=3)
    global_normal = Expression(("x[0]", "x[1]", "x[2]"))
    mesh.init_cell_orientations(global_normal)

    fieldlist = ['u', 'D']
    timestepping = TimesteppingParameters(dt=3600.)
    output = OutputParameters(dirname=dirname+"/sw_linear_w2", steady_state_dump_err=True)
    parameters = ShallowWaterParameters(H=H)
    diagnostics = Diagnostics(fieldlist)

    state = ShallowWaterState(mesh, vertical_degree=None, horizontal_degree=1,
                              family="BDM",
                              timestepping=timestepping,
                              output=output,
                              parameters=parameters,
                              diagnostics=diagnostics,
                              fieldlist=fieldlist)

    g = parameters.g
    Omega = parameters.Omega

    # Coriolis expression
    R = Constant(R)
    Omega = Constant(parameters.Omega)
    x = SpatialCoordinate(mesh)
    fexpr = 2*Omega*x[2]/R
    V = FunctionSpace(mesh, "CG", 1)
    state.f = Function(V).interpolate(fexpr)  # Coriolis frequency (1/s)
    u_max = Constant(u_0)

    # interpolate initial conditions
    # Initial/current conditions
    u0, D0 = Function(state.V[0]), Function(state.V[1])
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    g = Constant(parameters.g)
    Dexpr = - ((R * Omega * u_max)*(x[2]*x[2]/(R*R)))/g
    u0.project(uexpr)
    D0.interpolate(Dexpr)
    state.initialise([u0, D0])

    advection_list = []
    velocity_advection = NoAdvection(state)
    advection_list.append((velocity_advection, 0))
    D_advection = NoAdvection(state)
    advection_list.append((D_advection, 1))

    linear_solver = ShallowWaterSolver(state)

    # Set up forcing
    sw_forcing = ShallowWaterForcing(state)

    # build time stepper
    stepper = Timestepper(state, advection_list, linear_solver,
                          sw_forcing)

    return stepper, 5*day


def run_sw(dirname):

    stepper, tmax = setup_sw(dirname)
    stepper.run(t=0, tmax=tmax)


def test_sw_linear(tmpdir):
    dirname = str(tmpdir)
    run_sw(dirname)
    with open(path.join(dirname, "sw_linear_w2/diagnostics.json"), "r") as f:
        data = json.load(f)
    Dl2 = data["Derr"]["l2"][-1]/data["D"]["l2"][0]
    ul2 = data["uerr"]["l2"][-1]/data["u"]["l2"][0]

    assert Dl2 < 1.e-3
    assert ul2 < 6.e-2
