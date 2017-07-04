from gusto import *
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
    output = OutputParameters(dirname=dirname+"/sw_linear_w2", steady_state_error_fields=['u', 'D'], dumpfreq=12)
    parameters = ShallowWaterParameters(H=H)
    diagnostics = Diagnostics(*fieldlist)

    state = State(mesh, horizontal_degree=1,
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
    f = state.fields("coriolis", Function(V))
    f.interpolate(fexpr)  # Coriolis frequency (1/s)
    u_max = Constant(u_0)

    # interpolate initial conditions
    # Initial/current conditions
    u0 = state.fields("u")
    D0 = state.fields("D")
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    g = Constant(parameters.g)
    Dexpr = - ((R * Omega * u_max)*(x[2]*x[2]/(R*R)))/g
    u0.project(uexpr)
    D0.interpolate(Dexpr)
    state.initialise({'u': u0, 'D': D0})

    Deqn = LinearAdvection(state, D0.function_space(), state.parameters.H, ibp="once", equation_form="continuity")
    advection_dict = {}
    advection_dict["u"] = NoAdvection(state, u0, None)
    advection_dict["D"] = ForwardEuler(state, D0, Deqn)

    linear_solver = ShallowWaterSolver(state)

    # Set up forcing
    sw_forcing = ShallowWaterForcing(state, linear=True)

    # build time stepper
    stepper = Timestepper(state, advection_dict, linear_solver,
                          sw_forcing)

    return stepper, 2*day


def run_sw(dirname):

    stepper, tmax = setup_sw(dirname)
    stepper.run(t=0, tmax=tmax)


def test_sw_linear(tmpdir):
    dirname = str(tmpdir)
    run_sw(dirname)
    with open(path.join(dirname, "sw_linear_w2/diagnostics.json"), "r") as f:
        data = json.load(f)
    Dl2 = data["D_error"]["l2"][-1]/data["D"]["l2"][0]
    ul2 = data["u_error"]["l2"][-1]/data["u"]["l2"][0]

    assert Dl2 < 3.e-3
    assert ul2 < 6.e-2
