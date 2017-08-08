from os import path
from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, as_vector, \
    FunctionSpace, Function
from math import pi
from netCDF4 import Dataset


def setup_sw(dirname):
    refinements = 3  # number of horizontal cells = 20*(4^refinements)

    R = 6371220.
    H = 2000.
    day = 24.*60.*60.

    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=refinements, degree=3)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

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

    # Coriolis
    Omega = parameters.Omega
    fexpr = 2*Omega*x[2]/R
    V = FunctionSpace(mesh, "CG", 1)
    f = state.fields("coriolis", Function(V))
    f.interpolate(fexpr)  # Coriolis frequency (1/s)

    # interpolate initial conditions
    # Initial/current conditions
    u0 = state.fields("u")
    D0 = state.fields("D")
    u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    g = parameters.g
    Dexpr = - ((R * Omega * u_max)*(x[2]*x[2]/(R*R)))/g
    u0.project(uexpr)
    D0.interpolate(Dexpr)
    state.initialise([('u', u0),
                      ('D', D0)])

    Deqn = LinearAdvection(state, D0.function_space(), state.parameters.H, ibp="once", equation_form="continuity")
    advected_fields = []
    advected_fields.append(("u", NoAdvection(state, u0, None)))
    advected_fields.append(("D", ForwardEuler(state, D0, Deqn)))

    linear_solver = ShallowWaterSolver(state)

    # Set up forcing
    sw_forcing = ShallowWaterForcing(state, linear=True)

    # build time stepper
    stepper = Timestepper(state, advected_fields, linear_solver,
                          sw_forcing)

    return stepper, 2*day


def run_sw(dirname):

    stepper, tmax = setup_sw(dirname)
    stepper.run(t=0, tmax=tmax)


def test_sw_linear(tmpdir):
    dirname = str(tmpdir)
    run_sw(dirname)
    filename = path.join(dirname, "sw_linear_w2/diagnostics.nc")
    data = Dataset(filename, "r")

    Derr = data.groups["D_error"]
    D = data.groups["D"]
    Dl2 = Derr["l2"][-1]/D["l2"][0]
    uerr = data.groups["u_error"]
    u = data.groups["u"]
    ul2 = uerr["l2"][-1]/u["l2"][0]

    assert Dl2 < 3.e-3
    assert ul2 < 6.e-2
