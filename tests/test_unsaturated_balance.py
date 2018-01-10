from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, Constant, Function
from os import path
from netCDF4 import Dataset


def setup_unsaturated(dirname):

    # set up grid and time stepping parameters
    dt = 1.
    tmax = 5.
    deltax = 400
    L = 2000.
    H = 10000.

    nlayers = int(H/deltax)
    ncolumns = int(L/deltax)

    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

    fieldlist = ['u', 'rho', 'theta']
    timestepping = TimesteppingParameters(dt=dt, maxk=4, maxi=1)
    output = OutputParameters(dirname=dirname+'/unsaturated_balance', dumpfreq=1, dumplist=['u'])
    parameters = CompressibleParameters()
    diagnostics = Diagnostics(*fieldlist)
    diagnostic_fields = []

    state = State(mesh, vertical_degree=1, horizontal_degree=1,
                  family="CG",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  diagnostics=diagnostics,
                  fieldlist=fieldlist,
                  diagnostic_fields=diagnostic_fields)

    # Initial conditions
    u0 = state.fields("u")
    rho0 = state.fields("rho")
    theta0 = state.fields("theta")
    water_v0 = state.fields("water_v", theta0.function_space())
    water_c0 = state.fields("water_c", theta0.function_space())
    moisture = ['water_v', 'water_c']

    # spaces
    Vu = u0.function_space()
    Vt = theta0.function_space()
    Vr = rho0.function_space()

    # Isentropic background state
    Tsurf = Constant(300.)
    humidity = Constant(0.5)
    theta_d = Function(Vt).interpolate(Tsurf)
    RH = Function(Vt).interpolate(humidity)

    # Calculate hydrostatic Pi
    unsaturated_hydrostatic_balance(state, theta_d, RH)
    water_c0.assign(0.0)

    state.initialise([('u', u0),
                      ('rho', rho0),
                      ('theta', theta0),
                      ('water_v', water_v0),
                      ('water_c', water_c0)])
    state.set_reference_profiles([('rho', rho0),
                                  ('theta', theta0)])

    # Set up advection schemes
    ueqn = EulerPoincare(state, Vu)
    rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
    thetaeqn = EmbeddedDGAdvection(state, Vt, equation_form="advective")

    advected_fields = [("u", ThetaMethod(state, u0, ueqn)),
                       ("rho", SSPRK3(state, rho0, rhoeqn)),
                       ("theta", SSPRK3(state, theta0, thetaeqn)),
                       ("water_v", SSPRK3(state, water_v0, thetaeqn)),
                       ("water_c", SSPRK3(state, water_c0, thetaeqn))]

    # Set up linear solver
    linear_solver = CompressibleSolver(state, moisture=moisture)

    # Set up forcing
    compressible_forcing = CompressibleForcing(state, moisture=moisture)

    # Set up physics
    physics_list = [Condensation(state)]

    # build time stepper
    stepper = CrankNicolson(state, advected_fields, linear_solver,
                            compressible_forcing, physics_list=physics_list)

    return stepper, tmax


def run_unsaturated(dirname):

    stepper, tmax = setup_unsaturated(dirname)
    stepper.run(t=0, tmax=tmax)


def test_unsaturated_setup(tmpdir):

    dirname = str(tmpdir)
    run_unsaturated(dirname)
    filename = path.join(dirname, "unsaturated_balance/diagnostics.nc")
    data = Dataset(filename, "r")

    u = data.groups['u']
    umax = u.variables['max']

    assert umax[-1] < 1e-5
