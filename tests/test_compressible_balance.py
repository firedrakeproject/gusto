from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, Constant
from os import path
from netCDF4 import Dataset

# this tests the dry compressible hydrostatic balance, by setting up a vertical slice
# with this initial procedure, before taking a few time steps and ensuring that
# the resulting velocities are very small


def setup_balance(dirname):

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
    output = OutputParameters(dirname=dirname+'/dry_balance', dumpfreq=10, dumplist=['u'])
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

    # spaces
    Vu = u0.function_space()
    Vt = theta0.function_space()
    Vr = rho0.function_space()

    # Isentropic background state
    Tsurf = Constant(300.)
    theta0.interpolate(Tsurf)

    # Calculate hydrostatic Pi
    compressible_hydrostatic_balance(state, theta0, rho0, solve_for_rho=True)

    state.initialise([('u', u0),
                      ('rho', rho0),
                      ('theta', theta0)])
    state.set_reference_profiles([('rho', rho0),
                                  ('theta', theta0)])

    # Set up advection schemes
    ueqn = VectorInvariant(state, Vu)
    rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
    thetaeqn = EmbeddedDGAdvection(state, Vt, equation_form="advective", options=EmbeddedDGOptions())

    advected_fields = [("u", ThetaMethod(state, u0, ueqn)),
                       ("rho", SSPRK3(state, rho0, rhoeqn)),
                       ("theta", SSPRK3(state, theta0, thetaeqn))]

    # Set up linear solver
    linear_solver = CompressibleSolver(state)

    # Set up forcing
    compressible_forcing = CompressibleForcing(state)

    # build time stepper
    stepper = CrankNicolson(state, advected_fields, linear_solver,
                            compressible_forcing)

    return stepper, tmax


def run_balance(dirname):

    stepper, tmax = setup_balance(dirname)
    stepper.run(t=0, tmax=tmax)


def test_balance_setup(tmpdir):

    dirname = str(tmpdir)
    run_balance(dirname)
    filename = path.join(dirname, "dry_balance/diagnostics.nc")
    data = Dataset(filename, "r")

    u = data.groups['u']
    umax = u.variables['max']

    assert umax[-1] < 1e-8
