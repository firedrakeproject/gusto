"""
This tests the dry compressible hydrostatic balance, by setting up a vertical
slice with the appropriate initialisation procedure, before taking a few time
steps and ensuring that the resulting velocities are very small.
"""

from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, Constant
from os import path
from netCDF4 import Dataset


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

    output = OutputParameters(dirname=dirname+'/dry_balance', dumpfreq=10, dumplist=['u'])
    parameters = CompressibleParameters()

    state = State(mesh,
                  dt=dt,
                  output=output,
                  parameters=parameters)

    eqns = CompressibleEulerEquations(state, "CG", 1)

    # Initial conditions
    rho0 = state.fields("rho")
    theta0 = state.fields("theta")

    # Isentropic background state
    Tsurf = Constant(300.)
    theta0.interpolate(Tsurf)

    # Calculate hydrostatic exner
    compressible_hydrostatic_balance(state, theta0, rho0, solve_for_rho=True)

    state.set_reference_profiles([('rho', rho0),
                                  ('theta', theta0)])

    # Set up transport schemes
    transported_fields = [ImplicitMidpoint(state, "u"),
                          SSPRK3(state, "rho"),
                          SSPRK3(state, "theta", options=EmbeddedDGOptions())]

    # Set up linear solver
    linear_solver = CompressibleSolver(state, eqns)

    # build time stepper
    stepper = SemiImplicitQuasiNewton(eqns, state, transported_fields,
                                      linear_solver=linear_solver)

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
