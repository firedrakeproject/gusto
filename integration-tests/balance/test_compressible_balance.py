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

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Parameters
    dt = 1.
    tmax = 5.
    deltax = 400
    L = 2000.
    H = 10000.
    nlayers = int(H/deltax)
    ncolumns = int(L/deltax)

    # Domain
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
    domain = Domain(mesh, dt, "CG", 1)

    # Equation
    parameters = CompressibleParameters(mesh)
    eqns = CompressibleEulerEquations(domain, parameters)

    # I/O
    output = OutputParameters(dirname=dirname+'/dry_balance', dumpfreq=10, dumplist=['u'])
    io = IO(domain, output)

    # Set up transport schemes
    transported_fields = [TrapeziumRule(domain, "u"),
                          SSPRK3(domain, "rho"),
                          SSPRK3(domain, "theta", options=EmbeddedDGOptions())]
    transport_methods = [DGUpwind(eqns, 'u'),
                         DGUpwind(eqns, 'rho'),
                         DGUpwind(eqns, 'theta')]

    # Set up linear solver
    linear_solver = CompressibleSolver(eqns)

    # build time stepper
    stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                      transport_methods,
                                      linear_solver=linear_solver)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")

    # Isentropic background state
    Tsurf = Constant(300.)
    theta0.interpolate(Tsurf)

    # Calculate hydrostatic exner
    compressible_hydrostatic_balance(eqns, theta0, rho0, solve_for_rho=True)

    stepper.set_reference_profiles([('rho', rho0),
                                    ('theta', theta0)])

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
