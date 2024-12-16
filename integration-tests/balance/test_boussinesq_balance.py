"""
This tests the dry compressible hydrostatic balance, by setting up a vertical
slice with the appropriate initialisation procedure, before taking a few time
steps and ensuring that the resulting velocities are very small.
"""

from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh
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
    parameters = BoussinesqParameters()
    eqns = BoussinesqEquations(domain, parameters)

    # I/O
    output = OutputParameters(dirname=dirname+'/boussinesq_balance', dumpfreq=10, dumplist=['u'])
    io = IO(domain, output)

    # Set up transport schemes
    transported_fields = [TrapeziumRule(domain, "u"),
                          SSPRK3(domain, "p"),
                          SSPRK3(domain, "b", options=EmbeddedDGOptions())]
    transport_methods = [DGUpwind(eqns, 'u'),
                         DGUpwind(eqns, 'p'),
                         DGUpwind(eqns, 'b')]

    # Set up linear solver
    linear_solver = BoussinesqSolver(eqns)

    # build time stepper
    stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                      transport_methods,
                                      linear_solver=linear_solver)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    p0 = stepper.fields("p")
    b0 = stepper.fields("b")

    # first setup the background buoyancy profile
    # z.grad(bref) = N**2
    N = parameters.N
    _, z = SpatialCoordinate(mesh)
    bref = z*(N**2)
    # interpolate the expression to the function
    b0.interpolate(bref)

    # Calculate hydrostatic exner
    boussinesq_hydrostatic_balance(eqns, b0, p0)

    stepper.set_reference_profiles([('p', p0),
                                    ('b', b0)])

    return stepper, tmax


def run_balance(dirname):

    stepper, tmax = setup_balance(dirname)
    stepper.run(t=0, tmax=tmax)


def test_balance_setup(tmpdir):

    dirname = str(tmpdir)
    run_balance(dirname)
    filename = path.join(dirname, "boussinesq_balance/diagnostics.nc")
    data = Dataset(filename, "r")

    u = data.groups['u']
    umax = u.variables['max']

    assert umax[-1] < 1e-8
