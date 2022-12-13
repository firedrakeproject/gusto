"""
This runs an incompressible example with a perturbation in a hydrostatic
atmosphere, and checks the example against a known good checkpointed answer.
"""

from os.path import join, abspath, dirname
from gusto import *
from firedrake import (SpatialCoordinate, PeriodicIntervalMesh, exp,
                       sqrt, ExtrudedMesh, Function, norm)


def run_incompressible(tmpdir):

    dt = 6.0
    tmax = 2*dt
    nlayers = 10  # horizontal layers
    ncols = 10  # number of columns
    Lx = 1000.0
    Lz = 1000.0
    m = PeriodicIntervalMesh(ncols, Lx)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=Lz/nlayers)
    domain = Domain(mesh, dt, "CG", 1)

    parameters = CompressibleParameters()
    eqn = IncompressibleBoussinesqEquations(domain, parameters)

    output = OutputParameters(dirname=tmpdir+"/incompressible",
                              dumpfreq=2, chkptfreq=2)
    io = IO(domain, eqn, output=output)

    # Initial conditions
    p0 = eqn.fields("p")
    b0 = eqn.fields("b")

    # z.grad(bref) = N**2
    x, z = SpatialCoordinate(mesh)
    N = parameters.N
    bref = z*(N**2)

    b_b = Function(b0.function_space()).interpolate(bref)
    incompressible_hydrostatic_balance(eqn, b_b, p0)
    eqn.set_reference_profiles([('p', p0), ('b', b_b)])

    # Add perturbation
    r = sqrt((x-Lx/2)**2 + (z-Lz/2)**2)
    b_pert = 0.1*exp(-(r/(Lx/5)**2))
    b0.interpolate(b_b + b_pert)

    # Set up transport schemes
    b_opts = SUPGOptions()
    transported_fields = [ImplicitMidpoint(domain, "u"),
                          SSPRK3(domain, "b", options=b_opts)]

    # Set up linear solver for the timestepping scheme
    linear_solver = IncompressibleSolver(eqn)

    # build time stepper
    stepper = SemiImplicitQuasiNewton(eqn, io, transported_fields,
                                      linear_solver=linear_solver)

    # Run
    stepper.run(t=0, tmax=tmax)

    # State for checking checkpoints
    checkpoint_name = 'incompressible_chkpt'
    new_path = join(abspath(dirname(__file__)), '..', f'data/{checkpoint_name}')
    check_eqn = IncompressibleBoussinesqEquations(domain, parameters)
    check_eqn.set_reference_profiles([])
    check_output = OutputParameters(dirname=tmpdir+"/incompressible",
                                    checkpoint_pickup_filename=new_path)
    check_io = IO(domain, check_eqn, output=check_output)
    check_stepper = SemiImplicitQuasiNewton(check_eqn, check_io, [])
    check_stepper.run(t=0, tmax=0, pickup=True)

    return eqn, check_eqn


def test_incompressible(tmpdir):

    dirname = str(tmpdir)
    eqn, check_eqn = run_incompressible(dirname)

    for variable in ['u', 'b', 'p']:
        new_variable = eqn.fields(variable)
        check_variable = check_eqn.fields(variable)
        error = norm(new_variable - check_variable) / norm(check_variable)

        # Slack values chosen to be robust to different platforms
        assert error < 1e-10, f'Values for {variable} in ' + \
            'Incompressible test do not match KGO values'
