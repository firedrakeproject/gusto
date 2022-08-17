"""
This runs an incompressible example with a perturbation in a hydrostatic
atmosphere, and checks the example against a known good checkpointed answer.
"""

from os import path
from gusto import *
from firedrake import (SpatialCoordinate, PeriodicIntervalMesh, exp,
                       sqrt, ExtrudedMesh, Function, norm)

def run_incompressible(dirname):

    dt = 6.0
    tmax = 2*dt
    nlayers = 10  # horizontal layers
    ncols = 10  # number of columns
    Lx = 1000.0
    Lz = 1000.0
    m = PeriodicIntervalMesh(ncols, Lx)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=Lz/nlayers)

    output = OutputParameters(dirname=dirname+"/incompressible",
                              dumpfreq=2, chkptfreq=2)
    parameters = CompressibleParameters()

    state = State(mesh,
                  dt=dt,
                  output=output,
                  parameters=parameters)

    eqns = IncompressibleBoussinesqEquations(state, "CG", 1)

    # Initial conditions
    p0 = state.fields("p")
    b0 = state.fields("b")

    # z.grad(bref) = N**2
    x, z = SpatialCoordinate(mesh)
    N = parameters.N
    bref = z*(N**2)

    b_b = Function(b0.function_space()).interpolate(bref)
    incompressible_hydrostatic_balance(state, b_b, p0)
    state.initialise([('p', p0),
                      ('b', b_b)])

    # Add perturbation
    r = sqrt((x-Lx/2)**2 + (z-Lz/2)**2)
    b_pert = 0.1*exp(-(r/(Lx/5)**2))
    b0.interpolate(b_b + b_pert)

    # Set up transport schemes
    b_opts = SUPGOptions()
    transported_fields = [ImplicitMidpoint(state, "u"),
                          SSPRK3(state, "b", options=b_opts)]

    # Set up linear solver for the timestepping scheme
    linear_solver = IncompressibleSolver(state, eqns)

    # build time stepper
    stepper = CrankNicolson(state, eqns, transported_fields,
                            linear_solver=linear_solver)

    # Run
    stepper.run(t=0, tmax=tmax)

    # State for checking checkpoints
    import gusto
    checkpoint_name = 'incompressible_chkpt'
    new_path = path.join(path.split(path.split(gusto.__file__)[0])[0], f'integration-tests/data/{checkpoint_name}')
    check_output = OutputParameters(dirname=dirname+"/incompressible",
                                    checkpoint_pickup_filename=new_path)
    check_state = State(mesh, dt=dt, output=check_output)
    check_eqn = IncompressibleBoussinesqEquations(check_state, "CG", 1)
    # TODO: Would like to use a normal TimeStepper here but then get into problems
    # with eqns needing to be part of a list of a list
    check_stepper = CrankNicolson(check_state, check_eqn, [])
    check_stepper.run(t=0, tmax=0, pickup=True)

    return state, check_state

def test_incompressible(tmpdir):

    dirname = str(tmpdir)
    state, check_state = run_incompressible(dirname)

    for variable in ['u','b','p']:
        new_variable = state.fields(variable)
        check_variable = check_state.fields(variable)
        error = norm(new_variable - check_variable) / norm(check_variable)

        assert error < 1e-12, f'Values for {variable} in ' + \
            'Incompressible test do not match KGO values'
