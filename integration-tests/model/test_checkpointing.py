"""
Runs a compressible Euler test that uses checkpointing. The test runs for two
timesteps, checkpoints and then starts a new run from the checkpoint file.
"""

from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, norm,
                       SpatialCoordinate, exp, sin, Function, as_vector,
                       pi, DumbCheckpoint, FILE_READ)


def setup_checkpointing(dirname):

    nlayers = 5   # horizontal layers
    columns = 15  # number of columns
    L = 3.e5
    m = PeriodicIntervalMesh(columns, L)
    dt = 2.0

    # build volume mesh
    H = 1.0e4  # Height position of the model top
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

    output = OutputParameters(dirname=dirname, dumpfreq=1,
                              chkptfreq=2, log_level='INFO')
    parameters = CompressibleParameters()

    state = State(mesh,
                  dt=dt,
                  output=output,
                  parameters=parameters)

    eqns = CompressibleEulerEquations(state, "CG", 1)

    # Set up transport schemes
    transported_fields = []
    transported_fields.append(SSPRK3(state, "u"))
    transported_fields.append(SSPRK3(state, "rho"))
    transported_fields.append(SSPRK3(state, "theta"))

    # Set up linear solver
    linear_solver = CompressibleSolver(state, eqns)

    # build time stepper
    stepper = SemiImplicitQuasiNewton(state, eqns, transported_fields,
                                      linear_solver=linear_solver)

    return state, stepper, dt


def initialise_fields(state):

    L = 1.e5
    H = 1.0e4  # Height position of the model top

    # Initial conditions
    u0 = state.fields("u")
    rho0 = state.fields("rho")
    theta0 = state.fields("theta")

    # spaces
    Vt = theta0.function_space()
    Vr = rho0.function_space()

    # Thermodynamic constants required for setting initial conditions
    # and reference profiles
    g = state.parameters.g
    N = state.parameters.N

    # N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
    x, z = SpatialCoordinate(state.mesh)
    Tsurf = 300.
    thetab = Tsurf*exp(N**2*z/g)

    theta_b = Function(Vt).interpolate(thetab)
    rho_b = Function(Vr)

    # Calculate hydrostatic exner
    compressible_hydrostatic_balance(state, theta_b, rho_b)

    a = 5.0e3
    deltaTheta = 1.0e-2
    theta_pert = deltaTheta*sin(pi*z/H)/(1 + (x - L/2)**2/a**2)
    theta0.interpolate(theta_b + theta_pert)
    rho0.assign(rho_b)
    u0.project(as_vector([20.0, 0.0]))

    state.set_reference_profiles([('rho', rho_b), ('theta', theta_b)])


def test_checkpointing(tmpdir):

    dirname_1 = str(tmpdir)+'/checkpointing_1'
    dirname_2 = str(tmpdir)+'/checkpointing_2'
    state_1, stepper_1, dt = setup_checkpointing(dirname_1)
    state_2, stepper_2, dt = setup_checkpointing(dirname_2)

    # ------------------------------------------------------------------------ #
    # Run for 4 time steps and store values
    # ------------------------------------------------------------------------ #

    initialise_fields(state_1)
    stepper_1.run(t=0.0, tmax=4*dt)

    # ------------------------------------------------------------------------ #
    # Start again, run for 2 time steps, checkpoint and then run for 2 more
    # ------------------------------------------------------------------------ #

    initialise_fields(state_2)
    stepper_2.run(t=0.0, tmax=2*dt)

    # Wipe fields, then pickup
    state_2.fields('u').project(as_vector([-10.0, 0.0]))
    state_2.fields('rho').interpolate(Constant(0.0))
    state_2.fields('theta').interpolate(Constant(0.0))

    stepper_2.run(t=2*dt, tmax=4*dt, pickup=True)

    # ------------------------------------------------------------------------ #
    # Compare fields against saved values for run without checkpointing
    # ------------------------------------------------------------------------ #

    # Pick up from both stored checkpoint files
    # This is the best way to compare fields from different meshes
    for field_name in ['u', 'rho', 'theta']:
        with DumbCheckpoint(dirname_1+'/chkpt', mode=FILE_READ) as chkpt:
            field_1 = Function(state_1.fields(field_name).function_space(),
                               name=field_name)
            chkpt.load(field_1)
            # These are preserved in the comments for when we can use CheckpointFile
            # mesh = chkpt.load_mesh(name='firedrake_default_extruded')
            # field_1 = chkpt.load_function(mesh, name=field_name)
        with DumbCheckpoint(dirname_2+'/chkpt', mode=FILE_READ) as chkpt:
            field_2 = Function(state_1.fields(field_name).function_space(),
                               name=field_name)
            chkpt.load(field_2)
            # These are preserved in the comments for when we can use CheckpointFile
            # field_2 = chkpt.load_function(mesh, name=field_name)

        error = norm(field_1 - field_2)
        assert error < 1e-15, \
            f'Checkpointed field {field_name} is not equal to non-checkpointed field'
