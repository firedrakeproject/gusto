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
    domain = Domain(mesh, "CG", 1)

    parameters = CompressibleParameters()
    eqns = CompressibleEulerEquations(domain, parameters)

    output = OutputParameters(dirname=dirname, dumpfreq=1,
                              chkptfreq=2, log_level='INFO')
    io = IO(domain, eqns, dt=dt, output=output)

    initialise_fields(eqns)

    # Set up transport schemes
    transported_fields = []
    transported_fields.append(SSPRK3(domain, io, "u"))
    transported_fields.append(SSPRK3(domain, io, "rho"))
    transported_fields.append(SSPRK3(domain, io, "theta"))

    # Set up linear solver
    linear_solver = CompressibleSolver(eqns, io)

    # build time stepper
    stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                      linear_solver=linear_solver)

    return eqns, stepper, dt


def initialise_fields(eqns):

    L = 1.e5
    H = 1.0e4  # Height position of the model top

    # Initial conditions
    u0 = eqns.fields("u")
    rho0 = eqns.fields("rho")
    theta0 = eqns.fields("theta")

    # spaces
    Vt = theta0.function_space()
    Vr = rho0.function_space()

    # Thermodynamic constants required for setting initial conditions
    # and reference profiles
    g = eqns.parameters.g
    N = eqns.parameters.N

    # N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
    x, z = SpatialCoordinate(eqns.domain.mesh)
    Tsurf = 300.
    thetab = Tsurf*exp(N**2*z/g)

    theta_b = Function(Vt).interpolate(thetab)
    rho_b = Function(Vr)

    # Calculate hydrostatic exner
    compressible_hydrostatic_balance(eqns, theta_b, rho_b)

    a = 5.0e3
    deltaTheta = 1.0e-2
    theta_pert = deltaTheta*sin(pi*z/H)/(1 + (x - L/2)**2/a**2)
    theta0.interpolate(theta_b + theta_pert)
    rho0.assign(rho_b)
    u0.project(as_vector([20.0, 0.0]))

    eqns.set_reference_profiles([('rho', rho_b), ('theta', theta_b)])


def test_checkpointing(tmpdir):

    dirname_1 = str(tmpdir)+'/checkpointing_1'
    dirname_2 = str(tmpdir)+'/checkpointing_2'
    eqns_1, stepper_1, dt = setup_checkpointing(dirname_1)
    eqns_2, stepper_2, dt = setup_checkpointing(dirname_2)

    # ------------------------------------------------------------------------ #
    # Run for 4 time steps and store values
    # ------------------------------------------------------------------------ #

    stepper_1.run(t=0.0, tmax=4*dt)

    # ------------------------------------------------------------------------ #
    # Start again, run for 2 time steps, checkpoint and then run for 2 more
    # ------------------------------------------------------------------------ #

    stepper_2.run(t=0.0, tmax=2*dt)

    # Wipe fields, then pickup
    eqns_2.fields('u').project(as_vector([-10.0, 0.0]))
    eqns_2.fields('rho').interpolate(Constant(0.0))
    eqns_2.fields('theta').interpolate(Constant(0.0))

    stepper_2.run(t=2*dt, tmax=4*dt, pickup=True)

    # ------------------------------------------------------------------------ #
    # Compare fields against saved values for run without checkpointing
    # ------------------------------------------------------------------------ #

    # Pick up from both stored checkpoint files
    # This is the best way to compare fields from different meshes
    for field_name in ['u', 'rho', 'theta']:
        with DumbCheckpoint(dirname_1+'/chkpt', mode=FILE_READ) as chkpt:
            field_1 = Function(eqns_1.fields(field_name).function_space(),
                               name=field_name)
            chkpt.load(field_1)
            # These are preserved in the comments for when we can use CheckpointFile
            # mesh = chkpt.load_mesh(name='firedrake_default_extruded')
            # field_1 = chkpt.load_function(mesh, name=field_name)
        with DumbCheckpoint(dirname_2+'/chkpt', mode=FILE_READ) as chkpt:
            field_2 = Function(eqns_1.fields(field_name).function_space(),
                               name=field_name)
            chkpt.load(field_2)
            # These are preserved in the comments for when we can use CheckpointFile
            # field_2 = chkpt.load_function(mesh, name=field_name)

        error = norm(field_1 - field_2)
        assert error < 1e-15, \
            f'Checkpointed field {field_name} is not equal to non-checkpointed field'
