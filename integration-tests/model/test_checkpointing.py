"""
Runs a compressible Euler test that uses checkpointing. The test runs for two
timesteps, checkpoints and then starts a new run from the checkpoint file.
"""

from os import path
import numpy as np
from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, norm,
                       SpatialCoordinate, exp, sin, Function, as_vector,
                       pi, CheckpointFile)


def set_up_model_objects(mesh, dt, output):

    domain = Domain(mesh, dt, "CG", 1)

    parameters = CompressibleParameters()
    eqns = CompressibleEulerEquations(domain, parameters)

    io = IO(domain, output)

    # Set up transport schemes
    transported_fields = []
    transported_fields.append(SSPRK3(domain, "u"))
    transported_fields.append(SSPRK3(domain, "rho"))
    transported_fields.append(SSPRK3(domain, "theta"))

    # Set up linear solver
    linear_solver = CompressibleSolver(eqns)

    # build time stepper
    stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                      linear_solver=linear_solver)

    return stepper, eqns


def initialise_fields(eqns, stepper):

    L = 1.e5
    H = 1.0e4  # Height position of the model top

    # Initial conditions
    u0 = stepper.fields("u")
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")

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

    stepper.set_reference_profiles([('rho', rho_b), ('theta', theta_b)])


def test_checkpointing(tmpdir):

    mesh_name = 'checkpointing_mesh'

    # Set up mesh
    nlayers = 5   # horizontal layers
    columns = 15  # number of columns
    L = 3.e5
    m = PeriodicIntervalMesh(columns, L)
    dt = 2.0

    # build volume mesh
    H = 1.0e4  # Height position of the model top
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers, name=mesh_name)

    dirname_1 = str(tmpdir)+'/checkpointing_1'
    dirname_2 = str(tmpdir)+'/checkpointing_2'
    dirname_3 = str(tmpdir)+'/checkpointing_3'

    output_1 = OutputParameters(dirname=dirname_1, dumpfreq=1,
                                chkptfreq=4, log_level='INFO')
    output_2 = OutputParameters(dirname=dirname_2, dumpfreq=1,
                                chkptfreq=2, log_level='INFO')


    stepper_1, eqns_1 = set_up_model_objects(mesh, dt, output_1)
    stepper_2, eqns_2 = set_up_model_objects(mesh, dt, output_2)

    initialise_fields(eqns_1, stepper_1)
    initialise_fields(eqns_2, stepper_2)

    # ------------------------------------------------------------------------ #
    # Run for 4 time steps and store values
    # ------------------------------------------------------------------------ #

    stepper_1.run(t=0.0, tmax=4*dt)

    # ------------------------------------------------------------------------ #
    # Run other timestepper for 2 time steps and checkpoint
    # ------------------------------------------------------------------------ #

    stepper_2.run(t=0.0, tmax=2*dt)

    # ------------------------------------------------------------------------ #
    # Pick up from checkpoint and run *new* timestepper for 2 time steps
    # ------------------------------------------------------------------------ #

    chkpt_2_path = path.join(stepper_2.io.dumpdir, "chkpt.h5")
    output_3 = OutputParameters(dirname=dirname_3, dumpfreq=1,
                                chkptfreq=2, log_level='INFO',
                                checkpoint_pickup_filename=chkpt_2_path)

    mesh = pick_up_mesh(output_3, mesh_name)
    stepper_3, _ = set_up_model_objects(mesh, dt, output_3)

    stepper_3.run(t=2*dt, tmax=4*dt, pickup=True)

    # ------------------------------------------------------------------------ #
    # Pick up from checkpoint and run *same* timestepper for 2 more time steps
    # ------------------------------------------------------------------------ #

    mesh = pick_up_mesh(output_2, mesh_name)
    stepper_2, _ = set_up_model_objects(mesh, dt, output_2)

    stepper_2.run(t=2*dt, tmax=4*dt, pickup=True)

    # ------------------------------------------------------------------------ #
    # Compare fields against saved values for run without checkpointing
    # ------------------------------------------------------------------------ #

    for field_name in ['rho', 'theta', 'u']:
        diff_array = stepper_1.fields(field_name).dat.data - stepper_2.fields(field_name).dat.data
        error = np.linalg.norm(diff_array)
        assert error < 1e-14, \
            f'Checkpointed field {field_name} with same time stepper is not equal to non-checkpointed field'

        diff_array = stepper_1.fields(field_name).dat.data - stepper_3.fields(field_name).dat.data
        error = np.linalg.norm(diff_array)
        assert error < 1e-14, \
            f'Checkpointed field {field_name} with new time stepper is not equal to non-checkpointed field'
