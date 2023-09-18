"""
Runs a compressible Euler test that uses checkpointing. The test runs for two
timesteps, checkpoints and then starts a new run from the checkpoint file.
"""

from os import path
import numpy as np
from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, pi,
                       SpatialCoordinate, exp, sin, Function, as_vector)
import pytest


def set_up_model_objects(mesh, dt, output, stepper_type):

    domain = Domain(mesh, dt, "CG", 1)

    parameters = CompressibleParameters()
    eqns = CompressibleEulerEquations(domain, parameters)

    # Have two diagnostic fields that depend on initial values -- check if
    # these diagnostics are preserved by checkpointing
    diagnostic_fields = [SteadyStateError('rho'), Perturbation('theta')]

    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    transport_methods = [DGUpwind(eqns, 'u'),
                         DGUpwind(eqns, 'rho'),
                         DGUpwind(eqns, 'theta')]

    if stepper_type == 'semi_implicit':
        # Set up transport schemes
        transported_fields = [SSPRK3(domain, "u"),
                              SSPRK3(domain, "rho"),
                              SSPRK3(domain, "theta")]

        # Set up linear solver
        linear_solver = CompressibleSolver(eqns)

        # build time stepper
        stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                          transport_methods,
                                          linear_solver=linear_solver)

    elif stepper_type == 'multi_level':
        scheme = AdamsBashforth(domain, order=2)
        stepper = Timestepper(eqns, scheme, io, spatial_methods=transport_methods)

    else:
        raise ValueError(f'stepper_type {stepper_type} not recognised')

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


@pytest.mark.parametrize("stepper_type", ["multi_level", "semi_implicit"])
@pytest.mark.parametrize("checkpoint_method", ["dumbcheckpoint", "checkpointfile"])
def test_checkpointing(tmpdir, stepper_type, checkpoint_method):

    mesh_name = 'checkpointing_mesh'

    # Set up mesh
    nlayers = 5   # horizontal layers
    columns = 15  # number of columns
    L = 3.e5
    m = PeriodicIntervalMesh(columns, L)
    dt = 0.2

    # build volume mesh
    H = 1.0e4  # Height position of the model top
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers, name=mesh_name)

    dirname_1 = str(tmpdir)+'/checkpointing_1'
    dirname_2 = str(tmpdir)+'/checkpointing_2'
    dirname_3 = str(tmpdir)+'/checkpointing_3'

    output_1 = OutputParameters(
        dirname=dirname_1,
        dumpfreq=1,
        checkpoint=True,
        checkpoint_method=checkpoint_method,
        chkptfreq=4,
    )
    output_2 = OutputParameters(
        dirname=dirname_2,
        dumpfreq=1,
        checkpoint=True,
        checkpoint_method=checkpoint_method,
        chkptfreq=2,
    )

    stepper_1, eqns_1 = set_up_model_objects(mesh, dt, output_1, stepper_type)
    stepper_2, eqns_2 = set_up_model_objects(mesh, dt, output_2, stepper_type)

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

    chkpt_filename = 'chkpt' if checkpoint_method == 'dumbcheckpoint' else 'chkpt.h5'
    chkpt_2_path = path.join(stepper_2.io.dumpdir, chkpt_filename)
    output_3 = OutputParameters(
        dirname=dirname_3,
        dumpfreq=1,
        chkptfreq=2,
        checkpoint=True,
        checkpoint_method=checkpoint_method,
        checkpoint_pickup_filename=chkpt_2_path,
    )

    if checkpoint_method == 'checkpointfile':
        mesh = pick_up_mesh(output_3, mesh_name)
    stepper_3, _ = set_up_model_objects(mesh, dt, output_3, stepper_type)
    stepper_3.io.pick_up_from_checkpoint(stepper_3.fields)

    # ------------------------------------------------------------------------ #
    # Check that checkpointed values are picked up to almost machine precision
    # ------------------------------------------------------------------------ #

    # With old checkpointing this worked exactly
    # With new checkpointing this creates some small error
    for field_name in ['rho', 'theta', 'u']:
        diff_array = stepper_2.fields(field_name).dat.data - stepper_3.fields(field_name).dat.data
        error = np.linalg.norm(diff_array) / np.linalg.norm(stepper_2.fields(field_name).dat.data)
        assert error < 5e-16, \
            f'Checkpointed and picked up field {field_name} is not equal'

    # ------------------------------------------------------------------------ #
    # Pick up from checkpoint and run *same* timestepper for 2 more time steps
    # ------------------------------------------------------------------------ #

    # Wipe fields from second time stepper
    if checkpoint_method == 'dumbcheckpoint':
        # Get an error when picking up fields with the same stepper with new method
        initialise_fields(eqns_2, stepper_2)
        stepper_2.run(t=2*dt, tmax=4*dt, pick_up=True)

    # ------------------------------------------------------------------------ #
    # Run *new* timestepper for 2 time steps
    # ------------------------------------------------------------------------ #

    output_3 = OutputParameters(
        dirname=dirname_3,
        dumpfreq=1,
        chkptfreq=2,
        checkpoint=True,
        checkpoint_method=checkpoint_method,
        checkpoint_pickup_filename=chkpt_2_path
    )
    if checkpoint_method == 'checkpointfile':
        mesh = pick_up_mesh(output_3, mesh_name)
    stepper_3, _ = set_up_model_objects(mesh, dt, output_3, stepper_type)
    stepper_3.run(t=2*dt, tmax=4*dt, pick_up=True)

    # ------------------------------------------------------------------------ #
    # Compare fields against saved values for run without checkpointing
    # ------------------------------------------------------------------------ #

    # Rather than use Firedrake norm routine, take numpy norm of data arrays.
    # This is because Firedrake may see the fields from the different time
    # steppers as being on different meshes

    for field_name in ['rho', 'theta', 'u', 'rho_error', 'theta_perturbation']:
        if checkpoint_method == 'dumbcheckpoint':
            # Check final fields are the same when checkpointing with the same time
            # stepper -- very tight tolerance as there should be no error
            diff_array = stepper_1.fields(field_name).dat.data - stepper_2.fields(field_name).dat.data
            error = np.linalg.norm(diff_array) / np.linalg.norm(stepper_1.fields(field_name).dat.data)
            assert error < 1e-15, \
                f'Checkpointed field {field_name} with same time stepper is not equal to non-checkpointed field'

        # Check final fields when picking up with a new time stepper. As the
        # solver objects are different, certain cached information will no
        # longer be available so expect a bigger error
        diff_array = stepper_1.fields(field_name).dat.data - stepper_3.fields(field_name).dat.data
        error = np.linalg.norm(diff_array) / np.linalg.norm(stepper_1.fields(field_name).dat.data)
        assert error < 1e-8, \
            f'Checkpointed field {field_name} with new time stepper is not equal to non-checkpointed field'
