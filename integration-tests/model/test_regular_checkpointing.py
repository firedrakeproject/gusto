"""
Runs a shallow water test that uses checkpointing. The test runs for 8
timesteps and outputs a checkpoint file every 2 timesteps. The checkpoint
after 4 steps is picked up and run for another 4 steps and the results are
compared to the original run of 8 timesteps.
"""

from os import path
import numpy as np
from gusto import *
from firedrake import (PeriodicSquareMesh, SpatialCoordinate,
                       Function, cos, pi, as_vector, sin,
                       CheckpointFile)


def run_sw_fplane(run_num, ndt, output, chkfile=None):
    # Domain
    if run_num == 1:
        # Set up a mesh
        Nx = 32
        Ny = Nx
        Lx = 10
        mesh = PeriodicSquareMesh(Nx, Ny, Lx, quadrilateral=True)

    else:
        # On this run we are picking up a checkpoint from a previous run and
        # we recover the mesh from the checkpoint
        mesh = CheckpointFile(chkfile, 'r').load_mesh()

    dt = 0.01
    domain = Domain(mesh, dt, 'RTCF', 1)

    # Equation
    H = 2
    g = 50
    parameters = ShallowWaterParameters(mesh, H=H, g=g)
    f0 = 10
    fexpr = Constant(f0)
    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr)

    # I/O
    output = output

    io = IO(domain, output, diagnostic_fields=[CourantNumber()])

    # Transport schemes
    vorticity_transport = VorticityTransport(domain, eqns, supg=True)
    transported_fields = [
        TrapeziumRule(domain, "u", augmentation=vorticity_transport),
        SSPRK3(domain, "D")
    ]
    transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]

    # Time stepper
    stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                      transport_methods,
                                      num_outer=4, num_inner=1)

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")

    if run_num == 1:
        # Initialise fields with the initial condition
        u_expr, D_expr = initialise(mesh, parameters, Lx, f0)
        u0.project(u_expr)
        D0.interpolate(D_expr)

    else:
        # Initialise fields from previous run's checkpoint
        with CheckpointFile(chkfile, 'r') as chk:
            start_D = chk.load_function(mesh, 'D')
            start_u = chk.load_function(mesh, 'u')

        u0.project(start_u)
        D0.interpolate(start_D)

    Dbar = Function(D0.function_space()).assign(H)
    stepper.set_reference_profiles([('D', Dbar)])

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=dt*ndt)

    return stepper


def initialise(mesh, parameters, Lx, f0):

    x, y = SpatialCoordinate(mesh)
    N0 = 0.1
    g = parameters.g
    H = parameters.H
    Lx = Lx
    gamma = sqrt(g*H)
    ###############################
    #  Fast wave:
    k1 = 5*(2*pi/Lx)

    K1sq = k1**2
    psi1 = sqrt(f0**2 + g*H*K1sq)
    xi1 = sqrt(2*K1sq)*psi1

    c1 = cos(k1*x)
    s1 = sin(k1*x)
    ################################
    #  Slow wave:
    k2 = -k1
    l2 = k1

    K2sq = k2**2 + l2**2
    psi2 = sqrt(f0**2 + g*H*K2sq)

    c2 = cos(k2*x + l2*y)
    s2 = sin(k2*x + l2*y)
    ################################
    #  Construct the initial condition:
    A1 = N0/xi1
    u1 = A1*(k1*psi1*c1)
    v1 = A1*(f0*k1*s1)
    phi1 = A1*(K1sq*gamma*c1)

    A2 = N0/psi2
    u2 = A2*(l2*gamma*s2)
    v2 = A2*(-k2*gamma*s2)
    phi2 = A2*(f0*c2)

    u_expr = as_vector([u1+u2, v1+v2])
    D_expr = H + sqrt(H/g)*(phi1+phi2)

    return u_expr, D_expr


def test_regular_checkpointing(tmpdir):

    output1 = OutputParameters(
        dirname=str(tmpdir)+"/sw_fplane_run1",
        dumpfreq=1,
        chkptfreq=2,
        checkpoint=True,
        multichkpt=True
    )
    stepper1 = run_sw_fplane(run_num=1, ndt=8, output=output1)

    # Pick up the checkpoint after 4 timesteps and run with that as the IC
    chkpt1_filename = 'chkpt4.h5'
    chkpt1_path = path.join(stepper1.io.dumpdir, 'chkpts', chkpt1_filename)
    # First check that the checkpoint was sucessfully created - the test will fail if
    # this is not the case
    assert path.isfile(chkpt1_path), "The checkpoint from the first run was not created"

    output2 = OutputParameters(
        dirname=str(tmpdir)+"/sw_fplane_run2",
        dumpfreq=1,
        checkpoint=True,
        checkpoint_pickup_filename=chkpt1_path
    )
    stepper2 = run_sw_fplane(run_num=2, ndt=4, output=output2, chkfile=chkpt1_path)

    # ------------------------------------------------------------------------ #
    # Check that checkpointed values agree
    # ------------------------------------------------------------------------ #

    for field_name in ['u', 'D']:
        run1_variable = stepper1.fields(field_name)
        run2_variable = stepper2.fields(field_name)
        diff_array = run2_variable.dat.data - run1_variable.dat.data
        error = np.linalg.norm(diff_array) / np.linalg.norm(run1_variable.dat.data)

        assert error < 1e-10, f'Values for {field_name} in ' + \
            'shallow water fplane test do not match after ' + \
            'picking up checkpoint halfway through'
