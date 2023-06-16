"""
This runs an incompressible example with a perturbation in a hydrostatic
atmosphere, and checks the example against a known good checkpointed answer.
"""

from os.path import join, abspath, dirname
from gusto import *
from firedrake import (SpatialCoordinate, PeriodicIntervalMesh, exp,
                       sqrt, ExtrudedMesh, Function, as_vector)
import numpy as np


def run_incompressible(tmpdir):

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    dt = 6.0
    tmax = 2*dt
    nlayers = 10  # horizontal layers
    ncols = 10  # number of columns
    Lx = 1000.0
    Lz = 1000.0
    mesh_name = 'incompressible_mesh'
    m = PeriodicIntervalMesh(ncols, Lx)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=Lz/nlayers, name=mesh_name)
    domain = Domain(mesh, dt, "CG", 1)

    # Equation
    parameters = CompressibleParameters()
    eqn = IncompressibleBoussinesqEquations(domain, parameters)

    # I/O
    output = OutputParameters(dirname=tmpdir+"/incompressible",
                              dumpfreq=2, chkptfreq=2)
    io = IO(domain, output)

    # Transport Schemes
    b_opts = SUPGOptions()
    transported_fields = [ImplicitMidpoint(domain, "u"),
                          SSPRK3(domain, "b", options=b_opts)]

    # Linear solver
    linear_solver = IncompressibleSolver(eqn)

    # Time stepper
    stepper = SemiImplicitQuasiNewton(eqn, io, transported_fields,
                                      linear_solver=linear_solver)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    p0 = stepper.fields("p")
    b0 = stepper.fields("b")
    u0 = stepper.fields("u")

    # Add horizontal translation to ensure some transport happens
    u0.project(as_vector([0.5, 0.0]))

    # z.grad(bref) = N**2
    x, z = SpatialCoordinate(mesh)
    N = parameters.N
    bref = z*(N**2)

    b_b = Function(b0.function_space()).interpolate(bref)
    incompressible_hydrostatic_balance(eqn, b_b, p0)
    stepper.set_reference_profiles([('p', p0), ('b', b_b)])

    # Add perturbation
    r = sqrt((x-Lx/2)**2 + (z-Lz/2)**2)
    b_pert = 0.1*exp(-(r/(Lx/5)**2))
    b0.interpolate(b_b + b_pert)

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=tmax)

    # State for checking checkpoints
    checkpoint_name = 'incompressible_chkpt.h5'
    new_path = join(abspath(dirname(__file__)), '..', f'data/{checkpoint_name}')
    check_output = OutputParameters(dirname=tmpdir+"/incompressible",
                                    checkpoint_pickup_filename=new_path)
    check_mesh = pick_up_mesh(check_output, mesh_name)
    check_domain = Domain(check_mesh, dt, "CG", 1)
    check_eqn = IncompressibleBoussinesqEquations(check_domain, parameters)
    check_io = IO(check_domain, check_output)
    check_stepper = SemiImplicitQuasiNewton(check_eqn, check_io, [])
    check_stepper.io.pick_up_from_checkpoint(check_stepper.fields)

    return stepper, check_stepper


def test_incompressible(tmpdir):

    dirname = str(tmpdir)
    stepper, check_stepper = run_incompressible(dirname)

    for variable in ['u', 'b', 'p']:
        new_variable = stepper.fields(variable)
        check_variable = check_stepper.fields(variable)
        diff_array = new_variable.dat.data - check_variable.dat.data
        error = np.linalg.norm(diff_array) / np.linalg.norm(check_variable.dat.data)

        # Slack values chosen to be robust to different platforms
        assert error < 1e-10, f'Values for {variable} in ' + \
            'Incompressible test do not match KGO values'
