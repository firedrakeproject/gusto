"""
This runs a Boussinesq example with a perturbation in a hydrostatic
atmosphere, and checks the example against a known good checkpointed answer
for both the compressible and incompressible forms of the equations.
"""

from os.path import join, abspath, dirname
from gusto import *
from firedrake import (SpatialCoordinate, PeriodicIntervalMesh, exp,
                       sqrt, ExtrudedMesh, Function, as_vector)
import numpy as np
import pytest


def run_boussinesq(tmpdir, compressible):

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
    mesh_name = 'boussinesq_mesh'
    m = PeriodicIntervalMesh(ncols, Lx)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=Lz/nlayers, name=mesh_name)
    domain = Domain(mesh, dt, "CG", 1)

    # Equation
    parameters = BoussinesqParameters()
    eqn = BoussinesqEquations(domain, parameters, compressible=compressible)

    # I/O
    if compressible:
        output_dirname = tmpdir+"/boussinesq_compressible"
    else:
        output_dirname = tmpdir+"/boussinesq_incompressible"
    output = OutputParameters(dirname=output_dirname,
                              dumpfreq=2, chkptfreq=2, checkpoint=True)
    io = IO(domain, output)

    # Transport Schemes
    b_opts = SUPGOptions()
    if compressible:
        transported_fields = [TrapeziumRule(domain, "u"),
                              SSPRK3(domain, "p"),
                              SSPRK3(domain, "b", options=b_opts)]
        transport_methods = [DGUpwind(eqn, "u"),
                             DGUpwind(eqn, "p"),
                             DGUpwind(eqn, "b", ibp=b_opts.ibp)]
    else:
        transported_fields = [TrapeziumRule(domain, "u"),
                              SSPRK3(domain, "b", options=b_opts)]
        transport_methods = [DGUpwind(eqn, "u"),
                             DGUpwind(eqn, "b", ibp=b_opts.ibp)]

    # Linear solver
    linear_solver = BoussinesqSolver(eqn)

    # Time stepper
    stepper = SemiImplicitQuasiNewton(eqn, io, transported_fields,
                                      transport_methods,
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
    boussinesq_hydrostatic_balance(eqn, b_b, p0)
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
    if compressible:
        checkpoint_name = 'compressible_boussinesq_chkpt.h5'
    else:
        checkpoint_name = 'incompressible_boussinesq_chkpt.h5'
    new_path = join(abspath(dirname(__file__)), '..', f'data/{checkpoint_name}')
    check_output = OutputParameters(dirname=output_dirname,
                                    checkpoint_pickup_filename=new_path,
                                    checkpoint=True)
    check_mesh = pick_up_mesh(check_output, mesh_name)
    check_domain = Domain(check_mesh, dt, "CG", 1)
    check_eqn = BoussinesqEquations(check_domain, parameters, compressible)
    check_io = IO(check_domain, check_output)
    check_stepper = SemiImplicitQuasiNewton(check_eqn, check_io, [], [])
    check_stepper.io.pick_up_from_checkpoint(check_stepper.fields)

    return stepper, check_stepper


@pytest.mark.parametrize("compressible", [True, False])
def test_boussinesq(tmpdir, compressible):

    dirname = str(tmpdir)
    stepper, check_stepper = run_boussinesq(dirname, compressible)

    for variable in ['u', 'b', 'p']:
        new_variable = stepper.fields(variable)
        check_variable = check_stepper.fields(variable)
        diff_array = new_variable.dat.data - check_variable.dat.data
        error = np.linalg.norm(diff_array) / np.linalg.norm(check_variable.dat.data)

        # Slack values chosen to be robust to different platforms
        assert error < 1e-10, f'Values for {variable} in ' + \
            'Incompressible test do not match KGO values'
