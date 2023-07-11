"""
This runs a dry compressible example with a perturbation in a vertical slice,
and checks the example against a known good checkpointed answer.
"""

from os.path import join, abspath, dirname
from gusto import *
from gusto import thermodynamics as tde
from firedrake import (SpatialCoordinate, PeriodicIntervalMesh, exp,
                       sqrt, ExtrudedMesh, as_vector)
import numpy as np


def run_dry_compressible(tmpdir):

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
    mesh_name = 'dry_compressible_mesh'
    m = PeriodicIntervalMesh(ncols, Lx)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=Lz/nlayers, name=mesh_name)
    domain = Domain(mesh, dt, "CG", 1)

    # Equation
    parameters = CompressibleParameters()
    eqn = CompressibleEulerEquations(domain, parameters)

    # I/O
    output = OutputParameters(dirname=tmpdir+"/dry_compressible",
                              dumpfreq=2, chkptfreq=2)
    io = IO(domain, output)

    # Transport schemes
    transported_fields = [ImplicitMidpoint(domain, "u"),
                          SSPRK3(domain, "rho"),
                          SSPRK3(domain, "theta")]
    transport_methods = [DGUpwind(eqn, 'u'),
                         DGUpwind(eqn, 'rho'),
                         DGUpwind(eqn, 'theta')]

    # Linear solver
    linear_solver = CompressibleSolver(eqn)

    # Time stepper
    stepper = SemiImplicitQuasiNewton(eqn, io, transported_fields,
                                      transport_methods,
                                      linear_solver=linear_solver)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    R_d = parameters.R_d
    g = parameters.g

    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")
    u0 = stepper.fields("u")

    # Approximate hydrostatic balance
    x, z = SpatialCoordinate(mesh)
    T = Constant(300.0)
    zH = R_d * T / g
    p = Constant(100000.0) * exp(-z / zH)
    theta0.interpolate(tde.theta(parameters, T, p))
    rho0.interpolate(p / (R_d * T))

    # Add horizontal translation to ensure some transport happens
    u0.project(as_vector([0.5, 0.0]))

    stepper.set_reference_profiles([('rho', rho0), ('theta', theta0)])

    # Add perturbation
    r = sqrt((x-Lx/2)**2 + (z-Lz/2)**2)
    theta_pert = 1.0*exp(-(r/(Lx/5))**2)
    theta0.interpolate(theta0 + theta_pert)

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=tmax)

    # IO for checking checkpoints
    checkpoint_name = 'dry_compressible_chkpt.h5'
    new_path = join(abspath(dirname(__file__)), '..', f'data/{checkpoint_name}')
    check_output = OutputParameters(dirname=tmpdir+"/dry_compressible",
                                    checkpoint_pickup_filename=new_path)
    check_mesh = pick_up_mesh(check_output, mesh_name)
    check_domain = Domain(check_mesh, dt, "CG", 1)
    check_eqn = CompressibleEulerEquations(check_domain, parameters)
    check_io = IO(check_domain, check_output)
    check_stepper = SemiImplicitQuasiNewton(check_eqn, check_io, [], [])
    check_stepper.io.pick_up_from_checkpoint(check_stepper.fields)

    return stepper, check_stepper


def test_dry_compressible(tmpdir):

    dirname = str(tmpdir)
    stepper, check_stepper = run_dry_compressible(dirname)

    for variable in ['u', 'rho', 'theta']:
        new_variable = stepper.fields(variable)
        check_variable = check_stepper.fields(variable)
        diff_array = new_variable.dat.data - check_variable.dat.data
        error = np.linalg.norm(diff_array) / np.linalg.norm(check_variable.dat.data)

        # Slack values chosen to be robust to different platforms
        assert error < 1e-10, f'Values for {variable} in ' + \
            'Dry Compressible test do not match KGO values'
