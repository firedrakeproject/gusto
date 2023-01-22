"""
This runs a dry compressible example with a perturbation in a vertical slice,
and checks the example against a known good checkpointed answer.
"""

from os.path import join, abspath, dirname
from gusto import *
from gusto import thermodynamics as tde
from firedrake import (SpatialCoordinate, PeriodicIntervalMesh, exp,
                       sqrt, ExtrudedMesh, norm, as_vector)


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
    m = PeriodicIntervalMesh(ncols, Lx)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=Lz/nlayers)
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

    # Linear solver
    linear_solver = CompressibleSolver(eqn)

    # Time stepper
    stepper = SemiImplicitQuasiNewton(eqn, io, transported_fields,
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

    # State for checking checkpoints
    checkpoint_name = 'dry_compressible_chkpt.h5'
    new_path = join(abspath(dirname(__file__)), '..', f'data/{checkpoint_name}')
    check_eqn = CompressibleEulerEquations(domain, parameters)
    check_output = OutputParameters(dirname=tmpdir+"/dry_compressible",
                                    checkpoint_pickup_filename=new_path)
    check_io = IO(domain, check_output)
    check_stepper = SemiImplicitQuasiNewton(check_eqn, check_io, [])
    check_stepper.set_reference_profiles([])
    check_stepper.run(t=0, tmax=0, pickup=True)

    return stepper, check_stepper


def test_dry_compressible(tmpdir):

    dirname = str(tmpdir)
    stepper, check_stepper = run_dry_compressible(dirname)

    for variable in ['u', 'rho', 'theta']:
        new_variable = stepper.fields(variable)
        check_variable = check_stepper.fields(variable)
        error = norm(new_variable - check_variable) / norm(check_variable)

        # Slack values chosen to be robust to different platforms
        assert error < 1e-10, f'Values for {variable} in ' + \
            'Dry Compressible test do not match KGO values'
