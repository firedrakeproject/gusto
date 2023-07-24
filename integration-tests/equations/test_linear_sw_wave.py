"""
This runs a wave scenario for the linear f-plane shallow water equations and
checks the result against a known checkpointed answer.
"""

from os.path import join, abspath, dirname
from gusto import *
from firedrake import (PeriodicSquareMesh, SpatialCoordinate, Constant, sin,
                       cos, pi)
import numpy as np


def run_linear_sw_wave(tmpdir):
    # Paramerers
    dt = 0.001
    tmax = 30*dt
    H = 1
    wx = 2
    wy = 1
    g = 1

    # Domain
    mesh_name = 'linear_sw_mesh'
    L = 1
    nx = ny = 20
    mesh = PeriodicSquareMesh(nx, ny, L, direction='both', name=mesh_name)
    x, y = SpatialCoordinate(mesh)
    domain = Domain(mesh, dt, 'BDM', 1)

    # Equation
    parameters = ShallowWaterParameters(H=H, g=g)
    fexpr = Constant(1)
    eqns = LinearShallowWaterEquations(domain, parameters, fexpr=fexpr)

    # I/O
    output = OutputParameters(dirname=str(tmpdir)+"/linear_sw_wave",
                              dumpfreq=1,
                              log_level='INFO')
    io = IO(domain, output)
    transport_methods = [DefaultTransport(eqns, "D")]

    # Timestepper
    stepper = Timestepper(eqns, RK4(domain), io, transport_methods)

    # ---------------------------------------------------------------------- #
    # Initial conditions
    # ---------------------------------------------------------------------- #

    eta = sin(2*pi*(x-L/2)*wx)*cos(2*pi*(y-L/2)*wy) - (1/5)*cos(2*pi*(x-L/2)*wx)*sin(4*pi*(y-L/2)*wy)
    Dexpr = H + eta

    u = cos(4*pi*(x-L/2)*wx)*cos(2*pi*(y-L/2)*wy)
    v = cos(2*pi*(x-L/2)*wx)*cos(4*pi*(y-L/2)*wy)
    uexpr = as_vector([u, v])

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")

    u0.project(uexpr)
    D0.interpolate(Dexpr)

    # --------------------------------------------------------------------- #
    # Run
    # --------------------------------------------------------------------- #

    stepper.run(t=0, tmax=tmax)

    # State for checking checkpoints
    checkpoint_name = 'linear_sw_wave_chkpt.h5'
    new_path = join(abspath(dirname(__file__)), '..', f'data/{checkpoint_name}')
    check_output = OutputParameters(dirname=tmpdir+"/linear_sw_wave",
                                    checkpoint_pickup_filename=new_path)
    check_mesh = pick_up_mesh(check_output, mesh_name)
    check_domain = Domain(check_mesh, dt, 'BDM', 1)
    check_eqn = ShallowWaterEquations(check_domain, parameters, fexpr=fexpr)
    check_io = IO(check_domain, output=check_output)
    check_stepper = Timestepper(check_eqn, RK4(check_domain), check_io)
    check_stepper.io.pick_up_from_checkpoint(check_stepper.fields)

    return stepper, check_stepper


def test_linear_sw_wave(tmpdir):

    dirname = str(tmpdir)
    stepper, check_stepper = run_linear_sw_wave(dirname)

    for variable in ['u', 'D']:
        new_variable = stepper.fields(variable)
        check_variable = check_stepper.fields(variable)
        diff_array = new_variable.dat.data - check_variable.dat.data
        error = np.linalg.norm(diff_array) / np.linalg.norm(check_variable.dat.data)

        # Slack values chosen to be robust to different platforms
        assert error < 1e-10, f'Values for {variable} in ' + \
            'linear shallow water wave test do not match KGO values'
