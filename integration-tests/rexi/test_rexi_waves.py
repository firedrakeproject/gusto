"""
This runs the wave scenario from Schreiber, Peixoto, Haut and Wingate (2018)
for the linear f-plane shallow water equations using REXI. It checks the output
against a known checkpointed answer.
"""

from os.path import join, abspath, dirname
from gusto import *
from firedrake import (PeriodicSquareMesh, SpatialCoordinate, Constant, sin,
                       cos, pi, as_vector)

import numpy as np


def run_linear_sw_rexi_wave(tmpdir):
    # Parameters
    dt = 0.001
    tmax = 0.1
    H = 1
    wx = 2
    wy = 1
    g = 1

    # Domain
    mesh_name = 'linear_sw_mesh'
    L = 1
    nx = ny = 16
    mesh = PeriodicSquareMesh(nx, ny, L, direction='both', name=mesh_name)
    x, y = SpatialCoordinate(mesh)
    domain = Domain(mesh, dt, 'BDM', 1)

    # Equation
    parameters = ShallowWaterParameters(H=H, g=g)
    fexpr = Constant(1)
    eqns = LinearShallowWaterEquations(domain, parameters, fexpr=fexpr)

    # I/O
    output = OutputParameters(
        dirname=str(tmpdir)+"/linear_sw_rexi_wave",
        dumpfreq=1,
    )
    io = IO(domain, output)

    # ---------------------------------------------------------------------- #
    # Initial conditions
    # ---------------------------------------------------------------------- #

    eta = sin(2*pi*(x-L/2)*wx)*cos(2*pi*(y-L/2)*wy) - (1/5)*cos(2*pi*(x-L/2)*wx)*sin(4*pi*(y-L/2)*wy)
    Dexpr = H + eta

    u = cos(4*pi*(x-L/2)*wx)*cos(2*pi*(y-L/2)*wy)
    v = cos(2*pi*(x-L/2)*wx)*cos(4*pi*(y-L/2)*wy)
    uexpr = as_vector([u, v])

    U_in = Function(eqns.function_space, name="U_in")
    Uexpl = Function(eqns.function_space, name="Uexpl")
    u, D = U_in.split()
    u.project(uexpr)
    D.interpolate(Dexpr)

    # --------------------------------------------------------------------- #
    # Run
    # --------------------------------------------------------------------- #

    rexi_params = RexiParameters()
    rexi = Rexi(eqns, rexi_params)
    rexi.solve(Uexpl, U_in, dt)

    uexpl, Dexpl = Uexpl.split()

    # --------------------------------------------------------------------- #
    # Checkpointing
    # --------------------------------------------------------------------- #

    # State for checking checkpoints
    checkpoint_name = 'linear_sw_wave_rexi_chkpt.h5'
    new_path = join(abspath(dirname(__file__)), '..', f'data/{checkpoint_name}')
    check_output = OutputParameters(dirname=tmpdir+"/linear_sw_rexi_wave",
                                    checkpoint_pickup_filename=new_path)
    check_mesh = pick_up_mesh(check_output, mesh_name)
    check_domain = Domain(check_mesh, dt, 'BDM', 1)
    check_eqn = ShallowWaterEquations(check_domain, parameters, fexpr=fexpr)
    check_io = IO(check_domain, output=check_output)
    check_stepper = Timestepper(check_eqn, RK4(check_domain), check_io)
    check_stepper.io.pick_up_from_checkpoint(check_stepper.fields)
    check_u = check_stepper.fields("u")
    check_D = check_stepper.fields("D")

    return uexpl, Dexpl, check_u, check_D


def test_linear_sw_rexi_wave(tmpdir):

    dirname = str(tmpdir)
    uexpl, Dexpl, check_u, check_D = run_linear_sw_rexi_wave(dirname)

    diff_array_u = uexpl.dat.data - check_u.dat.data
    diff_array_D = Dexpl.dat.data - check_D.dat.data
    u_error = np.linalg.norm(diff_array_u) / np.linalg.norm(check_u.dat.data)
    D_error = np.linalg.norm(diff_array_D) / np.linalg.norm(check_D.dat.data)

    # Slack values chosen to be robust to different platforms
    assert u_error < 1e-10, 'u values in REXI linear shallow water wave test do not match KGO values'
    assert D_error < 1e-10, 'D values in REXI linear shallow water wave test do not match KGO values'
