"""
This runs the wave scenario from Schreiber et al 2018 for the linear f-plane
shallow water equations and compares the REXI output to a known checkpointed
answer to confirm that REXI is correct.
"""

from os.path import join, abspath, dirname
from gusto import *
from gusto.rexi import *
from firedrake import (PeriodicUnitSquareMesh, SpatialCoordinate, Constant, sin,
                       cos, pi, as_vector, Function, File)

import numpy as np


def run_rexi_sw(tmpdir):
    # Parameters
    dt = 0.001
    tmax = 0.1
    H = 1
    f = 1
    g = 1

    # Domain
    mesh_name = 'linear_sw_mesh'
    L = 1
    n = 20
    mesh = PeriodicUnitSquareMesh(n, n, name=mesh_name)
    domain = Domain(mesh, dt, 'BDM', 1)

    # Equation
    parameters = ShallowWaterParameters(H=H, g=g)
    fexpr = Constant(f)
    eqns = LinearShallowWaterEquations(domain, parameters, fexpr=fexpr)

    # REXI output
    rexi_output = File(str(tmpdir)+"/waves_sw/rexi.pvd")

    # Initial conditions
    x, y = SpatialCoordinate(mesh)
    uexpr = as_vector([cos(8*pi*(x-L/2))*cos(2*pi*(y-L/2)), cos(4*pi*(x-L/2))*cos(4*pi*(y-L/2))])
    Dexpr = H + sin(4*pi*(x-L/2))*cos(2*pi*(y-L/2)) - 0.2*cos(4*pi*(x-L/2))*sin(4*pi*(y-L/2))

    U_in = Function(eqns.function_space, name="U_in")
    Uexpl = Function(eqns.function_space, name="Uexpl")
    u, D = U_in.split()
    u.project(uexpr)
    D.interpolate(Dexpr)
    rexi_output.write(u, D)

    # Compute exponential solution and write it out
    rexi = Rexi(eqns, RexiParameters())
    rexi.solve(Uexpl, U_in, tmax)

    uexpl, Dexpl = Uexpl.split()
    u.assign(uexpl)
    D.assign(Dexpl)
    rexi_output.write(u, D)

    # Checkpointing
    checkpoint_name = 'linear_sw_wave_rexi_chkpt.h5'
    new_path = join(abspath(dirname(__file__)), '..', f'data/{checkpoint_name}')
    check_output = OutputParameters(dirname=tmpdir+"/linear_sw_wave",
                                    checkpoint_pickup_filename=new_path)
    check_mesh = pick_up_mesh(check_output, mesh_name)
    check_domain = Domain(check_mesh, dt, 'BDM', 1)
    check_eqn = ShallowWaterEquations(check_domain, parameters, fexpr=fexpr)
    check_io = IO(check_domain, output=check_output)
    check_stepper = Timestepper(check_eqn, RK4(check_domain), check_io)
    check_stepper.io.pick_up_from_checkpoint(check_stepper.fields)
    usoln = check_stepper.fields("u")
    Dsoln = check_stepper.fields("D")

    return usoln, Dsoln, uexpl, Dexpl


def test_rexi_sw(tmpdir):

    dirname = str(tmpdir)

    usoln, Dsoln, uexpl, Dexpl = run_rexi_sw(dirname)

    udiff_arr = uexpl.dat.data - usoln.dat.data
    Ddiff_arr = Dexpl.dat.data - Dsoln.dat.data

    uerror = np.linalg.norm(udiff_arr) / np.linalg.norm(usoln.dat.data)
    Derror = np.linalg.norm(Ddiff_arr) / np.linalg.norm(Dsoln.dat.data)

    assert uerror < 0.04, 'u values in REXI linear shallow water wave test do not match KGO values'
    assert Derror < 0.02, 'D values in REXI linear shallow water wave test do not match KGO values'
