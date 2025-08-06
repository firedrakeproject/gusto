"""
This runs the wave scenario from Schreiber et al 2018 for the linear f-plane
shallow water equations and compares the REXI output to a known checkpointed
answer to confirm that REXI is correct.
"""

from os.path import join, abspath, dirname
from gusto import *
from gusto.rexi import *
from firedrake import (PeriodicUnitSquareMesh, SpatialCoordinate, Constant, sin,
                       cos, pi, as_vector, Function, COMM_WORLD, Ensemble)
from firedrake.output import VTKFile

import numpy as np
import pytest


def run_rexi_sw(tmpdir, coefficients, approx_type, ensemble=None):
    # Parameters
    tmax = 0.1
    H = 1
    f = 1
    g = 1

    # Domain
    mesh_name = 'linear_sw_mesh'
    L = 1
    n = 20
    write_output = True
    if ensemble is not None:
        comm = ensemble.comm
        mesh = PeriodicUnitSquareMesh(n, n, name=mesh_name, comm=comm)
        # REXI output
        rexi_output = VTKFile(
            str(tmpdir)+"/waves_sw/rexi.pvd",
            comm=ensemble.comm
        )

    else:
        comm = COMM_WORLD
        mesh = PeriodicUnitSquareMesh(n, n, name=mesh_name)
        # REXI output
        rexi_output = VTKFile(str(tmpdir)+"/waves_sw/rexi.pvd")

    domain = Domain(mesh, tmax, 'BDM', 1)

    # Equation
    parameters = ShallowWaterParameters(mesh, H=H, g=g)
    fexpr = Constant(f)
    eqns = LinearShallowWaterEquations(domain, parameters, fexpr=fexpr)

    # Initial conditions
    x, y = SpatialCoordinate(mesh)
    uexpr = as_vector([cos(8*pi*(x-L/2))*cos(2*pi*(y-L/2)), cos(4*pi*(x-L/2))*cos(4*pi*(y-L/2))])
    Dexpr = H + sin(4*pi*(x-L/2))*cos(2*pi*(y-L/2)) - 0.2*cos(4*pi*(x-L/2))*sin(4*pi*(y-L/2))

    U_in = Function(eqns.function_space, name="U_in")
    Uexpl = Function(eqns.function_space, name="Uexpl")
    u, D = U_in.subfunctions
    Dbar = eqns.X_ref.subfunctions[1]

    u.project(uexpr)
    D.interpolate(Dexpr)
    Dbar.interpolate(H)
    if write_output:
        rexi_output.write(u, D)

    # Compute exponential solution and write it out
    if approx_type == "REXI":
        rexi = Rexi(eqns, RexiParameters(coefficients=coefficients),
                    manager=ensemble)
    elif approx_type == "REXII":
        rexi = Rexii(eqns, RexiParameters(coefficients=coefficients),
                    manager=ensemble)
    else:
        raise ValueError("approx type must be REXI or REXII")

    rexi.solve(Uexpl, U_in, tmax)

    uexpl, Dexpl = Uexpl.subfunctions
    u.assign(uexpl)
    D.assign(Dexpl)

    if write_output:
        rexi_output.write(u, D)

    # Checkpointing
    if write_output:
        checkpoint_name = 'linear_sw_wave_rexi_chkpt.h5'
        new_path = join(abspath(dirname(__file__)), '..', f'data/{checkpoint_name}')
        check_output = OutputParameters(dirname=tmpdir+"/linear_sw_wave",
                                        checkpoint_pickup_filename=new_path,
                                        checkpoint=True)
        check_mesh = pick_up_mesh(check_output, mesh_name, comm=comm)
        check_domain = Domain(check_mesh, tmax, 'BDM', 1)
        check_eqn = ShallowWaterEquations(check_domain, parameters, fexpr=fexpr)
        check_io = IO(check_domain, output=check_output)
        check_stepper = Timestepper(check_eqn, RK4(check_domain), check_io)
        check_stepper.io.pick_up_from_checkpoint(check_stepper.fields, comm=comm)
        usoln = check_stepper.fields("u")
        Dsoln = check_stepper.fields("D")

        udiff_arr = uexpl.dat.data - usoln.dat.data
        Ddiff_arr = Dexpl.dat.data - Dsoln.dat.data

        uerror = np.linalg.norm(udiff_arr) / np.linalg.norm(usoln.dat.data)
        Derror = np.linalg.norm(Ddiff_arr) / np.linalg.norm(Dsoln.dat.data)

    return uerror, Derror


@pytest.mark.parametrize("algorithm", ["REXI_Haut", "REXI_Caliari", "REXII_Caliari"])
def test_rexi_sw(tmpdir, algorithm):

    match algorithm:
        case "REXI_Haut":
            coefficients="Haut"
            approx_type = "REXI"
        case "REXI_Caliari":
            coefficients="Caliari"
            approx_type = "REXI"
        case "REXII_Caliari":
            coefficients="Caliari"
            approx_type = "REXII"
        case _:
            raise ValueError("Algorithm must be one of: REXI_Haut, REXI_Caliari or REXII_Caliari.")

    dirname = str(tmpdir)

    uerror, Derror = run_rexi_sw(dirname, coefficients, approx_type)

    assert uerror < 1e-10, 'u values in REXI linear shallow water wave test do not match KGO values'
    assert Derror < 1e-10, 'D values in REXI linear shallow water wave test do not match KGO values'


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize("algorithm", ["REXI_Haut", "REXI_Caliari", "REXII_Caliari"])
def test_parallel_rexi_sw(tmpdir, algorithm):

    match algorithm:
        case "REXI_Haut":
            coefficients="Haut"
            approx_type = "REXI"
        case "REXI_Caliari":
            coefficients="Caliari"
            approx_type = "REXI"
        case "REXII_Caliari":
            coefficients="Caliari"
            approx_type = "REXII"
        case _:
            raise ValueError("Algorithm must be one of: REXI_Haut, REXI_Caliari or REXII_Caliari.")

    dirname = str(tmpdir)
    ensemble = Ensemble(COMM_WORLD, 1)

    uerror, Derror = run_rexi_sw(dirname, coefficients, approx_type, ensemble=ensemble)

    assert uerror < 1e-10, 'u values in REXI linear shallow water wave test do not match KGO values'
    assert Derror < 1e-10, 'D values in REXI linear shallow water wave test do not match KGO values'
