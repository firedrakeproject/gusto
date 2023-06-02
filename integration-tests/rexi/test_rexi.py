"""
This runs the wave scenario from Schreiber et al 2018 for the linear f-plane
shallow water equations and compares the REXI output to the Implicit Midpoint
output to confirm that REXI is correct
"""

from os.path import join, abspath, dirname
from gusto import *
from gusto.rexi import *
from firedrake import (PeriodicUnitSquareMesh, SpatialCoordinate, Constant, sin,
                       cos, pi, norm, as_vector, Function, File)


def run_rexi_sw(tmpdir):
    # timestepping parameters
    dt = 0.001
    tmax = 0.1

    # Domain
    n = 20
    mesh = PeriodicUnitSquareMesh(n, n)
    domain = Domain(mesh, dt, 'BDM', 1)

    # set up linear shallow water equations
    H = 1
    f = 1.
    g = 1
    parameters = ShallowWaterParameters(H=H, g=g)
    eqns = LinearShallowWaterEquations(domain, parameters, fexpr=Constant(f))

    # I/O
    output = OutputParameters(dirname=str(tmpdir)+"/waves_sw",
                              log_level='INFO')
    io = IO(domain, output)

    # Timestepper
    stepper = Timestepper(eqns, ImplicitMidpoint(domain), io)

    # Initial conditions
    x, y = SpatialCoordinate(mesh)
    u0 = stepper.fields("u")
    D0 = stepper.fields("D")
    uexpr = as_vector([cos(8*pi*x)*cos(2*pi*y), cos(4*pi*x)*cos(4*pi*y)])
    Dexpr = sin(4*pi*x)*cos(2*pi*y) - 0.2*cos(4*pi*x)*sin(4*pi*y)
    u0.project(uexpr)
    D0.interpolate(Dexpr)

    # Compute implicit midpoint solution
    stepper.run(t=0, tmax=tmax)
    usoln = stepper.fields("u")
    Dsoln = stepper.fields("D")

    # Compute exponential solution and write out
    rexi_output = File(str(tmpdir)+"/waves_sw/rexi.pvd")
    domain = Domain(mesh, dt, 'BDM', 1)
    parameters = ShallowWaterParameters(H=H, g=g)
    linearisation_map = lambda t: \
        t.get(prognostic) in ["u", "D"] \
        and (any(t.has_label(time_derivative, pressure_gradient, coriolis))
             or (t.get(prognostic) == "D" and t.has_label(transport)))
    eqns = ShallowWaterEquations(domain, parameters, fexpr=Constant(f),
                                 linearisation_map=linearisation_map)

    U_in = Function(eqns.function_space)
    Uexpl = Function(eqns.function_space)
    u, D = U_in.split()
    u.project(uexpr)
    D.interpolate(Dexpr)
    rexi_output.write(u, D)

    rexi = Rexi(eqns, RexiParameters())
    Uexpl.assign(rexi.solve(U_in, tmax))

    uexpl, Dexpl = Uexpl.split()
    u.assign(uexpl)
    D.assign(Dexpl)
    rexi_output.write(u, D)    

    return usoln, Dsoln, uexpl, Dexpl


def test_rexi_sw(tmpdir):

    dirname = str(tmpdir)

    usoln, Dsoln, uexpl, Dexpl = run_rexi_sw(dirname)

    uerror = norm(usoln - uexpl) / norm(usoln)
    assert uerror < 0.04

    Derror = norm(Dsoln - Dexpl) / norm(Dsoln)
    assert Derror < 0.02
