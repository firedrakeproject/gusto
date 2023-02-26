"""
This runs a shallow water simulation on the fplane with 3 waves that interact.
"""

from gusto import *
from firedrake import PeriodicSquareMesh, SpatialCoordinate, Function, cos, pi
from os.path import join, abspath, dirname
import pytest


def run_sw_fplane(tmpdir):
    # Domain
    Nx = 32
    Ny = Nx
    Lx = 10
    mesh = PeriodicSquareMesh(Nx, Ny, Lx, quadrilateral=True)
    dt = 0.01
    domain = Domain(mesh, dt, 'RTCF', 1)

    # Equation
    H = 2
    g = 50
    parameters = ShallowWaterParameters(H=H, g=g)
    f0 = 10
    fexpr = Constant(f0)
    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr)

    # I/O
    output = OutputParameters(dirname=str(tmpdir)+"/sw_fplane",
                              dumpfreq=1,
                              log_level='INFO')

    io = IO(domain, output, diagnostic_fields=[CourantNumber()])

    # Transport schemes
    transported_fields = []
    transported_fields.append((ImplicitMidpoint(domain, "u")))
    transported_fields.append((SSPRK3(domain, "D")))

    # Time stepper
    stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")
    x, y = SpatialCoordinate(mesh)
    Ly = Lx
    N0 = 0.1
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

    u0.project(u_expr)
    D0.interpolate(D_expr)

    Dbar = Function(D0.function_space()).assign(H)
    stepper.set_reference_profiles([('D', Dbar)])

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=20*dt)

    # State for checking checkpoints
    checkpoint_name = 'sw_fplane_chkpt'
    new_path = join(abspath(dirname(__file__)), '..', f'data/{checkpoint_name}')
    check_eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr)
    check_output = OutputParameters(dirname=tmpdir+"/sw_fplane",
                                    checkpoint_pickup_filename=new_path)
    check_io = IO(domain, check_output)
    check_stepper = SemiImplicitQuasiNewton(check_eqn, check_io, [])
    check_stepper.run(t=0, tmax=0, pickup=True)

    return stepper, check_stepper


def test_sw_fplane(tmpdir):

    stepper, check_stepper = run_sw_fplane(tmpdir)

    for variable in ['u', 'D']:
        new_variable = stepper.fields(variable)
        check_variable = check_stepper.fields(variable)
        error = norm(new_variable - check_variable) / norm(check_variable)

        # Slack values chosen to be robust to different platforms
        assert error < 1e-10, f'Values for {variable} in ' + \
            'shallow water fplane test do not match KGO values'

