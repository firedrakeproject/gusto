"""
This runs a shallow water simulation on the fplane with 3 waves
that interact and checks the results agains a known checkpointed answer.
"""

from os.path import join, abspath, dirname
from gusto import *
from firedrake import (PeriodicSquareMesh, SpatialCoordinate, Function,
                       cos, pi, as_vector, sin)
from firedrake.adjoint import *
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def handle_taping():
    yield
    tape = get_working_tape()
    tape.clear_tape()


@pytest.fixture(autouse=True, scope="module")
def handle_annotation():
    from firedrake.adjoint import annotate_tape, continue_annotation
    if not annotate_tape():
        continue_annotation()
    yield
    # Ensure annotation is paused when we finish.
    annotate = annotate_tape()
    if annotate:
        pause_annotation()


def run_sw_fplane(tmpdir):
    # Domain
    mesh_name = 'sw_fplane_mesh'
    Nx = 32
    Ny = Nx
    Lx = 10
    # mesh = PeriodicSquareMesh(Nx, Ny, Lx, quadrilateral=True, name=mesh_name)
    mesh = PeriodicSquareMesh(Nx, Ny, Lx, name=mesh_name)
    dt = 0.001
    domain = Domain(mesh, dt, 'BDM', 1)

    # Equation
    H = 2
    g = 50
    f0 = 10
    parameters = ShallowWaterParameters(mesh, rotation=CoriolisOptions.fplane,
                                        f0=f0, H=H, g=g)
    eqns = ShallowWaterEquations(domain, parameters)

    # I/O
    output = OutputParameters(
        dirname=str(tmpdir)+"/sw_fplane",
        dumpfreq=1,
        checkpoint=True
    )

    io = IO(domain, output, diagnostic_fields=[CourantNumber()])

    # Transport schemes
    # vorticity_transport = VorticityTransport(domain, eqns, supg=True)
    # transported_fields = [
    #     TrapeziumRule(domain, "u", augmentation=vorticity_transport),
    #     SSPRK3(domain, "D")
    # ]
    transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]

    # Time stepper
    # stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
    #                                   transport_methods,
    #                                   num_outer=4, num_inner=1)
    linear_solver_parameters = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
    }
    stepper = Timestepper(
        eqns, RK4(domain, solver_parameters=linear_solver_parameters),
        io, spatial_methods=transport_methods
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")
    x, y = SpatialCoordinate(mesh)
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

    u_expr = as_vector([u1*10+u2, v1*10+v2])
    D_expr = H + sqrt(H/g)*(phi1+phi2)

    # controls m for the initial conditions
    m_u = Function(u0.function_space()).project(u_expr)
    m_D = Function(D0.function_space()).interpolate(D_expr)

    with set_working_tape() as tape:
        u0.assign(m_u)
        D0.assign(m_D)

        Dbar = Function(D0.function_space()).assign(H)
        stepper.set_reference_profiles([('D', Dbar)])

        # ------------------------------------------------------------------------ #
        # Run
        # ------------------------------------------------------------------------ #

        stepper.run(t=0, tmax=10*dt)


        u_tf = stepper.fields('u')  # Final velocity field
        D_tf = stepper.fields('D')  # Final depth field

        # J = assemble(u_tf**4*dx + g*D_tf**4*dx)
        J = assemble(inner(u_tf, u_tf)**2*dx)

        # control = [Control(m_D), Control(m_u)]  # Control variables
        control = [Control(m_u)]  # Control variables
        Jhat = ReducedFunctional(J, control, tape=tape)

    # m = [m_D, m_u]
    m = [m_u]
    # assert np.allclose(J, Jhat(m)), "Functional re-evaluation does not match original evaluation."

    # perturbation directions for taylor test
    # h_D = Function(D0.function_space())
    # h_D.interpolate(D_expr - H)

    h_u = Function(u0.function_space())
    h_u.project(1e-1*u_expr)
    #
    # # h_D.interpolate(1e-100*(Dexpr - (H - bexpr)))

    # h = [h_D, h_u]
    h = [h_u]

    # # Check the TLM explicitly before checking the Hessian (which relies on the tlm)
    # assert taylor_test(Jhat, m, h, dJdm=Jhat.tlm(h)) > 1.95, "TLM is not second order accurate."
    #
    # assert taylor_test(Jhat, m, h) > 1.95, "Adjoint derivative is not second order accurate."

    # Check the re-evaluation, derivative, and Hessian all converge at the expected rates.
    taylor = taylor_to_dict(Jhat, m, h)
    assert min(taylor['R0']['Rate']) > 0.95, taylor['R0']
    assert min(taylor['R1']['Rate']) > 1.95, taylor['R1']
    assert min(taylor['R2']['Rate']) > 2.95, taylor['R2']

    breakpoint()

    # State for checking checkpoints
    checkpoint_name = 'sw_fplane_chkpt.h5'
    new_path = join(abspath(dirname(__file__)), '..', f'data/{checkpoint_name}')
    check_output = OutputParameters(dirname=tmpdir+"/sw_fplane",
                                    checkpoint_pickup_filename=new_path,
                                    checkpoint=True)
    check_mesh = pick_up_mesh(check_output, mesh_name)
    check_domain = Domain(check_mesh, dt, 'RTCF', 1)
    check_parameters = ShallowWaterParameters(check_mesh,
                                              rotation=CoriolisOptions.fplane,
                                              f0=f0, H=H, g=g)
    check_eqn = ShallowWaterEquations(check_domain, check_parameters)
    check_io = IO(check_domain, output=check_output)
    check_stepper = SemiImplicitQuasiNewton(check_eqn, check_io, [], [])
    check_stepper.io.pick_up_from_checkpoint(check_stepper.fields)

    return stepper, check_stepper


def test_sw_fplane(tmpdir):

    dirname = str(tmpdir)
    stepper, check_stepper = run_sw_fplane(dirname)

    for variable in ['u', 'D']:
        new_variable = stepper.fields(variable)
        check_variable = check_stepper.fields(variable)
        diff_array = new_variable.dat.data - check_variable.dat.data
        error = np.linalg.norm(diff_array) / np.linalg.norm(check_variable.dat.data)

        # Slack values chosen to be robust to different platforms
        assert error < 1e-10, f'Values for {variable} in ' + \
            'shallow water fplane test do not match KGO values'
