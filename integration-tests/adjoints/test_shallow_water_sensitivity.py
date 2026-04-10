import pytest
import numpy as np

from firedrake import *
from firedrake.adjoint import *
from pyadjoint import get_working_tape
from gusto import *


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


@pytest.mark.parametrize("control", ["u", "D"])
@pytest.mark.parametrize("stepper_type", ["BackwardEuler", "RK4", "SemiImplicitQuasiNewton"])
def test_shallow_water(tmpdir, control, stepper_type):
    assert get_working_tape()._blocks == []
    # setup shallow water parameters
    R = 1
    H = 1e-5
    dt = 0.01

    # Domain
    mesh = IcosahedralSphereMesh(radius=R, refinement_level=3, degree=2)
    x = SpatialCoordinate(mesh)
    domain = Domain(mesh, dt, 'BDM', 1)

    # Equation
    lamda, theta, _ = lonlatr_from_xyz(x[0], x[1], x[2])
    R0 = pi/9.
    R0sq = R0**2
    lamda_c = -pi/2.
    lsq = (lamda - lamda_c)**2
    theta_c = pi/6.
    thsq = (theta - theta_c)**2
    rsq = min_value(R0sq, lsq+thsq)
    r = sqrt(rsq)
    bexpr = 1e-6 * (1 - r/R0)
    parameters = ShallowWaterParameters(mesh, H=H, topog_expr=bexpr)
    eqn = ShallowWaterEquations(domain, parameters)

    # I/O
    output = OutputParameters(dirname=str(tmpdir), log_courant=False)
    io = IO(domain, output)

    # Don't let an inexact solve get in the way of a good Taylor test
    linear_solver_parameters = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
    }
    solver_parameters = {
        'snes_rtol': 1e-10,
        **linear_solver_parameters
    }

    # Transport schemes
    transport_methods = [DGUpwind(eqn, "u"), DGUpwind(eqn, "D")]

    # Time stepper
    if stepper_type == "BackwardEuler":
        stepper = Timestepper(
            eqn, BackwardEuler(domain, solver_parameters=linear_solver_parameters),
            io, spatial_methods=transport_methods
        )
    elif stepper_type == "RK4":
        stepper = Timestepper(
            eqn, RK4(domain, solver_parameters=linear_solver_parameters),
            io, spatial_methods=transport_methods
        )
    else:
        assert stepper_type == "SemiImplicitQuasiNewton"
        transported_fields = [TrapeziumRule(domain, "u", solver_parameters=solver_parameters), SSPRK3(domain, "D", solver_parameters=solver_parameters)]
        stepper = SemiImplicitQuasiNewton(
            eqn, io, transported_fields, transport_methods
        )

    u0 = stepper.fields('u')
    D0 = stepper.fields('D')

    u_max = 1e-4   # Maximum amplitude of the zonal wind (m/s)
    uexpr = as_vector([-u_max*x[1], u_max*x[0], 0.0])
    g = parameters.g
    Omega = parameters.Omega
    R = 1
    Rsq = R**2
    Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g - bexpr

    # controls m for the initial conditions
    m_u = Function(u0.function_space()).project(uexpr)
    m_D = Function(D0.function_space()).interpolate(Dexpr)

    Dbar = Function(D0.function_space()).assign(H)
    stepper.set_reference_profiles([('D', Dbar)])

    # These are the only operations we are interested in rerunning with the tape.
    with set_working_tape() as tape:
        # initialise the solution
        u0.assign(m_u)
        D0.assign(m_D)

        # propagate forwards
        stepper.run(0., 10*dt)

        u_tf = stepper.fields('u')  # Final velocity field
        D_tf = stepper.fields('D')  # Final depth field

        J = assemble(inner(u_tf, u_tf)*dx)**2

        if control == "u":
            controls = [Control(m_u)]
        else:
            controls = [Control(m_D)]
        Jhat = ReducedFunctional(J, controls, tape=tape)

    # perturbation directions for taylor test
    h_D = Function(D0.function_space()).interpolate(1e-3*sin(x[0]))
    h_u = Function(u0.function_space()).interpolate(1e-4*as_vector([cos(4*pi*x[1]), sin(4*pi*x[0]), 0]))

    if control == "u":
        m = [m_u]
        h = [h_u]
    else:
        m = [m_D]
        h = [h_D]
    assert np.allclose(J, Jhat(m)), "Functional re-evaluation does not match original evaluation."

    # # Check the TLM explicitly before checking the Hessian (which relies on the tlm)
    assert taylor_test(Jhat, m, h, dJdm=Jhat.tlm(h)) > 1.95, "TLM is not second order accurate."

    assert taylor_test(Jhat, m, h) > 1.95, "Adjoint derivative is not second order accurate."

    # Check the re-evaluation, derivative, and Hessian all converge at the expected rates.
    taylor = taylor_to_dict(Jhat, m, h)
    assert min(taylor['R0']['Rate']) > 0.95, taylor['R0']
    assert min(taylor['R1']['Rate']) > 1.95, taylor['R1']
    assert min(taylor['R2']['Rate']) > 2.95, taylor['R2']
