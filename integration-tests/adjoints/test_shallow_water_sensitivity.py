import pytest

from firedrake import *
from firedrake.adjoint import *
from gusto import *


@pytest.fixture(autouse=True)
def autouse_set_test_tape(set_test_tape):
    pass


@pytest.mark.parametrize("control", ["u", "D"])
@pytest.mark.parametrize("stepper_type", ["BackwardEuler", "RK4", "SemiImplicitQuasiNewton"])
def test_shallow_water(tmpdir, control, stepper_type):
    # setup shallow water parameters
    R = 6371220.
    H = 5960.
    mountain = 2000.
    dt = 300.
    u_max = 20.

    # Domain
    mesh = IcosahedralSphereMesh(radius=R, refinement_level=3, degree=1)
    x, y, z = SpatialCoordinate(mesh)
    domain = Domain(mesh, dt, 'BDM', 1)

    # Equation
    lamda, theta, _ = lonlatr_from_xyz(x, y, z)
    R0 = pi/9.
    R0sq = R0**2
    lamda_c = -pi/2.
    lsq = (lamda - lamda_c)**2
    theta_c = pi/6.
    thsq = (theta - theta_c)**2
    rsq = min_value(R0sq, lsq+thsq)
    r = sqrt(rsq)
    bexpr = mountain * (1 - r/R0)
    b = Function(domain.spaces("DG")).interpolate(bexpr)
    parameters = ShallowWaterParameters(mesh, H=H, topog_expr=b)
    eqn = ShallowWaterEquations(domain, parameters)

    # I/O
    output = OutputParameters(dirname=str(tmpdir), log_courant=False)
    io = IO(domain, output)

    # Don't let an inexact solve get in the way of a good Taylor test
    lu_parameters = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
    }
    mass_parameters = {
        'snes_type': 'ksponly',
        **lu_parameters,
        'ksp_reuse_preconditioner': None,
    }
    snes_parameters = {
        'snes_monitor': None,
        'snes_converged_reason': None,
        'snes_type': 'newtonls',
        'snes_stol': 0,
        'snes_rtol': 1e-10,
        **lu_parameters,
    }
    linear_solver_parameters = {
        'mat_type': 'matfree',
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.HybridizationPC',
        'hybridization': lu_parameters,
    }

    # Transport schemes
    transport_methods = [DGUpwind(eqn, "u"), DGUpwind(eqn, "D")]

    # Time stepper
    if stepper_type == "BackwardEuler":
        stepper = Timestepper(
            eqn, BackwardEuler(domain, solver_parameters=snes_parameters),
            io, spatial_methods=transport_methods
        )
    elif stepper_type == "RK4":
        stepper = Timestepper(
            eqn, RK4(domain, solver_parameters=mass_parameters),
            io, spatial_methods=transport_methods
        )
    else:
        assert stepper_type == "SemiImplicitQuasiNewton"
        transported_fields = [
            TrapeziumRule(domain, "u", solver_parameters=snes_parameters),
            SSPRK3(domain, "D", solver_parameters=mass_parameters),
        ]
        stepper = SemiImplicitQuasiNewton(
            eqn, io, transported_fields, transport_methods,
            linear_solver_parameters=linear_solver_parameters
        )

    u0 = stepper.fields('u')
    D0 = stepper.fields('D')

    uexpr = as_vector([-u_max*y/R, u_max*x/R, 0.0])
    g = parameters.g
    Omega = parameters.Omega
    Rsq = R**2
    Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*z**2/Rsq)/g - bexpr

    # controls m for the initial conditions
    m_u = Function(u0.function_space()).project(uexpr)
    m_D = Function(D0.function_space()).interpolate(Dexpr)

    if control == "u":
        m = m_u
    else:
        m = m_D

    Dbar = Function(D0.function_space()).assign(H)
    stepper.set_reference_profiles([('D', Dbar)])

    # These are the only operations we are interested in rerunning with the tape.
    with set_working_tape() as tape:
        # initialise the solution
        u0.assign(m_u)
        D0.assign(m_D)

        # propagate forwards
        stepper.run(0., 2*dt)

        if control == "u":
            u_tf = stepper.fields('u')
            J = assemble(inner(u_tf, u_tf)*dx)**2
        else:
            D_tf = stepper.fields('D')
            dD = D_tf - (H - b)
            J = assemble(0.5*g*inner(dD, dD)*dx)**2

        Jhat = ReducedFunctional(J, Control(m), tape=tape)

    # Perturbation directions for taylor test
    # pyadjoint will multiply h by 1e-2, 1e-4 etc so we pre-multiply by 10
    if control == "u":
        h = Function(u0.function_space()).interpolate(
            10.0*u_max*as_vector([cos(2*pi*y/R), 0.0, 1 + cos(pi*z/R)*sin(4*pi*x/R)]))
    else:
        h = Function(D0.function_space()).interpolate(1.0*(Dexpr - (H - b))*cos(pi*z/R))

    assert abs(float(J) - float(Jhat(m))) < 1e-10, "Functional re-evaluation does not match original evaluation."

    # Check the TLM explicitly before checking the Hessian (which relies on the tlm)
    assert taylor_test(Jhat, m, h, dJdm=Jhat.tlm(h)) > 1.95, "TLM is not second order accurate."

    # Check the re-evaluation, derivative, and Hessian all converge at the expected rates.
    taylor = taylor_to_dict(Jhat, m, h)
    assert min(taylor['R0']['Rate']) > 0.95, taylor['R0']
    assert min(taylor['R1']['Rate']) > 1.95, taylor['R1']
    assert min(taylor['R2']['Rate']) > 2.95, taylor['R2']
