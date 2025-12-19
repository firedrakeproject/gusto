"""
Testing inner loop physics and the MoistThermalSWSolver.
Based on the moist thermal gravity wave test. Takes 3 timesteps with inner
loop physics and compares this to the solution using standard, end-of-timestep
physics and the ThermalSWSolver.
"""

from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate, pi, sqrt,
                       min_value, cos, sin, Constant, exp, Function, dx)
import numpy as np


def set_up_model_objects(mesh, dt, q0, beta2, nu, physics_type, output):

    # ----------------------------------------------------------------- #
    # Test case parameters
    # ----------------------------------------------------------------- #

    physics_beta = 0.5
    H = 5960.

    # moist shallow water parameters
    gamma_v = 1

    # Domain
    domain = Domain(mesh, dt, "BDM", degree=1)

    # Equation parameters
    parameters = ShallowWaterParameters(mesh, H=H, nu=nu,
                                        beta2=beta2, q0=q0)

    # Equation
    tracers = [WaterVapour(space='DG'), CloudWater(space='DG')]
    if physics_type == "final":
        eqns = ThermalShallowWaterEquations(domain, parameters,
                                            active_tracers=tracers)
    elif physics_type == "inner_loop":
        eqns = ThermalShallowWaterEquations(domain, parameters,
                                            active_tracers=tracers)

        _, Dbar, bbar, qvbar, _ = eqns.X_ref.subfunctions[::]
        _, D, b, qv, _ = eqns.X.subfunctions[::]
        _, _, lamda, tau1, tau2 = eqns.tests[::]

        # Get parameters from equation
        q0 = eqns.parameters.q0
        nu = eqns.parameters.nu
        beta2 = eqns.parameters.beta2
        g = eqns.parameters.g
        H = eqns.parameters.H
        b_ebar = bbar - beta2*qvbar
        sat_expr = q0*H/(Dbar) * exp(nu*(1 - b_ebar/g))
        P_expr = (
            qv - sat_expr*(-D/Dbar - b*nu/g + qv*nu*beta2/g)
        )
        bform = subject(prognostic(physics_label(physics_beta * lamda * beta2 * P_expr * dx)), eqns.X)
        eqns.residual += bform

        qvform = subject(prognostic(physics_label(physics_beta * tau1 * P_expr * dx)), eqns.X)
        eqns.residual += qvform

        qcform = subject(prognostic(physics_label(-physics_beta * tau2 * P_expr * dx)), eqns.X)
        eqns.residual += qcform

    io = IO(domain, output)

    # Limiters
    DG1limiter = DG1Limiter(domain.spaces('DG'))
    zerolimiter = ZeroLimiter(domain.spaces('DG'))

    physics_sublimiters = {'water_vapour': zerolimiter,
                           'cloud_water': zerolimiter}

    physics_limiter = MixedFSLimiter(eqns, physics_sublimiters)

    transport_methods = [DGUpwind(eqns, field_name) for field_name in eqns.field_names]

    transported_fields = [TrapeziumRule(domain, "u"),
                          SSPRK3(domain, "D"),
                          SSPRK3(domain, "b", limiter=DG1limiter),
                          SSPRK3(domain, "water_vapour", limiter=DG1limiter),
                          SSPRK3(domain, "cloud_water", limiter=DG1limiter)]

    # Physics
    def phys_sat_func(x_in):
        D = x_in.subfunctions[1]
        b = x_in.subfunctions[2]
        q_v = x_in.subfunctions[3]
        b_e = Function(b.function_space()).interpolate(b - beta2*q_v)
        return (q0*H/D) * exp(nu*(1 - b_e/parameters.g))

    # Physics schemes
    sat_adj = SWSaturationAdjustment(eqns, phys_sat_func,
                                     time_varying_saturation=True,
                                     parameters=parameters,
                                     thermal_feedback=True,
                                     beta2=beta2, gamma_v=gamma_v)

    physics_schemes = [(sat_adj, ForwardEuler(domain, limiter=physics_limiter))]

    # build timestepper
    if physics_type == "final":
        stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                          transport_methods,
                                          final_physics_schemes=physics_schemes,
                                          num_outer=2, num_inner=2)
    elif physics_type == "inner_loop":
        solver_parameters = monolithic_solver_parameters()
        stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                          transport_methods,
                                          inner_physics_schemes=physics_schemes,
                                          num_outer=2, num_inner=2,
                                          solver_prognostics=eqns.field_names,
                                          linear_solver_parameters=solver_parameters)

    return stepper, eqns


def initialise_fields(eqns, stepper):

    params = eqns.parameters
    u_max = 20.
    R = 6371220.

    # perturbation parameters
    R0 = pi/9.
    R0sq = R0**2
    lamda_c = -pi/2.
    phi_c = pi/6.
    # parameters for initial buoyancy
    phi_0 = 3e4
    epsilon = 1/300
    theta_0 = epsilon*phi_0**2

    # Perturbation
    x = SpatialCoordinate(eqns.domain.mesh)
    lamda, phi, _ = lonlatr_from_xyz(x[0], x[1], x[2])
    lsq = (lamda - lamda_c)**2
    thsq = (phi - phi_c)**2
    rsq = min_value(R0sq, lsq+thsq)
    r = sqrt(rsq)
    pert = 2000.0 * (1 - r/R0)

    # saturation function (depending on b_e)
    def sat_func(D, b_e):
        return (params.q0*params.H/D) * exp(params.nu*(1 - b_e/params.g))

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")
    b0 = stepper.fields("b")
    v0 = stepper.fields("water_vapour")
    c0 = stepper.fields("cloud_water")

    # velocity
    uexpr = xyz_vector_from_lonlatr(u_max*cos(phi), 0, 0, x)

    # buoyancy
    Omega = params.Omega
    g = params.g
    w = Omega*R*u_max + (u_max**2)/2
    sigma = w/10
    numerator = theta_0 + sigma*((cos(phi))**2) * ((w + sigma)*(cos(phi))**2 + 2*(phi_0 - w - sigma))
    denominator = phi_0**2 + (w + sigma)**2*(sin(phi))**4 - 2*phi_0*(w + sigma)*(sin(phi))**2
    theta = numerator/denominator
    b_guess = params.g * (1 - theta)

    # depth
    Dexpr = params.H - (1/g)*(w + sigma)*((sin(phi))**2) + pert

    # iterate to find initial b_e_expr, from which the vapour and saturation
    # function are recovered
    q_t = 0.03

    def iterate():
        n_iterations = 10
        D_init = Function(D0.function_space()).interpolate(Dexpr)
        b_init = Function(b0.function_space()).interpolate(b_guess)
        b_e_init = Function(b0.function_space()).interpolate(b_init - params.beta2*q_t)
        q_v_init = Function(v0.function_space()).interpolate(q_t)
        for i in range(n_iterations):
            q_sat_expr = sat_func(D_init, b_e_init)
            dq_sat_dq_v_expr = params.nu*params.beta2/g*q_sat_expr
            q_v_init.interpolate(q_v_init - (q_sat_expr - q_v_init)/(dq_sat_dq_v_expr - 1.0))
            b_e_init.interpolate(b_init - params.beta2*q_v_init)
        return b_e_init

    b_e = iterate()

    initial_sat = sat_func(Dexpr, b_e)

    vexpr = initial_sat

    # back out the initial buoyancy using b_e and q_v
    bexpr = b_e + params.beta2*vexpr

    # cloud is the rest of q_t that isn't vapour
    cexpr = Constant(q_t) - vexpr

    u0.project(uexpr)
    D0.interpolate(Dexpr)
    b0.interpolate(bexpr)
    v0.interpolate(vexpr)
    c0.interpolate(cexpr)

    # Set reference profiles
    Dbar = Function(D0.function_space()).assign(params.H)
    bbar = Function(b0.function_space()).interpolate(bexpr)
    vbar = Function(v0.function_space()).interpolate(vexpr)
    cbar = Function(c0.function_space()).interpolate(cexpr)
    stepper.set_reference_profiles([('D', Dbar), ('b', bbar),
                                    ('water_vapour', vbar), ('cloud_water', cbar)])


def test_inner_loop_physics(tmpdir):

    # Parameters
    q0 = 0.0115
    beta2 = 9.80616*10
    nu = 1.5

    # Set up mesh
    R = 6371220.
    ref = 3
    mesh = IcosahedralSphereMesh(radius=R, refinement_level=ref, degree=2)

    # Set up timestepping parameters
    dt = 600

    dirname_1 = str(tmpdir)+'/standard_physics'
    dirname_2 = str(tmpdir)+'/inner_loop_physics'

    # Set up equations and timesteppers
    output1 = OutputParameters(dirname=dirname_1,
                               dump_vtus=True,
                               checkpoint=True)

    output2 = OutputParameters(dirname=dirname_2,
                               dump_vtus=True,
                               checkpoint=True)

    stepper1, eqns1 = set_up_model_objects(mesh, dt, q0, beta2, nu, physics_type="final", output=output1)
    stepper2, eqns2 = set_up_model_objects(mesh, dt, q0, beta2, nu, physics_type="inner_loop", output=output2)

    # Initialise
    initialise_fields(eqns1, stepper1)
    initialise_fields(eqns2, stepper2)

    # Run
    stepper1.run(t=0.0, tmax=3*dt)
    stepper2.run(t=0.0, tmax=3*dt)

    for field_name in ['u', 'D', 'b', 'water_vapour', 'cloud_water']:
        diff_array = stepper1.fields(field_name).dat.data - stepper2.fields(field_name).dat.data
        error = np.linalg.norm(diff_array) / np.linalg.norm(stepper1.fields(field_name).dat.data)
        assert error < 0.0015, \
            f'Field {field_name} is not the same when using inner loop physics as final physics'
