"""
A moist thermal in an unsaturated atmosphere, including a rain species. This
test is based on that of Grabowski and Clark, 1991:
``Cloud–environment interface instability: Rising thermal calculations in two
spatial dimensions'', JAS.

and is described in Bendall et al, 2020:
``A compatible finite‐element discretisation for the moist compressible Euler
equations'', QJRMS.

As the thermal rises, water vapour condenses into cloud and forms rain.
Limiters are applied to the transport of the water species.

This configuration uses the lowest-order finite elements, and the recovery
wrapper to provide higher-order accuracy.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import (
    PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate, conditional, cos, pi,
    sqrt, exp, TestFunction, dx, TrialFunction, Constant, Function, errornorm,
    LinearVariationalProblem, LinearVariationalSolver, as_vector
)
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
from gusto import (
    Domain, IO, OutputParameters, SemiImplicitQuasiNewton, SSPRK3, DGUpwind,
    Perturbation, RecoverySpaces, BoundaryMethod, Recoverer, Fallout,
    Coalescence, SaturationAdjustment, EvaporationOfRain, thermodynamics,
    CompressibleParameters, CompressibleEulerEquations, CompressibleSolver,
    unsaturated_hydrostatic_balance, WaterVapour, CloudWater, Rain,
    RelativeHumidity, ForwardEuler, MixedFSLimiter, ZeroLimiter
)

unsaturated_bubble_defaults = {
    'ncolumns': 180,
    'nlayers': 120,
    'dt': 1.0,
    'tmax': 600.,
    'dumpfreq': 300,
    'dirname': 'unsaturated_bubble'
}


def unsaturated_bubble(
        ncolumns=unsaturated_bubble_defaults['ncolumns'],
        nlayers=unsaturated_bubble_defaults['nlayers'],
        dt=unsaturated_bubble_defaults['dt'],
        tmax=unsaturated_bubble_defaults['tmax'],
        dumpfreq=unsaturated_bubble_defaults['dumpfreq'],
        dirname=unsaturated_bubble_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    domain_width = 3600.         # domain width (m)
    domain_height = 2400.        # domain height (m)
    zc = 800.                    # height of centre of perturbation (m)
    r1 = 300.                    # outer radius of perturbation (m)
    r2 = 200.                    # inner radius of perturbation (m)
    Tsurf = 283.0                # surface temperature (K)
    psurf = 85000.               # surface pressure (Pa)
    rel_hum_background = 0.2     # background relative humidity (dimensionless)
    S = 1.3e-5                   # height factor for theta profile (1/m)
    max_outer_solve_count = 20   # max num outer iterations for initialisation
    max_inner_solve_count = 10   # max num inner iterations for initialisation
    tol_initialisation = 1e-10   # tolerance for initialisation

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    element_order = 0
    u_eqn_type = 'vector_advection_form'

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    base_mesh = PeriodicIntervalMesh(ncolumns, domain_width)
    mesh = ExtrudedMesh(base_mesh, nlayers, layer_height=domain_height/nlayers)
    domain = Domain(mesh, dt, "CG", element_order)

    # Equation
    params = CompressibleParameters()
    tracers = [WaterVapour(), CloudWater(), Rain()]
    eqns = CompressibleEulerEquations(
        domain, params, active_tracers=tracers, u_transport_option=u_eqn_type
    )

    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_vtus=False, dump_nc=True,
        dumplist=['cloud_water', 'rain']
    )
    diagnostic_fields = [
        RelativeHumidity(eqns), Perturbation('theta'), Perturbation('rho'),
        Perturbation('water_vapour'), Perturbation('RelativeHumidity')
    ]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes -- specify options for using recovery wrapper
    boundary_methods = {'DG': BoundaryMethod.taylor,
                        'HDiv': BoundaryMethod.taylor}

    recovery_spaces = RecoverySpaces(domain, boundary_method=boundary_methods, use_vector_spaces=True)

    u_opts = recovery_spaces.HDiv_options
    rho_opts = recovery_spaces.DG_options
    theta_opts = recovery_spaces.theta_options

    VDG1 = domain.spaces("DG1_equispaced")
    limiter = VertexBasedLimiter(VDG1)

    transported_fields = [
        SSPRK3(domain, "u", options=u_opts),
        SSPRK3(domain, "rho", options=rho_opts),
        SSPRK3(domain, "theta", options=theta_opts),
        SSPRK3(domain, "water_vapour", options=theta_opts, limiter=limiter),
        SSPRK3(domain, "cloud_water", options=theta_opts, limiter=limiter),
        SSPRK3(domain, "rain", options=theta_opts, limiter=limiter)
    ]

    transport_methods = [
        DGUpwind(eqns, field) for field in
        ["u", "rho", "theta", "water_vapour", "cloud_water", "rain"]
    ]

    # Linear solver
    linear_solver = CompressibleSolver(eqns)

    # Physics schemes
    Vt = domain.spaces('theta')
    rainfall_method = DGUpwind(eqns, 'rain', outflow=True)
    zero_limiter = MixedFSLimiter(
        eqns,
        {'water_vapour': ZeroLimiter(Vt), 'cloud_water': ZeroLimiter(Vt)}
    )
    physics_schemes = [
        (Fallout(eqns, 'rain', domain, rainfall_method), SSPRK3(domain)),
        (Coalescence(eqns), ForwardEuler(domain)),
        (EvaporationOfRain(eqns), ForwardEuler(domain)),
        (SaturationAdjustment(eqns), ForwardEuler(domain, limiter=zero_limiter))
    ]

    # Time stepper
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transported_fields, transport_methods,
        linear_solver=linear_solver, physics_schemes=physics_schemes
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")
    water_v0 = stepper.fields("water_vapour")
    water_c0 = stepper.fields("cloud_water")
    water_r0 = stepper.fields("rain")

    # spaces
    Vr = domain.spaces("DG")
    x, z = SpatialCoordinate(mesh)
    quadrature_degree = (4, 4)
    dxp = dx(degree=(quadrature_degree))

    physics_boundary_method = BoundaryMethod.extruded

    # Define constant theta_e and water_t
    exner_surf = (psurf / eqns.parameters.p_0) ** eqns.parameters.kappa
    theta_surf = thermodynamics.theta(eqns.parameters, Tsurf, psurf)
    theta_d = Function(Vt).interpolate(theta_surf * exp(S*z))
    rel_hum = Function(Vt).assign(rel_hum_background)

    # Calculate hydrostatic fields
    unsaturated_hydrostatic_balance(
        eqns, stepper.fields, theta_d, rel_hum,
        exner_boundary=Constant(exner_surf)
    )

    # make mean fields
    theta_b = Function(Vt).assign(theta0)
    rho_b = Function(Vr).assign(rho0)
    water_vb = Function(Vt).assign(water_v0)

    # define perturbation to RH
    xc = domain_width / 2
    r = sqrt((x - xc) ** 2 + (z - zc) ** 2)

    rel_hum_pert_expr = conditional(
        r > r1,
        0.0,
        conditional(
            r > r2,
            (1 - rel_hum_background) * cos(pi*(r - r2) / (2*(r1 - r2)))**2,
            1 - rel_hum_background
        )
    )
    rel_hum.interpolate(rel_hum_background + rel_hum_pert_expr)

    # now need to find perturbed rho, theta_vd and r_v
    # follow approach used in unsaturated hydrostatic setup
    rho_averaged = Function(Vt)
    rho_recoverer = Recoverer(
        rho0, rho_averaged, boundary_method=physics_boundary_method
    )
    rho_eval = Function(Vr)
    water_v_eval = Function(Vt)
    delta = 1.0

    R_d = eqns.parameters.R_d
    R_v = eqns.parameters.R_v
    epsilon = R_d / R_v

    # make expressions to evaluate residual
    exner_expr = thermodynamics.exner_pressure(eqns.parameters, rho_averaged, theta0)
    p_expr = thermodynamics.p(eqns.parameters, exner_expr)
    T_expr = thermodynamics.T(eqns.parameters, theta0, exner_expr, water_v0)
    water_v_expr = thermodynamics.r_v(eqns.parameters, rel_hum, T_expr, p_expr)
    rel_hum_expr = thermodynamics.RH(eqns.parameters, water_v0, T_expr, p_expr)
    rel_hum_eval = Function(Vt)

    # set-up rho problem to keep exner constant
    gamma = TestFunction(Vr)
    rho_trial = TrialFunction(Vr)
    lhs = gamma * rho_trial * dxp
    rhs = gamma * (rho_b * theta_b / theta0) * dxp
    rho_problem = LinearVariationalProblem(lhs, rhs, rho_eval)
    rho_solver = LinearVariationalSolver(rho_problem)

    for i in range(max_outer_solve_count):
        # calculate averaged rho
        rho_recoverer.project()

        rel_hum_eval.interpolate(rel_hum_expr)
        if errornorm(rel_hum_eval, rel_hum) < tol_initialisation:
            break

        # first solve for r_v
        for _ in range(max_inner_solve_count):
            water_v_eval.interpolate(water_v_expr)
            water_v0.assign(water_v0 * (1 - delta) + delta * water_v_eval)

            # compute theta_vd
            theta0.interpolate(theta_d * (1 + water_v0 / epsilon))

            # test quality of solution by re-evaluating expression
            rel_hum_eval.interpolate(rel_hum_expr)
            if errornorm(rel_hum_eval, rel_hum) < tol_initialisation:
                break

        # now solve for rho with theta_vd and w_v guesses
        rho_solver.solve()

        # damp solution
        rho0.assign(rho0 * (1 - delta) + delta * rho_eval)

        if i == max_outer_solve_count:
            raise RuntimeError(
                f'Hydrostatic balance solve has not converged within {i} iterations'
            )

    # Set wind, cloud and rain to be zero
    zero = Constant(0.0, domain=mesh)
    u0.project(as_vector([zero, zero]))
    water_c0.interpolate(zero)
    water_r0.interpolate(zero)

    # initialise reference profiles
    stepper.set_reference_profiles(
        [('rho', rho_b), ('theta', theta_b), ('water_vapour', water_vb)]
    )

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=tmax)

# ---------------------------------------------------------------------------- #
# MAIN
# ---------------------------------------------------------------------------- #


if __name__ == "__main__":

    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--ncolumns',
        help="The number of columns in the vertical slice mesh.",
        type=int,
        default=unsaturated_bubble_defaults['ncolumns']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=unsaturated_bubble_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=unsaturated_bubble_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=unsaturated_bubble_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=unsaturated_bubble_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=unsaturated_bubble_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    unsaturated_bubble(**vars(args))
