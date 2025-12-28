"""
A gravity wave on the sphere, solved with the moist thermal shallow water
equations. The initial conditions are saturated and cloudy everywhere.

This example is implemented in two versions:
- The first uses the equivalent buoyancy formulation and uses a hybridised
  linear solver.
- The second uses the standard buoyancy formulation and a monolithic linear
  solver that includes moist physics.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import (
    SpatialCoordinate, pi, sqrt, min_value, cos, Constant, Function, exp, sin,
)
from gusto import (
    Domain, IO, OutputParameters, DGUpwind, ShallowWaterParameters,
    ThermalShallowWaterEquations, lonlatr_from_xyz, SubcyclingOptions,
    RungeKuttaFormulation, SSPRK3, MeridionalComponent,
    SemiImplicitQuasiNewton, ForwardEuler, WaterVapour, CloudWater,
    xyz_vector_from_lonlatr, SWSaturationAdjustment, ZonalComponent,
    GeneralIcosahedralSphereMesh, PartitionedCloud, monolithic_solver_parameters
)

moist_thermal_gw_defaults = {
    'ncells_per_edge': 16,         # number of cells per icosahedron edge
    'dt': 900.0,                   # 15 minutes
    'tmax': 5.*24.*60.*60.,        # 5 days
    'dumpfreq': 96,                # dump once per day
    'dirname': 'moist_thermal_gw',
    'equivb': False
}


def moist_thermal_gw(
        ncells_per_edge=moist_thermal_gw_defaults['ncells_per_edge'],
        dt=moist_thermal_gw_defaults['dt'],
        tmax=moist_thermal_gw_defaults['tmax'],
        dumpfreq=moist_thermal_gw_defaults['dumpfreq'],
        dirname=moist_thermal_gw_defaults['dirname'],
        equivb=moist_thermal_gw_defaults['equivb']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    radius = 6371220.           # planetary radius (m)
    q0 = 0.0115                 # saturation curve coefficient (kg/kg)
    beta2 = 9.80616*10          # thermal feedback coefficient (m/s^2)
    nu = 1.5                    # dimensionless parameter in saturation curve
    R0 = pi/9.                  # radius of perturbation (rad)
    lamda_c = -pi/2.            # longitudinal centre of perturbation (rad)
    phi_c = pi/6.               # latitudinal centre of perturbation (rad)
    phi_0 = 3.0e4               # scale factor for poleward buoyancy gradient
    epsilon = 1/300             # linear air expansion coeff (1/K)
    u_max = 20.                 # max amplitude of the zonal wind (m/s)
    g = 9.80616                 # acceleration due to gravity (m/s^2)
    mean_depth = phi_0/g        # reference depth (m)

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = GeneralIcosahedralSphereMesh(radius, ncells_per_edge, degree=2)
    degree = 1
    domain = Domain(mesh, dt, "BDM", degree)
    xyz = SpatialCoordinate(mesh)

    # Equation parameters
    parameters = ShallowWaterParameters(
        mesh, H=mean_depth, q0=q0, nu=nu, beta2=beta2
    )

    # Equation
    tracers = None if equivb else [WaterVapour(space='DG'), CloudWater(space='DG')]
    eqns = ThermalShallowWaterEquations(
        domain, parameters, active_tracers=tracers, equivalent_buoyancy=equivb
    )

    # IO
    if dirname == moist_thermal_gw_defaults['dirname'] and equivb:
        dirname += '_equivb'

    if equivb:
        dumplist = ['b_e', 'D', 'q_t']
        diagnostic_fields = [
            ZonalComponent('u'), MeridionalComponent('u'),
            PartitionedCloud(eqns)
        ]
    else:
        diagnostic_fields = [MeridionalComponent('u'), ZonalComponent('u')]
        dumplist = ['b', 'water_vapour', 'cloud_water', 'D']

    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_nc=True, dump_vtus=False,
        dumplist=dumplist
    )
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport
    transport_methods = [
        DGUpwind(eqns, field_name) for field_name in eqns.field_names
    ]
    subcycling_opts = SubcyclingOptions(subcycle_by_courant=0.25)
    transported_fields = [
        SSPRK3(domain, "u", subcycling_options=subcycling_opts),
        SSPRK3(
            domain, "D", subcycling_options=subcycling_opts,
            rk_formulation=RungeKuttaFormulation.linear
        )
    ]
    if equivb:
        transported_fields += [
            SSPRK3(domain, "b_e", subcycling_options=subcycling_opts),
            SSPRK3(domain, "q_t", subcycling_options=subcycling_opts),
        ]
    else:
        transported_fields += [
            SSPRK3(domain, "b", subcycling_options=subcycling_opts),
            SSPRK3(domain, "water_vapour", subcycling_options=subcycling_opts),
            SSPRK3(domain, "cloud_water", subcycling_options=subcycling_opts)
        ]

    if equivb:
        tau_values = {'D': 1.0, 'b': 1.0}
        solver_parameters = None
        solver_prognostics = ['u', 'D', 'b_e']
    else:
        tau_values = {'D': 1.0, 'b': 1.0, 'water_vapour': 1.0, 'cloud_water': 1.0}
        solver_parameters = monolithic_solver_parameters()
        solver_prognostics = eqns.field_names

    if equivb:
        physics_schemes = None
    else:
        def sat_func(x_in):
            D = x_in.subfunctions[1]
            b = x_in.subfunctions[2]
            q_v = x_in.subfunctions[3]
            b_e = b - beta2*q_v
            sat = q0*mean_depth/D * exp(nu*(1-b_e/g))
            return sat

        # Physics schemes
        sat_adj = SWSaturationAdjustment(
            eqns, sat_func, time_varying_saturation=True,
            parameters=parameters, thermal_feedback=True, beta2=beta2
        )

        physics_schemes = [(sat_adj, ForwardEuler(domain))]

    # ------------------------------------------------------------------------ #
    # Timestepper
    # ------------------------------------------------------------------------ #

    stepper = SemiImplicitQuasiNewton(
        eqns, io, transported_fields, transport_methods,
        tau_values=tau_values, inner_physics_schemes=physics_schemes,
        num_outer=2, num_inner=2, solver_prognostics=solver_prognostics,
        linear_solver_parameters=solver_parameters, reference_update_freq=10800.
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")

    if equivb:
        b0 = stepper.fields("b_e")
        qt0 = stepper.fields("q_t")
    else:
        b0 = stepper.fields("b")
        v0 = stepper.fields("water_vapour")
        c0 = stepper.fields("cloud_water")

    lamda, phi, _ = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])

    # Velocity -- a solid body rotation
    uexpr = xyz_vector_from_lonlatr(u_max*cos(phi), 0, 0, xyz)

    # Buoyancy -- dependent on latitude
    g = parameters.g
    w = parameters.Omega*radius*u_max + (u_max**2)/2
    sigma = w/10
    theta_0 = epsilon*phi_0**2
    numerator = (
        theta_0 + sigma*((cos(phi))**2) * (
            (w + sigma)*(cos(phi))**2 + 2*(phi_0 - w - sigma)
        )
    )
    denominator = (
        phi_0**2 + (w + sigma)**2
        * (sin(phi))**4 - 2*phi_0*(w + sigma)*(sin(phi))**2
    )
    theta = numerator / denominator

    # Depth -- in balance before the addition of a perturbation
    Dbar_expr = mean_depth - (1/g)*(w + sigma)*((sin(phi))**2)

    # Perturbation
    lsq = (lamda - lamda_c)**2
    thsq = (phi - phi_c)**2
    rsq = min_value(R0**2, lsq+thsq)
    r = sqrt(rsq)
    Dpert = 2000 * (1 - r/R0)
    Dexpr = Dbar_expr + Dpert

    # Actual initial buoyancy is specified through equivalent buoyancy
    q_t = 0.03  # Large enough to prevent cloud ever going negative
    bexpr = parameters.g * (1 - theta)  # Find balanced b from theta
    b_init = Function(b0.function_space()).interpolate(bexpr)
    b_e_init = Function(b0.function_space()).interpolate(b_init - beta2*q_t)
    q_v_init = Function(b0.function_space()).interpolate(q_t)

    # Iterate to obtain equivalent buoyancy and saturation water vapour
    # Saturation curve depends on b_e, which depends on saturation curve
    # Use Newton-Raphson method to find an appropriate solution
    n_iterations = 10
    for _ in range(n_iterations):
        q_sat_expr = q0*mean_depth/Dexpr * exp(nu*(1-b_e_init/g))
        dq_sat_dq_v_expr = nu*beta2/g*q_sat_expr
        q_v_init.interpolate(q_v_init - (q_sat_expr - q_v_init)/(dq_sat_dq_v_expr - 1.0))
        b_e_init.interpolate(b_init - beta2*q_v_init)

    # Water vapour set to saturation amount
    vexpr = q0*mean_depth/Dexpr * exp(nu*(1-b_e_init/g))

    if equivb:
        bexpr = b_e_init
    # NB: to directly compare with equivalent buoyancy case, at this point
    # we would set bexpr = b_e_init + beta2*vexpr for the non-equivalent case

    # Cloud is the rest of total liquid that isn't vapour
    cexpr = Constant(q_t) - vexpr

    u0.project(uexpr)
    D0.interpolate(Dexpr)
    b0.interpolate(bexpr)

    if equivb:
        qt0.interpolate(Constant(q_t))
    else:
        v0.interpolate(vexpr)
        c0.interpolate(cexpr)

    # Set reference profiles to initial state
    Dbar = Function(D0.function_space()).interpolate(Dexpr)
    bbar = Function(b0.function_space()).interpolate(bexpr)
    if equivb:
        stepper.set_reference_profiles([('D', Dbar), ('b_e', bbar)])
    else:
        stepper.set_reference_profiles([
            ('D', Dbar), ('b', bbar), ('water_vapour', v0),
            ('cloud_water', c0)
        ])

    # ----------------------------------------------------------------- #
    # Run
    # ----------------------------------------------------------------- #

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
        '--ncells_per_edge',
        help="The number of cells per edge of icosahedron",
        type=int,
        default=moist_thermal_gw_defaults['ncells_per_edge']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=moist_thermal_gw_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=moist_thermal_gw_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=moist_thermal_gw_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=moist_thermal_gw_defaults['dirname']
    )
    parser.add_argument(
        '--equivb',
        help="Use equivalent buoyancy formulation.",
        action='store_true',
        default=moist_thermal_gw_defaults['equivb']
    )
    args, unknown = parser.parse_known_args()

    moist_thermal_gw(**vars(args))
