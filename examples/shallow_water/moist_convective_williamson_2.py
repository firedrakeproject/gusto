"""
A moist convective form of Test Case 2 (solid-body rotation with flow in
geostrophic balance) of Williamson 2 et al, 1992:
``A standard test set for numerical approximations to the shallow water
equations in spherical geometry'', JCP.

Three moist variables (vapour, cloud liquid and rain) are used. The saturation
function depends on height, with a temporally-constant background buoyancy/
temperature field. Vapour is initialised very close to saturation and
small overshoots in will generate clouds.

This example uses the icosahedral sphere mesh and degree 1 spaces.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import SpatialCoordinate, sin, cos, exp, Function
from gusto import (
    Domain, IO, OutputParameters, SemiImplicitQuasiNewton, SSPRK3, DGUpwind,
    TrapeziumRule, ShallowWaterParameters, ShallowWaterEquations,
    CourantNumber, RelativeVorticity, PotentialVorticity,
    SteadyStateError, ShallowWaterKineticEnergy, ShallowWaterPotentialEnergy,
    ShallowWaterPotentialEnstrophy, lonlatr_from_xyz, DG1Limiter, InstantRain,
    MoistConvectiveSWSolver, ForwardEuler, SWSaturationAdjustment,
    WaterVapour, CloudWater, Rain, GeneralIcosahedralSphereMesh,
    xyz_vector_from_lonlatr
)

moist_convect_williamson_2_defaults = {
    'ncells_per_edge': 16,     # number of cells per icosahedron edge
    'dt': 900.0,               # 15 minutes
    'tmax': 5.*24.*60.*60.,    # 5 days
    'dumpfreq': 96,            # once per day with default options
    'dirname': 'moist_convective_williamson_2'
}


def moist_convect_williamson_2(
        ncells_per_edge=moist_convect_williamson_2_defaults['ncells_per_edge'],
        dt=moist_convect_williamson_2_defaults['dt'],
        tmax=moist_convect_williamson_2_defaults['tmax'],
        dumpfreq=moist_convect_williamson_2_defaults['dumpfreq'],
        dirname=moist_convect_williamson_2_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    radius = 6371220.           # planetary radius (m)
    u_max = 20.                 # max amplitude of the zonal wind (m/s)
    phi_0 = 3.0e4               # reference geopotential height (m^2/s^2)
    epsilon = 1/300             # linear air expansion coeff (1/K)
    theta_0 = epsilon*phi_0**2  # ref depth-integrated temperature (no units)
    g = 9.80616                 # acceleration due to gravity (m/s^2)
    mean_depth = phi_0/g        # reference depth (m)
    xi = 0                      # fraction of excess vapour/cloud not converted
    q0 = 200                    # saturation mixing ratio scaling (kg/kg)
    beta1 = 1600                # depth-vaporisation factor (m)
    gamma_v = 0.98              # vaporisation implicit factor
    qprecip = 1e-4              # cloud to rain conversion threshold (kg/kg)
    gamma_r = 1e-3              # rain-coalescence implicit factor

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    element_order = 1
    u_eqn_type = 'vector_advection_form'

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = GeneralIcosahedralSphereMesh(radius, ncells_per_edge, degree=2)
    domain = Domain(mesh, dt, 'BDM', element_order)
    x, y, z = SpatialCoordinate(mesh)
    _, phi, _ = lonlatr_from_xyz(x, y, z)

    # Equations
    parameters = ShallowWaterParameters(H=mean_depth, g=g)
    Omega = parameters.Omega
    fexpr = 2*Omega*z/radius

    tracers = [
        WaterVapour(space='DG'), CloudWater(space='DG'), Rain(space='DG')
    ]

    eqns = ShallowWaterEquations(
        domain, parameters, fexpr=fexpr, u_transport_option=u_eqn_type,
        active_tracers=tracers
    )

    # IO
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_nc=True, dump_vtus=True,
        dumplist_latlon=['D', 'D_error']
    )
    diagnostic_fields = [
        CourantNumber(), RelativeVorticity(), PotentialVorticity(),
        ShallowWaterKineticEnergy(), ShallowWaterPotentialEnergy(parameters),
        ShallowWaterPotentialEnstrophy(), SteadyStateError('u'),
        SteadyStateError('D'), SteadyStateError('water_vapour'),
        SteadyStateError('cloud_water')
    ]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # define saturation function
    def sat_func(x_in):
        h = x_in.split()[1]
        numerator = (
            theta_0 + sigma*((cos(phi))**2)
            * ((w + sigma)*(cos(phi))**2 + 2*(phi_0 - w - sigma))
        )
        denominator = (
            phi_0**2 + (w + sigma)**2*(sin(phi))**4
            - 2*phi_0*(w + sigma)*(sin(phi))**2
        )
        theta = numerator/denominator
        return q0/(g*h) * exp(20*(theta))

    transport_methods = [DGUpwind(eqns, field_name) for field_name in eqns.field_names]

    limiter = DG1Limiter(domain.spaces('DG'))

    transported_fields = [
        TrapeziumRule(domain, "u"),
        SSPRK3(domain, "D"),
        SSPRK3(domain, "water_vapour", limiter=limiter),
        SSPRK3(domain, "cloud_water", limiter=limiter),
        SSPRK3(domain, "rain", limiter=limiter)
    ]

    linear_solver = MoistConvectiveSWSolver(eqns)

    # Physics schemes
    sat_adj = SWSaturationAdjustment(
        eqns, sat_func, time_varying_saturation=True,
        convective_feedback=True, beta1=beta1, gamma_v=gamma_v,
        time_varying_gamma_v=False, parameters=parameters
    )
    inst_rain = InstantRain(
        eqns, qprecip, vapour_name="cloud_water", rain_name="rain",
        gamma_r=gamma_r
    )

    physics_schemes = [
        (sat_adj, ForwardEuler(domain)), (inst_rain, ForwardEuler(domain))
    ]

    stepper = SemiImplicitQuasiNewton(
        eqns, io, transport_schemes=transported_fields,
        spatial_methods=transport_methods, linear_solver=linear_solver,
        physics_schemes=physics_schemes
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")
    v0 = stepper.fields("water_vapour")

    uexpr = xyz_vector_from_lonlatr(u_max*cos(phi), 0, 0, (x, y, z))
    g = parameters.g
    w = Omega*radius*u_max + (u_max**2)/2
    sigma = 0

    Dexpr = mean_depth - (1/g)*(w)*((sin(phi))**2)

    # though this set-up has no buoyancy, we use the expression for theta to
    # set up the initial vapour
    numerator = (
        theta_0 + sigma*((cos(phi))**2)
        * ((w + sigma)*(cos(phi))**2 + 2*(phi_0 - w - sigma))
    )
    denominator = (
        phi_0**2 + (w + sigma)**2*(sin(phi))**4
        - 2*phi_0*(w + sigma)*(sin(phi))**2
    )
    theta = numerator/denominator

    initial_msat = q0/(g*Dexpr) * exp(20*theta)
    vexpr = (1 - xi) * initial_msat

    u0.project(uexpr)
    D0.interpolate(Dexpr)
    v0.interpolate(vexpr)

    # Set reference profiles
    Dbar = Function(D0.function_space()).assign(mean_depth)
    stepper.set_reference_profiles([('D', Dbar)])

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
        '--ncells_per_edge',
        help="The number of cells per edge of icosahedron",
        type=int,
        default=moist_convect_williamson_2_defaults['ncells_per_edge']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=moist_convect_williamson_2_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=moist_convect_williamson_2_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=moist_convect_williamson_2_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=moist_convect_williamson_2_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    moist_convect_williamson_2(**vars(args))
