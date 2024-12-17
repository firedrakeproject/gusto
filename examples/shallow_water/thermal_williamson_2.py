"""
The thermal form of Test Case 2 (solid-body rotation with geostrophically
balanced flow) of Williamson et al, 1992:
``A standard test set for numerical approximations to the shallow water
equations in spherical geometry'', JCP.

The initial conditions are taken from Zerroukat & Allen, 2015:
``A moist Boussinesq shallow water equations set for testing atmospheric
models'', JCP.

The example here uses the icosahedral sphere mesh and degree 1 spaces.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import Function, SpatialCoordinate, sin, cos
from gusto import (
    Domain, IO, OutputParameters, SemiImplicitQuasiNewton, SSPRK3, DGUpwind,
    TrapeziumRule, ShallowWaterParameters, ThermalShallowWaterEquations,
    RelativeVorticity, PotentialVorticity, SteadyStateError,
    ZonalComponent, MeridionalComponent, ThermalSWSolver,
    xyz_vector_from_lonlatr, lonlatr_from_xyz, GeneralIcosahedralSphereMesh,
    SubcyclingOptions
)

thermal_williamson_2_defaults = {
    'ncells_per_edge': 16,     # number of cells per icosahedron edge
    'dt': 900.0,               # 15 minutes
    'tmax': 5.*24.*60.*60.,    # 5 days
    'dumpfreq': 96,            # once per day with default options
    'dirname': 'thermal_williamson_2'
}


def thermal_williamson_2(
        ncells_per_edge=thermal_williamson_2_defaults['ncells_per_edge'],
        dt=thermal_williamson_2_defaults['dt'],
        tmax=thermal_williamson_2_defaults['tmax'],
        dumpfreq=thermal_williamson_2_defaults['dumpfreq'],
        dirname=thermal_williamson_2_defaults['dirname']
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

    # Equations
    params = ShallowWaterParameters(H=mean_depth, g=g)
    Omega = params.Omega
    fexpr = 2*Omega*z/radius
    eqns = ThermalShallowWaterEquations(
        domain, params, fexpr=fexpr, u_transport_option=u_eqn_type
    )

    # IO
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dumplist_latlon=['D', 'D_error'],
        dump_vtus=False, dump_nc=True
    )

    diagnostic_fields = [
        RelativeVorticity(), PotentialVorticity(),
        SteadyStateError('u'), SteadyStateError('D'), SteadyStateError('b'),
        MeridionalComponent('u'), ZonalComponent('u')
    ]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    subcycling_options = SubcyclingOptions(fixed_subcycles=2)
    transported_fields = [
        TrapeziumRule(domain, "u"),
        SSPRK3(domain, "D", subcycling_options=subcycling_options),
        SSPRK3(domain, "b", subcycling_options=subcycling_options)
    ]
    transport_methods = [
        DGUpwind(eqns, "u"),
        DGUpwind(eqns, "D"),
        DGUpwind(eqns, "b")
    ]

    # Linear solver
    linear_solver = ThermalSWSolver(eqns)

    # Time stepper
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transported_fields, transport_methods,
        linear_solver=linear_solver
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")
    b0 = stepper.fields("b")

    _, phi, _ = lonlatr_from_xyz(x, y, z)

    uexpr = xyz_vector_from_lonlatr(u_max*cos(phi), 0, 0, (x, y, z))
    w = Omega*radius*u_max + (u_max**2)/2
    sigma = w/10

    Dexpr = mean_depth - (1/g)*(w + sigma)*((sin(phi))**2)

    numerator = (
        theta_0 + sigma*((cos(phi))**2)
        * ((w + sigma)*(cos(phi))**2 + 2*(phi_0 - w - sigma))
    )
    denominator = (
        phi_0**2 + (w + sigma)**2*(sin(phi))**4
        - 2*phi_0*(w + sigma)*(sin(phi))**2
    )

    theta = numerator/denominator
    bexpr = params.g * (1 - theta)

    u0.project(uexpr)
    D0.interpolate(Dexpr)
    b0.interpolate(bexpr)

    # Set reference profiles
    Dbar = Function(D0.function_space()).assign(mean_depth)
    bbar = Function(b0.function_space()).interpolate(bexpr)
    stepper.set_reference_profiles([('D', Dbar), ('b', bbar)])

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
        default=thermal_williamson_2_defaults['ncells_per_edge']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=thermal_williamson_2_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=thermal_williamson_2_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=thermal_williamson_2_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=thermal_williamson_2_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    thermal_williamson_2(**vars(args))
