"""
Test Case 5 (flow over a mountain) of Williamson et al, 1992:
``A standard test set for numerical approximations to the shallow water
equations in spherical geometry'', JCP.

The example here uses the icosahedral sphere mesh and degree 1 spaces.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import (
    SpatialCoordinate, as_vector, pi, sqrt, min_value, Function
)
from gusto import (
    Domain, IO, OutputParameters, SemiImplicitQuasiNewton, SSPRK3, DGUpwind,
    ShallowWaterParameters, ShallowWaterEquations, Sum,
    lonlatr_from_xyz, GeneralIcosahedralSphereMesh, ZonalComponent,
    MeridionalComponent, RelativeVorticity, RungeKuttaFormulation,
    SubcyclingOptions
)

williamson_5_defaults = {
    'ncells_per_edge': 16,     # number of cells per icosahedron edge
    'dt': 900.0,               # 15 minutes
    'tmax': 50.*24.*60.*60.,   # 50 days
    'dumpfreq': 960,           # once per 10 days with default options
    'dirname': 'williamson_5'
}


def williamson_5(
        ncells_per_edge=williamson_5_defaults['ncells_per_edge'],
        dt=williamson_5_defaults['dt'],
        tmax=williamson_5_defaults['tmax'],
        dumpfreq=williamson_5_defaults['dumpfreq'],
        dirname=williamson_5_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    radius = 6371220.           # planetary radius (m)
    mean_depth = 5960           # reference depth (m)
    g = 9.80616                 # acceleration due to gravity (m/s^2)
    u_max = 20.                 # max amplitude of the zonal wind (m/s)
    mountain_height = 2000.     # height of mountain (m)
    R0 = pi/9.                  # radius of mountain (rad)
    lamda_c = -pi/2.            # longitudinal centre of mountain (rad)
    phi_c = pi/6.               # latitudinal centre of mountain (rad)

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    element_order = 1

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = GeneralIcosahedralSphereMesh(radius, ncells_per_edge, degree=2)
    domain = Domain(mesh, dt, 'BDM', element_order)
    x, y, z = SpatialCoordinate(mesh)
    lamda, phi, _ = lonlatr_from_xyz(x, y, z)

    # Equation: coriolis
    parameters = ShallowWaterParameters(H=mean_depth, g=g)
    Omega = parameters.Omega
    fexpr = 2*Omega*z/radius

    # Equation: topography
    rsq = min_value(R0**2, (lamda - lamda_c)**2 + (phi - phi_c)**2)
    r = sqrt(rsq)
    tpexpr = mountain_height * (1 - r/R0)
    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, bexpr=tpexpr)

    # I/O
    output = OutputParameters(
        dirname=dirname, dumplist_latlon=['D'], dumpfreq=dumpfreq,
        dump_vtus=True, dump_nc=False, dumplist=['D', 'topography']
    )
    diagnostic_fields = [Sum('D', 'topography'), RelativeVorticity(),
                         MeridionalComponent('u'), ZonalComponent('u')]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    subcycling_options = SubcyclingOptions(subcycle_by_courant=0.25)
    transported_fields = [
        SSPRK3(domain, "u", subcycling_options=subcycling_options),
        SSPRK3(
            domain, "D", subcycling_options=subcycling_options,
            rk_formulation=RungeKuttaFormulation.linear
        )
    ]
    transport_methods = [
        DGUpwind(eqns, "u"),
        DGUpwind(eqns, "D", advective_then_flux=True)
    ]

    # Time stepper
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transported_fields, transport_methods
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields('u')
    D0 = stepper.fields('D')
    uexpr = as_vector([-u_max*y/radius, u_max*x/radius, 0.0])
    Dexpr = (
        mean_depth - tpexpr
        - (radius*Omega*u_max + 0.5*u_max**2)*(z/radius)**2/g
    )

    u0.project(uexpr)
    D0.interpolate(Dexpr)

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
        default=williamson_5_defaults['ncells_per_edge']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=williamson_5_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=williamson_5_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=williamson_5_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=williamson_5_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    williamson_5(**vars(args))
