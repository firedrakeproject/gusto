"""
Test Case 2 (solid-body rotation with geostrophically-balanced flow) of
Williamson et al, 1992:
``A standard test set for numerical approximations to the shallow water
equations in spherical geometry'', JCP.

The example here uses the icosahedral sphere mesh and degree 1 spaces.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import SpatialCoordinate, sin, cos, pi, Function, Constant
from gusto import (
    Domain, IO, OutputParameters, SemiImplicitQuasiNewton, SSPRK3, DGUpwind,
    TrapeziumRule, ShallowWaterParameters, ShallowWaterEquations,
    RelativeVorticity, PotentialVorticity, SteadyStateError,
    ShallowWaterKineticEnergy, ShallowWaterPotentialEnergy,
    ShallowWaterPotentialEnstrophy, lonlatr_from_xyz,
    ZonalComponent, MeridionalComponent, xyz_vector_from_lonlatr,
    GeneralIcosahedralSphereMesh, SubcyclingOptions
)

williamson_2_defaults = {
    'ncells_per_edge': 16,     # number of cells per icosahedron edge
    'dt': 900.0,               # 15 minutes
    'tmax': 5.*24.*60.*60.,    # 5 days
    'dumpfreq': 96,            # once per day with default options
    'dirname': 'williamson_2'
}


def williamson_2(
        ncells_per_edge=williamson_2_defaults['ncells_per_edge'],
        dt=williamson_2_defaults['dt'],
        tmax=williamson_2_defaults['tmax'],
        dumpfreq=williamson_2_defaults['dumpfreq'],
        dirname=williamson_2_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    radius = 6371220.                  # planetary radius (m)
    mean_depth = 5960.                 # reference depth (m)
    u_max = 2*pi*radius/(12*24*60*60)  # Max amplitude of the zonal wind (m/s)

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    element_order = 1
    u_eqn_type = 'vector_invariant_form'

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = GeneralIcosahedralSphereMesh(radius, ncells_per_edge, degree=2)
    domain = Domain(mesh, dt, 'BDM', element_order)
    xyz = SpatialCoordinate(mesh)

    # Equation
    parameters = ShallowWaterParameters(mesh, H=mean_depth)
    Omega = parameters.Omega
    _, lat, _ = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])
    e_lon = xyz_vector_from_lonlatr(
        Constant(1.0), Constant(0.0), Constant(0.0), xyz
    )

    eqns = ShallowWaterEquations(
        domain, parameters, u_transport_option=u_eqn_type)

    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_nc=True,
        dumplist_latlon=['D', 'D_error'],
    )
    diagnostic_fields = [
        RelativeVorticity(), SteadyStateError('RelativeVorticity'),
        PotentialVorticity(), ShallowWaterKineticEnergy(),
        ShallowWaterPotentialEnergy(parameters),
        ShallowWaterPotentialEnstrophy(),
        SteadyStateError('u'), SteadyStateError('D'),
        MeridionalComponent('u'),
        ZonalComponent('u')
    ]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    subcycling_options = SubcyclingOptions(fixed_subcycles=2)
    transported_fields = [
        TrapeziumRule(domain, "u"),
        SSPRK3(domain, "D", subcycling_options=subcycling_options)]
    transport_methods = [
        DGUpwind(eqns, "u"),
        DGUpwind(eqns, "D")
    ]

    # Time stepper
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transported_fields, transport_methods
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    g = parameters.g

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")

    uexpr = u_max*cos(lat)*e_lon
    Dexpr = mean_depth - (radius * Omega * u_max + 0.5*u_max**2)*(sin(lat))**2/g

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
        default=williamson_2_defaults['ncells_per_edge']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=williamson_2_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=williamson_2_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=williamson_2_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=williamson_2_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    williamson_2(**vars(args))
