"""
Test Case 2 (solid-body rotation with geostrophically-balanced flow) of
Williamson et al, 1992:
``A standard test set for numerical approximations to the shallow water
equations in spherical geometry'', JCP.

The example here uses the icosahedral sphere mesh and degree 1 spaces.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import SpatialCoordinate, sin, cos, pi, Function
from gusto import (
    Domain, IO, OutputParameters, SemiImplicitQuasiNewton, SSPRK3, DGUpwind,
    TrapeziumRule, ShallowWaterParameters, ShallowWaterEquations,
    RelativeVorticity, PotentialVorticity, SteadyStateError,
    ShallowWaterKineticEnergy, ShallowWaterPotentialEnergy,
    ShallowWaterPotentialEnstrophy, rotated_lonlatr_coords,
    rotated_lonlatr_vectors, GeneralIcosahedralSphereMesh
)

williamson_2_defaults = {
    'ncells_per_edge': 16,     # number of cells per icosahedron edge
    'dt': 1800.0,              # 30 minutes
    'tmax': 5.*24.*60.*60.,    # 5 days
    'dumpfreq': 48,            # once per day with default options
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
    # Test case parameters
    # ------------------------------------------------------------------------ #

    day = 24.*60.*60.
    if '--running-tests' in sys.argv:
        ref_dt = {3: 3000.}
        tmax = 3000.
        ndumps = 1
    else:
        # setup resolution and timestepping parameters for convergence test
        ref_dt = {3: 4000., 4: 2000., 5: 1000., 6: 500.}
        tmax = 5*day
        ndumps = 5

    # setup shallow water parameters
    R = 6371220.
    H = 5960.
    rotated_pole = (0.0, pi/3)

    # setup input that doesn't change with ref level or dt
    parameters = ShallowWaterParameters(H=H)

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = GeneralIcosahedralSphereMesh(radius, ncells_per_edge, degree=2)
    x = SpatialCoordinate(mesh)
    domain = Domain(mesh, dt, 'BDM', 1, rotated_pole=rotated_pole)

    # Equation
    Omega = parameters.Omega
    _, lat, _ = rotated_lonlatr_coords(x, rotated_pole)
    e_lon, _, _ = rotated_lonlatr_vectors(x, rotated_pole)
    fexpr = 2*Omega*sin(lat)
    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr)

    # I/O
    output = OutputParameters(
        dirname=dirname,
        dumpfreq=dumpfreq,
        dumplist_latlon=['D', 'D_error'],
        dump_nc=True,
    )

    diagnostic_fields = [RelativeVorticity(), SteadyStateError('RelativeVorticity'),
                         PotentialVorticity(),
                         ShallowWaterKineticEnergy(),
                         ShallowWaterPotentialEnergy(parameters),
                         ShallowWaterPotentialEnstrophy(),
                         SteadyStateError('u'), SteadyStateError('D'),
                         MeridionalComponent('u', rotated_pole),
                         ZonalComponent('u', rotated_pole)]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    transported_fields = [TrapeziumRule(domain, "u"),
                          SSPRK3(domain, "D", fixed_subcycles=2)]
    transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]

    # Time stepper
    stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields, transport_methods)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")
    x = SpatialCoordinate(mesh)
    u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
    uexpr = u_max*cos(lat)*e_lon
    g = parameters.g
    Dexpr = H - (R * Omega * u_max + u_max*u_max/2.0)*(sin(lat))**2/g

    u0.project(uexpr)
    D0.interpolate(Dexpr)

    Dbar = Function(D0.function_space()).assign(H)
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
