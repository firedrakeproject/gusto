"""
A linearised form of Test Case 2 (solid-body rotation) of Williamson et al 1992:
``A standard test set for numerical approximations to the shallow water
equations in spherical geometry'', JCP.

This uses an icosahedral mesh of the sphere, and the linear shallow water
equations.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import Function, SpatialCoordinate, as_vector, pi
from gusto import (
    Domain, IO, OutputParameters, SemiImplicitQuasiNewton, DefaultTransport,
    ForwardEuler, SteadyStateError, ShallowWaterParameters,
    LinearShallowWaterEquations, GeneralIcosahedralSphereMesh,
    ZonalComponent, MeridionalComponent, RelativeVorticity
)

linear_williamson_2_defaults = {
    'ncells_per_edge': 16,     # number of cells per icosahedron edge
    'dt': 900.0,               # 15 minutes
    'tmax': 5.*24.*60.*60.,    # 5 days
    'dumpfreq': 96,            # once per day with default options
    'dirname': 'linear_williamson_2'
}


def linear_williamson_2(
        ncells_per_edge=linear_williamson_2_defaults['ncells_per_edge'],
        dt=linear_williamson_2_defaults['dt'],
        tmax=linear_williamson_2_defaults['tmax'],
        dumpfreq=linear_williamson_2_defaults['dumpfreq'],
        dirname=linear_williamson_2_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    radius = 6371220.                  # planetary radius (m)
    mean_depth = 2000.                 # reference depth (m)
    u_max = 2*pi*radius/(12*24*60*60)  # Max amplitude of the zonal wind (m/s)

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    element_order = 1

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = GeneralIcosahedralSphereMesh(radius, ncells_per_edge, degree=2)
    x, y, z = SpatialCoordinate(mesh)
    domain = Domain(mesh, dt, 'BDM', element_order)

    # Equation
    parameters = ShallowWaterParameters(mesh, H=mean_depth)
    Omega = parameters.Omega
    fexpr = 2*Omega*z/radius
    eqns = LinearShallowWaterEquations(domain, parameters, fexpr=fexpr)

    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_nc=False, dump_vtus=True
    )
    diagnostic_fields = [SteadyStateError('u'), SteadyStateError('D'),
                         ZonalComponent('u'), MeridionalComponent('u'),
                         RelativeVorticity()]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    transport_schemes = [ForwardEuler(domain, "D")]
    transport_methods = [DefaultTransport(eqns, "D")]

    # Time stepper
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transport_schemes, transport_methods
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    g = parameters.g

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")

    uexpr = as_vector([-u_max*y/radius, u_max*x/radius, 0.0])
    Dexpr = - ((radius*Omega*u_max) * (z/radius)**2) / g

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
        default=linear_williamson_2_defaults['ncells_per_edge']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=linear_williamson_2_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=linear_williamson_2_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=linear_williamson_2_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=linear_williamson_2_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    linear_williamson_2(**vars(args))
