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
    LinearShallowWaterEquations, GeneralIcosahedralSphereMesh
)

linear_williamson_2_defaults = {
    'ncells_per_edge': 16,     # number of cells per icosahedron edge
    'dt': 1800.0,              # 30 minutes
    'tmax': 5.*24.*60.*60.,    # 5 days
    'dumpfreq': 48,            # once per day with default options
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
    # Test case parameters
    # ------------------------------------------------------------------------ #

    dt = 3600.
    day = 24.*60.*60.
    if '--running-tests' in sys.argv:
        tmax = dt
        dumpfreq = 1
    else:
        tmax = 5*day
        dumpfreq = int(tmax / (5*dt))

    refinements = 3  # number of horizontal cells = 20*(4^refinements)

    R = 6371220.
    H = 2000.

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = GeneralIcosahedralSphereMesh(radius, ncells_per_edge, degree=2)
    x = SpatialCoordinate(mesh)
    domain = Domain(mesh, dt, 'BDM', 1)

    # Equation
    parameters = ShallowWaterParameters(H=H)
    Omega = parameters.Omega
    x = SpatialCoordinate(mesh)
    fexpr = 2*Omega*x[2]/R
    eqns = LinearShallowWaterEquations(domain, parameters, fexpr=fexpr)

    # I/O
    output = OutputParameters(
        dirname=dirname,
        dumpfreq=dumpfreq,
    )
    diagnostic_fields = [SteadyStateError('u'), SteadyStateError('D')]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    transport_schemes = [ForwardEuler(domain, "D")]
    transport_methods = [DefaultTransport(eqns, "D")]

    # Time stepper
    stepper = SemiImplicitQuasiNewton(eqns, io, transport_schemes, transport_methods)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")
    u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    g = parameters.g
    Dexpr = - ((R * Omega * u_max)*(x[2]*x[2]/(R*R)))/g
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
