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
    TrapeziumRule, ShallowWaterParameters, ShallowWaterEquations,
    lonlatr_from_xyz, GeneralIcosahedralSphereMesh
)

williamson_5_defaults = {
    'ncells_per_edge': 16,     # number of cells per icosahedron edge
    'dt': 600.0,               # 10 minutes
    'tmax': 50.*24.*60.*60.,   # 50 days
    'dumpfreq': 144,           # once per day with default options
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
    # Test case parameters
    # ------------------------------------------------------------------------ #

    day = 24.*60.*60.
    if '--running-tests' in sys.argv:
        ref_dt = {3: 3000.}
        tmax = 3000.
        ndumps = 1
    else:
        # setup resolution and timestepping parameters for convergence test
        ref_dt = {3: 900., 4: 450., 5: 225., 6: 112.5}
        tmax = 50*day
        ndumps = 5

    # setup shallow water parameters
    R = 6371220.
    H = 5960.

    # setup input that doesn't change with ref level or dt
    parameters = ShallowWaterParameters(H=H)

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = GeneralIcosahedralSphereMesh(radius, ncells_per_edge, degree=2)
    x = SpatialCoordinate(mesh)
    domain = Domain(mesh, dt, 'BDM', 1)

    # Equation
    Omega = parameters.Omega
    fexpr = 2*Omega*x[2]/R
    lamda, theta, _ = lonlatr_from_xyz(x[0], x[1], x[2])
    R0 = pi/9.
    R0sq = R0**2
    lamda_c = -pi/2.
    lsq = (lamda - lamda_c)**2
    theta_c = pi/6.
    thsq = (theta - theta_c)**2
    rsq = min_value(R0sq, lsq+thsq)
    r = sqrt(rsq)
    bexpr = 2000 * (1 - r/R0)
    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, bexpr=bexpr)

    # I/O
    output = OutputParameters(
        dirname=dirname,
        dumplist_latlon=['D'],
        dumpfreq=dumpfreq,
    )
    diagnostic_fields = [Sum('D', 'topography')]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    transported_fields = [TrapeziumRule(domain, "u"),
                          SSPRK3(domain, "D")]
    transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]

    # Time stepper
    stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields, transport_methods)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields('u')
    D0 = stepper.fields('D')
    u_max = 20.   # Maximum amplitude of the zonal wind (m/s)
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    g = parameters.g
    Rsq = R**2
    Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g - bexpr

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
