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
    TrapeziumRule, ShallowWaterParameters, ShallowWaterEquations,
    RelativeVorticity, PotentialVorticity, SteadyStateError,
    ShallowWaterKineticEnergy, ShallowWaterPotentialEnergy,
    ShallowWaterPotentialEnstrophy, lonlatr_from_xyz, ThermalSWSolver,
    GeneralIcosahedralSphereMesh
)

thermal_williamson_2_defaults = {
    'ncells_per_edge': 16,     # number of cells per icosahedron edge
    'dt': 1800.0,              # 30 minutes
    'tmax': 5.*24.*60.*60.,    # 5 days
    'dumpfreq': 48,            # once per day with default options
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
    # Test case parameters
    # ------------------------------------------------------------------------ #

    dt = 4000

    if '--running-tests' in sys.argv:
        tmax = dt
        dumpfreq = 1
    else:
        day = 24*60*60
        tmax = 5*day
        ndumps = 5
        dumpfreq = int(tmax / (ndumps*dt))

    R = 6371220.
    u_max = 20
    phi_0 = 3e4
    epsilon = 1/300
    theta_0 = epsilon*phi_0**2
    g = 9.80616
    H = phi_0/g

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = GeneralIcosahedralSphereMesh(radius, ncells_per_edge, degree=2)
    degree = 1
    domain = Domain(mesh, dt, 'BDM', degree)
    x = SpatialCoordinate(mesh)

    # Equations
    params = ShallowWaterParameters(H=H, g=g)
    Omega = params.Omega
    fexpr = 2*Omega*x[2]/R
    eqns = ShallowWaterEquations(domain, params, fexpr=fexpr, u_transport_option='vector_advection_form', thermal=True)

    # IO
    output = OutputParameters(
        dirname=dirname,
        dumpfreq=dumpfreq,
        dumplist_latlon=['D', 'D_error'],
    )

    diagnostic_fields = [RelativeVorticity(), PotentialVorticity(),
                        ShallowWaterKineticEnergy(),
                        ShallowWaterPotentialEnergy(params),
                        ShallowWaterPotentialEnstrophy(),
                        SteadyStateError('u'), SteadyStateError('D'),
                        SteadyStateError('b'), MeridionalComponent('u'),
                        ZonalComponent('u')]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    transported_fields = [TrapeziumRule(domain, "u"),
                        SSPRK3(domain, "D", fixed_subcycles=2),
                        SSPRK3(domain, "b", fixed_subcycles=2)]
    transport_methods = [DGUpwind(eqns, "u"),
                        DGUpwind(eqns, "D"),
                        DGUpwind(eqns, "b")]

    # Linear solver
    linear_solver = ThermalSWSolver(eqns)

    # Time stepper
    stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                    transport_methods,
                                    linear_solver=linear_solver)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")
    b0 = stepper.fields("b")

    lamda, phi, _ = lonlatr_from_xyz(x[0], x[1], x[2])

    uexpr = xyz_vector_from_lonlatr(u_max*cos(phi), 0, 0, x)
    g = params.g
    w = Omega*R*u_max + (u_max**2)/2
    sigma = w/10

    Dexpr = H - (1/g)*(w + sigma)*((sin(phi))**2)

    numerator = theta_0 + sigma*((cos(phi))**2) * ((w + sigma)*(cos(phi))**2 + 2*(phi_0 - w - sigma))

    denominator = phi_0**2 + (w + sigma)**2*(sin(phi))**4 - 2*phi_0*(w + sigma)*(sin(phi))**2

    theta = numerator/denominator

    bexpr = params.g * (1 - theta)

    u0.project(uexpr)
    D0.interpolate(Dexpr)
    b0.interpolate(bexpr)

    # Set reference profiles
    Dbar = Function(D0.function_space()).assign(H)
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
