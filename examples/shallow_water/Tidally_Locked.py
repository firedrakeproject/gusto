"""
For Jemma, in the physics section the commented out evapouration is the problem for me, In the 
Tidally_locked_defaults I set siqn=True to choose the SemiImplicitQuasiNewton timestepper. 

2nd attempt at a tidally locked linear shallow water simulation. Previous version was wrong, but
had correct looking Ts and qS for the I.C.s. This is adapted from linear_williamson2.py from
gusto examples and from my previous version. Equations are from the hydro_circ branch of 
gusto. Files for output are read using ParaView and are found in the results folder.
I need to use explorer.exe . to see the files and copy them into a place that is viewable
"""
from firedrake import SpatialCoordinate, Constant, as_vector, cos, sin, exp, max_value, dot
from firedrake import *
from gusto import ( # pyright: ignore[reportMissingImports]
    Domain, IO, OutputParameters, Timestepper,
    ShallowWaterParameters, lonlatr_from_xyz, WaterVapour,
    GeneralIcosahedralSphereMesh, RelativeVorticity,
    ZonalComponent, MeridionalComponent, LinearShallowWaterEquations, 
    ShallowWaterEquations, DiagnosticField, SemiImplicitQuasiNewton, DefaultTransport,
    ForwardEuler, SteadyStateError, TrapeziumRule, LinearFriction, VerticalVelocity,
    Evaporation, Precipitation, MoistureDescent
)
# These contain all the imports from both previous examples and from the new stuff in the branch

Tidally_locked_defaults = {
    'ncells_per_edge': 16,          # number of cells per icosahedron edge
    'dt': 900.0,                    # 15 minutes
    'tmax': 5.*24.*60.*60.,         # 5 days
    'dumpfreq': 96,                 # number of timesteps before a dump
    'siqn': True,                  # Was in LW2, "default is TrapeziumRule"
    'dirname': 'Tidally_lockedV2R5' # Version 2, Run 1
}


def Tidally_locked(
        ncells_per_edge=Tidally_locked_defaults['ncells_per_edge'],
        dt=Tidally_locked_defaults['dt'],
        tmax=Tidally_locked_defaults['tmax'],
        dumpfreq=Tidally_locked_defaults['dumpfreq'],
        siqn=Tidally_locked_defaults["siqn"],
        dirname=Tidally_locked_defaults['dirname']
):
    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #
    radius = 6.371e6            # R - planetary radius (m)
    mean_depth = 1000           # H or D - reference depth (m)
    g = 9.80616                 # acceleration due to gravity (m/s^2)
    Omega = 1.992385e-8         # rotation rate (rad/s) - once every 10 years
    p_sfc = 100000.0            # Surface pressure (Pa)
    L = 2.5e6                   # latent heat (J/kg)

    # Tidally locked temperature field controls
    T_night = 230.0             # nightside baseline (K)
    T_day = 305.0               # substellar max (K)
    p_hot = 0.8                 # hotspot sharpness exponent
    substellar_lon_deg = 0.0    # where the star is overhead (longitude)
    substellar_lat_deg = 0.0    # where the star is overhead (latitude)

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
    lamda, phi, _ = lonlatr_from_xyz(x, y, z)


    # Equation
    parameters = ShallowWaterParameters(mesh, H=mean_depth, g=g, Omega=Omega)
    active_tracers = [WaterVapour(space='DG')]
    eqns = LinearShallowWaterEquations(domain, parameters,
                                       active_tracers=active_tracers,
                                       u_transport_option=u_eqn_type)

    # EDIT FROM HERE

    # I/O, all copied from LW2, should hopefully work?

    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_nc=False, dump_vtus=True,
        dumplist_latlon=["water_vapour"]
    )
    diagnostic_fields = [SteadyStateError('u'), SteadyStateError('D'),
                         ZonalComponent('u'), MeridionalComponent('u'),
                         RelativeVorticity()]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)


    # ------------------------------------------------------------------------ #
    # PHYSICS
    # ------------------------------------------------------------------------ #
    
    X = SpatialCoordinate(mesh)
    rhat = X / radius
    lon0 = substellar_lon_deg * np.pi/180.0
    lat0 = substellar_lat_deg * np.pi/180.0
    s_hat = as_vector((cos(lat0)*cos(lon0), cos(lat0)*sin(lon0), sin(lat0)))
    mu = dot(rhat, s_hat)
    day = max_value(mu, 0.0)

    TS_expr = T_night + (T_day - T_night) * day**p_hot
    e0 = 2300.0 * exp(L/(293.0*416.0))
    saturation_curve = 0.622 * e0 * exp(-L/(416.0*TS_expr)) / p_sfc




    LinearFriction(eqns)

    VerticalVelocity(eqns)

    #Evaporation(eqns, saturation_curve, wind_dependant=True)

    Precipitation(eqns)

    MoistureDescent(eqns)


    if siqn:
        # Transport schemes
        transport_schemes = [ForwardEuler(domain, "D")]
        transport_methods = [DefaultTransport(eqns, "D")]

        # Time stepper
        stepper = SemiImplicitQuasiNewton(
            eqns, io, transport_schemes, transport_methods
        )
    else:
        stepper = Timestepper(eqns, TrapeziumRule(domain), io)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")
    v0 = stepper.fields("water_vapour")

    RH0 = 0.8
    v0.interpolate(RH0 * saturation_curve)

    # Start with no wind and a flat layer depth
    u0.assign(0.0)
    D0.assign(mean_depth)

    ubar = Function(u0.function_space()).assign(0.0)
    Dbar = Function(D0.function_space()).assign(mean_depth)
    qbar = Function(v0.function_space()).assign(float(parameters.q_ut))

    stepper.set_reference_profiles([("u", ubar), ("D", Dbar), ("water_vapour", qbar)])

    stepper.run(t=0.0, tmax=tmax)

# ---------------------------------------------------------------------------- #
# MAIN
# ---------------------------------------------------------------------------- #


if __name__ == "__main__":

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--ncells_per_edge',
        help="The number of cells per edge of icosahedron",
        type=int,
        default=Tidally_locked_defaults['ncells_per_edge']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=Tidally_locked_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=Tidally_locked_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=Tidally_locked_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=Tidally_locked_defaults['dirname']
    )
    parser.add_argument(
        '--siqn',
        help=(
            "Whether to use the Semi-Implicit Quasi-Newton stepper. Otherwise "
            + "use the Trapezium Rule."
        ),
        action="store_true",
        default=Tidally_locked_defaults['siqn']
    )
    args, unknown = parser.parse_known_args()

    Tidally_locked(**vars(args))
