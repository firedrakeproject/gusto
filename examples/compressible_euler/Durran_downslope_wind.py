"""
The 1 metre high mountain test case from Melvin et al, 2010:
``An inherently mass-conserving iterative semi-implicit semi-Lagrangian
discretization of the non-hydrostatic vertical-slice equations.'', QJRMS.

This test describes a wave over a 1m high mountain. The domain is smaller than
that in the "non-hydrostatic mountain" case, so the solutions between the
hydrostatic and non-hydrostatic equations should be different.

The setup used here uses the order 1 finite elements.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from firedrake import (
    as_vector, VectorFunctionSpace, PeriodicIntervalMesh, ExtrudedMesh,
    SpatialCoordinate, exp, pi, cos, Function, conditional, Mesh, Constant,
    VTKFile
)
from gusto import (
    Domain, CompressibleParameters, logger, RungeKuttaFormulation,
    CompressibleEulerEquations, HydrostaticCompressibleEulerEquations,
    OutputParameters, IO, SSPRK3, DGUpwind, SemiImplicitQuasiNewton,
    compressible_hydrostatic_balance, SpongeLayerParameters, Exner, ZComponent,
    Perturbation, CourantNumber, MaxKernel, MinKernel, EmbeddedDGOptions,
    SubcyclingOptions
)

mountain_nonhydrostatic_defaults = {
    'ncolumns': 70,
    'nlayers': 20,
    'dt': 10.0,
    'tmax': 10000.,
    'dumpfreq': 1,
    'dirname': 'lateral_sponge_mountain',
    'hydrostatic': False
}


def mountain_nonhydrostatic(
        ncolumns=mountain_nonhydrostatic_defaults['ncolumns'],
        nlayers=mountain_nonhydrostatic_defaults['nlayers'],
        dt=mountain_nonhydrostatic_defaults['dt'],
        tmax=mountain_nonhydrostatic_defaults['tmax'],
        dumpfreq=mountain_nonhydrostatic_defaults['dumpfreq'],
        dirname=mountain_nonhydrostatic_defaults['dirname'],
        hydrostatic=mountain_nonhydrostatic_defaults['hydrostatic']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    domain_width = 210000.   # width of domain in x direction, in m
    domain_height = 9000.   # height of model top, in m
    a = 10000.                # scale width of mountain, in m
    hm = 600.                  # height of mountain, in m
    zh = 2500.               # height at which mesh is no longer distorted, in m
    Tsurf = 300.             # temperature of surface, in K
    initial_wind = 20.0      # initial horizontal wind, in m/s
    sponge_depth = 2000.0   # depth of sponge layer, in m
    sponge_width = 12000.0   # width of sponge layer, in m
    g = 9.80665              # acceleration due to gravity, in m/s^2
    cp = 1004.               # specific heat capacity at constant pressure
    mu_dt = 0.15             # parameter for strength of sponge layer, no units
    exner_surf = 1.0         # maximum value of Exner pressure at surface
    max_iterations = 10      # maximum number of hydrostatic balance iterations
    tolerance = 1e-7         # tolerance for hydrostatic balance iteration

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    element_order = 1
    u_eqn_type = 'vector_advection_form'
    alpha = 0.55  # Necessary to absorb grid scale waves

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    # Make normal extruded mesh which will be distorted to describe the mountain
    base_mesh = PeriodicIntervalMesh(ncolumns, domain_width)
    ext_mesh = ExtrudedMesh(
        base_mesh, layers=nlayers, layer_height=domain_height/nlayers
    )
    Vc = VectorFunctionSpace(ext_mesh, "DG", 2)

    # Describe the mountain
    xc = domain_width/2.
    x, z = SpatialCoordinate(ext_mesh)
    zs = hm * a**2 / ((x - xc)**2 + a**2)
    flatten_mesh = False
    if flatten_mesh:
        xexpr = as_vector(
            [x, conditional(z < zh, z + cos(0.5 * pi * z / zh)**6 * zs, z)]
        )
    else:
        xexpr = as_vector(
            [x, z + ((domain_height - z) / domain_height) * zs]
        )

    # Make new mesh
    new_coords = Function(Vc).interpolate(xexpr)
    mesh = Mesh(new_coords)
    mesh._base_mesh = base_mesh  # Force new mesh to inherit original base mesh
    domain = Domain(mesh, dt, "CG", element_order)

    # Equation
    parameters = CompressibleParameters(mesh, g=g, cp=cp)
    sponge = SpongeLayerParameters(
        mesh, H=domain_height, z_level=domain_height-sponge_depth,
        L=domain_width, x_level=domain_width-sponge_width,
        mubar=mu_dt/dt
    )
    if hydrostatic:
        eqns = HydrostaticCompressibleEulerEquations(
            domain, parameters, sponge_options=sponge,
            u_transport_option=u_eqn_type
        )
    else:
        eqns = CompressibleEulerEquations(
            domain, parameters, sponge_options=sponge,
            u_transport_option=u_eqn_type
        )

    sponge_out = VTKFile("sponge.pvd").write(eqns.prescribed_fields.sponge_vert,
                                             eqns.prescribed_fields.sponge_horiz)
    # I/O
    # Adjust default directory name
    if hydrostatic and dirname == mountain_nonhydrostatic_defaults['dirname']:
        dirname = f'hyd_switch_{dirname}'

    output = OutputParameters(
            dirname=dirname, dumpfreq=dumpfreq, dump_vtus=True, dump_nc=True
    )
    diagnostic_fields = [
        Exner(parameters), ZComponent('u'), Perturbation('theta'), CourantNumber()
    ]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    subcycling_opts = SubcyclingOptions(subcycle_by_courant=0.25)

    # Transport schemes
    theta_opts = EmbeddedDGOptions()
    transported_fields = [
        SSPRK3(domain, "u", subcycling_options=subcycling_opts),
        SSPRK3(domain, "rho", rk_formulation=RungeKuttaFormulation.linear,
               subcycling_options=subcycling_opts),
        SSPRK3(domain, "theta", options=theta_opts,
               subcycling_options=subcycling_opts)
    ]
    transport_methods = [
        DGUpwind(eqns, "u"),
        DGUpwind(eqns, "rho", advective_then_flux=True),
        DGUpwind(eqns, "theta")
    ]

    # Time stepper
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transported_fields, transport_methods, predictor='rho',
        alpha=alpha, tau_values={'rho': 1.0, 'theta': 1.0}
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")

    # spaces
    Vt = domain.spaces("theta")
    Vr = domain.spaces("DG")

    # Thermodynamic constants required for setting initial conditions
    # and reference profiles
    N = parameters.N

    # N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
    x, z = SpatialCoordinate(mesh)
    N1 = 0.02
    N2 = 0.01
    z_interface = 3142.
    theta_lower = Tsurf * exp((N1**2 / g) * z)
    theta_interface = Tsurf * exp((N1**2 / g) * z_interface)
    theta_upper = theta_interface * exp((N2**2 /g) * (z - z_interface))
    thetab = conditional(z < z_interface, theta_lower, theta_upper)
    theta_b = Function(Vt).interpolate(thetab)

    # Calculate hydrostatic exner
    exner = Function(Vr)
    rho_b = Function(Vr)

    # Set up kernels to evaluate global minima and maxima of fields
    min_kernel = MinKernel()
    max_kernel = MaxKernel()

    # First solve hydrostatic balance that gives Exner = 1 at bottom boundary
    # This gives us a guess for the top boundary condition
    bottom_boundary = Constant(exner_surf)
    logger.info(f'Solving hydrostatic with bottom Exner of {exner_surf}')
    compressible_hydrostatic_balance(
        eqns, theta_b, rho_b, exner, top=False, exner_boundary=bottom_boundary
    )

    # Solve hydrostatic balance again, but now use minimum value from first
    # solve as the *top* boundary condition for Exner
    top_value = min_kernel.apply(exner)
    top_boundary = Constant(top_value)
    logger.info(f'Solving hydrostatic with top Exner of {top_value}')
    compressible_hydrostatic_balance(
        eqns, theta_b, rho_b, exner, top=True, exner_boundary=top_boundary
    )

    max_bottom_value = max_kernel.apply(exner)

    # Now we iterate, adjusting the top boundary condition, until this gives
    # a maximum value of 1.0 at the surface
    lower_top_guess = 0.9*top_value
    upper_top_guess = 1.2*top_value
    for i in range(max_iterations):
        # If max bottom Exner value is equal to desired value, stop iteration
        if abs(max_bottom_value - exner_surf) < tolerance:
            break

        # Make new guess by average of previous guesses
        top_guess = 0.5*(lower_top_guess + upper_top_guess)
        top_boundary.assign(top_guess)

        logger.info(
            f'Solving hydrostatic balance iteration {i}, with top Exner value '
            + f'of {top_guess}'
        )

        compressible_hydrostatic_balance(
            eqns, theta_b, rho_b, exner, top=True, exner_boundary=top_boundary
        )

        max_bottom_value = max_kernel.apply(exner)

        # Adjust guesses based on new value
        if max_bottom_value < exner_surf:
            lower_top_guess = top_guess
        else:
            upper_top_guess = top_guess

    logger.info(f'Final max bottom Exner value of {max_bottom_value}')

    # Perform a final solve to obtain hydrostatically balanced rho
    compressible_hydrostatic_balance(
        eqns, theta_b, rho_b, exner, top=True, exner_boundary=top_boundary,
        solve_for_rho=True
    )

    theta0.assign(theta_b)
    rho0.assign(rho_b)
    u0.project(as_vector([initial_wind, 0.0]), bcs=eqns.bcs['u'])

    stepper.set_reference_profiles([('rho', rho_b), ('theta', theta_b)])

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
        '--ncolumns',
        help="The number of columns in the vertical slice mesh.",
        type=int,
        default=mountain_nonhydrostatic_defaults['ncolumns']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=mountain_nonhydrostatic_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=mountain_nonhydrostatic_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=mountain_nonhydrostatic_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=mountain_nonhydrostatic_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=mountain_nonhydrostatic_defaults['dirname']
    )
    parser.add_argument(
        '--hydrostatic',
        help=(
            "Whether to use the hydrostatic switch to emulate the "
            + "hydrostatic equations. Otherwise use the full non-hydrostatic"
            + "equations."
        ),
        action="store_true",
        default=mountain_nonhydrostatic_defaults['hydrostatic']
    )
    args, unknown = parser.parse_known_args()

    mountain_nonhydrostatic(**vars(args))
