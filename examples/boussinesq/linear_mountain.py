"""
This example uses the linear Boussinesq equations to solve the vertical
slice gravity wave test case of Skamarock and Klemp, 1994:
``Efficiency and Accuracy of the Klemp-Wilhelmson Time-Splitting Technique'',
MWR.

The degree 1 elements are used, with an explicit RK4 time stepper.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import (
    PeriodicIntervalMesh, Mesh, as_vector, cos, conditional, VectorFunctionSpace, 
    ExtrudedMesh, SpatialCoordinate, Function, pi
)
from gusto import (
    Domain, IO, SpongeLayerParameters, OutputParameters, RK4, DGUpwind, SUPGOptions, Divergence,
    Timestepper, Perturbation, CourantNumber, BoussinesqParameters,
    LinearBoussinesqEquations, boussinesq_hydrostatic_balance
)

skamarock_klemp_linear_bouss_defaults = {
    'ncolumns': 90,
    'nlayers': 35,
    'dt': 0.1,
    'tmax': 9000.,
    'dumpfreq': 1,
    'dirname': 'mountain_linear_bouss'
}


def skamarock_klemp_linear_bouss(
        ncolumns=skamarock_klemp_linear_bouss_defaults['ncolumns'],
        nlayers=skamarock_klemp_linear_bouss_defaults['nlayers'],
        dt=skamarock_klemp_linear_bouss_defaults['dt'],
        tmax=skamarock_klemp_linear_bouss_defaults['tmax'],
        dumpfreq=skamarock_klemp_linear_bouss_defaults['dumpfreq'],
        dirname=skamarock_klemp_linear_bouss_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Test case parameters
    # ------------------------------------------------------------------------ #

    domain_width = 144000      # Width of domain (m)
    domain_height = 35000    # Height of domain (m)
    N = 0.01                  # Brunt-Vaisala frequency (1/s)
    cs = 300.                 # Speed of sound (m/s)
    a = 1000.                 # scale width of mountain, in m
    hm = 1.                   # height of mountain, in m
    zh = 5000.                # height at which mesh is no longer distorted, in m
    sponge_depth = 10000.0    # depth of sponge layer, in m
    mu_dt = 0.15              # parameter for strength of sponge layer, no units
    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    element_order = 1

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
    xexpr = as_vector(
        [x, conditional(z < zh, z + cos(0.5 * pi * z / zh)**6 * zs, z)]
    )

    # Make new mesh
    new_coords = Function(Vc).interpolate(xexpr)
    mesh = Mesh(new_coords)
    mesh._base_mesh = base_mesh  # Force new mesh to inherit original base mesh
    domain = Domain(mesh, dt, "CG", element_order)

    # Equation
    parameters = BoussinesqParameters(cs=cs)
    sponge = SpongeLayerParameters(
        H=domain_height, z_level=domain_height-sponge_depth, mubar=mu_dt/dt
    )
    eqns = LinearBoussinesqEquations(domain, parameters, sponge_options=sponge)

    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_vtus=True, dump_nc=True,
    )
    # list of diagnostic fields, each defined in a class in diagnostics.py
    diagnostic_fields = [CourantNumber(), Divergence(), Perturbation('b')]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    b_opts = SUPGOptions()
    transport_methods = [
        DGUpwind(eqns, "p"),
        DGUpwind(eqns, "b", ibp=b_opts.ibp)
    ]

    # Time stepper
    stepper = Timestepper(
        eqns, RK4(domain), io, spatial_methods=transport_methods
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    b0 = stepper.fields("b")
    p0 = stepper.fields("p")

    # spaces
    Vb = b0.function_space()
    Vp = p0.function_space()

    x, z = SpatialCoordinate(mesh)

    # first setup the background buoyancy profile
    # z.grad(bref) = N**2
    bref = z*(N**2)
    # interpolate the expression to the function
    b_b = Function(Vb).interpolate(bref)

    p_b = Function(Vp)
    boussinesq_hydrostatic_balance(eqns, b_b, p_b)
    p0.assign(p_b)

    # set the background buoyancy
    stepper.set_reference_profiles([('p', p_b), ('b', b_b)])

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
        default=skamarock_klemp_linear_bouss_defaults['ncolumns']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=skamarock_klemp_linear_bouss_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=skamarock_klemp_linear_bouss_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=skamarock_klemp_linear_bouss_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=skamarock_klemp_linear_bouss_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=skamarock_klemp_linear_bouss_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    skamarock_klemp_linear_bouss(**vars(args))
