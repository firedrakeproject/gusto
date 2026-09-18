"""
This example uses the incompressible Boussinesq equations to solve the vertical
slice gravity wave test case of Skamarock and Klemp, 1994:
``Efficiency and Accuracy of the Klemp-Wilhelmson Time-Splitting Technique'',
MWR.

Buoyancy is transported using SUPG, and either degree 0 or degree 1 elements
may be used. The `ncolumns` and `nlayers` arguments always specify the grid
size for the degree 0 configuration; when degree 1 elements are used, the grid
is halved in each direction so that the total number of degrees of freedom is
kept (approximately) constant between the two configurations.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
from firedrake import (
    as_vector, PeriodicIntervalMesh, ExtrudedMesh, sin, SpatialCoordinate,
    Function, pi
)
from gusto import (
    Domain, IO, OutputParameters, SemiImplicitQuasiNewton, SSPRK3, DGUpwind,
    TrapeziumRule, SUPGOptions, Divergence, Perturbation, CourantNumber,
    BoussinesqParameters, BoussinesqEquations, boussinesq_hydrostatic_balance,
    initial_buoyancy_field_degree1_from_degree0
)

skamarock_klemp_incompressible_bouss_defaults = {
    'ncolumns': 300,
    'nlayers': 10,
    'dt': 6.0,
    'tmax': 3600.,
    'dumpfreq': 300,
    'element_order': 1,
    'dirname': 'skamarock_klemp_incompressible_bouss'
}


def skamarock_klemp_incompressible_bouss(
        ncolumns=skamarock_klemp_incompressible_bouss_defaults['ncolumns'],
        nlayers=skamarock_klemp_incompressible_bouss_defaults['nlayers'],
        dt=skamarock_klemp_incompressible_bouss_defaults['dt'],
        tmax=skamarock_klemp_incompressible_bouss_defaults['tmax'],
        dumpfreq=skamarock_klemp_incompressible_bouss_defaults['dumpfreq'],
        element_order=skamarock_klemp_incompressible_bouss_defaults['element_order'],
        dirname=skamarock_klemp_incompressible_bouss_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Test case parameters
    # ------------------------------------------------------------------------ #

    domain_width = 3.0e5      # Width of domain (m)
    domain_height = 1.0e4     # Height of domain (m)
    wind_initial = 20.        # Initial wind in x direction (m/s)
    pert_width = 5.0e3        # Width parameter of perturbation (m)
    deltab = 1.0e-2           # Magnitude of buoyancy perturbation (m/s^2)
    N = 0.01                  # Brunt-Vaisala frequency (1/s)

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    # ncolumns/nlayers always specify the degree 0 grid. Halve the grid for
    # degree 1 elements so the number of degrees of freedom stays constant.
    if element_order == 1:
        ncolumns = ncolumns // 2
        nlayers = nlayers // 2

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    base_mesh = PeriodicIntervalMesh(ncolumns, domain_width)
    mesh = ExtrudedMesh(base_mesh, nlayers, layer_height=domain_height/nlayers)
    domain = Domain(mesh, dt, 'CG', element_order)

    # Equation
    parameters = BoussinesqParameters(mesh)
    eqns = BoussinesqEquations(domain, parameters, compressible=False)

    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_vtus=True, dump_nc=True,
    )
    # list of diagnostic fields, each defined in a class in diagnostics.py
    diagnostic_fields = [CourantNumber(), Divergence(), Perturbation('b')]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    b_opts = SUPGOptions()
    transported_fields = [
        TrapeziumRule(domain, "u"),
        SSPRK3(domain, "b", options=b_opts)
    ]
    transport_methods = [
        DGUpwind(eqns, "u"),
        DGUpwind(eqns, "b", ibp=b_opts.ibp)
    ]

    # Timestepper - pressure p is diagnostic
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transported_fields, transport_methods, solver_prognostics=["u", "b"]
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    b0 = stepper.fields("b")
    p0 = stepper.fields("p")

    # spaces
    Vb = b0.function_space()

    x, z = SpatialCoordinate(mesh)

    # first setup the background buoyancy profile
    # z.grad(bref) = N**2
    bref = z*(N**2)
    # interpolate the expression to the function
    b_b = Function(Vb).interpolate(bref)

    # setup constants
    b_pert = (
        deltab * sin(pi*z/domain_height)
        / (1 + (x - domain_width/2)**2 / pert_width**2)
    )
    # interpolate the expression to the function
    b0.interpolate(b_b + b_pert)

    boussinesq_hydrostatic_balance(eqns, b_b, p0)

    uinit = (as_vector([wind_initial, 0.0]))
    u0.project(uinit)

    # set the background buoyancy
    stepper.set_reference_profiles([('b', b_b)])

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    # Run!
    stepper.run(t=0, tmax=tmax)


# ---------------------------------------------------------------------------- #
# BUOYANCY SPECTRUM ANALYSIS
# ---------------------------------------------------------------------------- #


def initial_buoyancy_field(ncolumns, nlayers, element_order, domain_width,
                            domain_height, pert_width, deltab, N):
    """
    Builds the mesh and function space for a given element order, and returns
    the initial buoyancy field on that mesh.
    """

    base_mesh = PeriodicIntervalMesh(ncolumns, domain_width)
    mesh = ExtrudedMesh(base_mesh, nlayers, layer_height=domain_height/nlayers)
    domain = Domain(mesh, 1.0, 'CG', element_order)
    Vb = domain.spaces('theta')

    x, z = SpatialCoordinate(mesh)
    bref = z*(N**2)
    b_b = Function(Vb).interpolate(bref)
    b_pert = (
        deltab * sin(pi*z/domain_height)
        / (1 + (x - domain_width/2)**2 / pert_width**2)
    )
    b0 = Function(Vb).interpolate(b_b + b_pert)

    return b0


def plot_buoyancy_spectra(
        ncolumns=skamarock_klemp_incompressible_bouss_defaults['ncolumns'],
        nlayers=skamarock_klemp_incompressible_bouss_defaults['nlayers'],
        dirname=skamarock_klemp_incompressible_bouss_defaults['dirname']
):
    """
    Compares the Fourier spectrum of the initial buoyancy field on a
    constant-z slice, for the degree 0 and degree 1 configurations. The
    degree 1 mesh is half the resolution of the degree 0 mesh, so that the
    number of degrees of freedom is kept the same. The buoyancy is sampled at
    the centres of the degree 0 cells for both configurations.
    """

    domain_width = 3.0e5      # Width of domain (m)
    domain_height = 1.0e4     # Height of domain (m)
    pert_width = 5.0e3        # Width parameter of perturbation (m)
    deltab = 1.0e-2           # Magnitude of buoyancy perturbation (m/s^2)
    N = 0.01                  # Brunt-Vaisala frequency (1/s)

    def buoyancy_expr(x, z):
        bref = z*(N**2)
        b_pert = (
            deltab * sin(pi*z/domain_height)
            / (1 + (x - domain_width/2)**2 / pert_width**2)
        )
        return bref + b_pert

    # Points are the centres of the degree 0 cells, on a slice at mid-height
    dx = domain_width / ncolumns
    x_points = [(i + 0.5)*dx for i in range(ncolumns)]
    z_value = domain_height / 2
    points = [(x, z_value) for x in x_points]

    amplitudes = {}
    for element_order in (0, 1):
        if element_order == 1:
            base_mesh_1 = PeriodicIntervalMesh(ncolumns // 2, domain_width)
            mesh_1 = ExtrudedMesh(
                base_mesh_1, nlayers // 2, layer_height=domain_height/(nlayers // 2)
            )
            domain_1 = Domain(mesh_1, 1.0, 'CG', 1)
            b0 = initial_buoyancy_field_degree1_from_degree0(
                domain_1, buoyancy_expr
            )
        else:
            b0 = initial_buoyancy_field(
                ncolumns, nlayers, element_order,
                domain_width, domain_height, pert_width, deltab, N
            )

        values = np.array(b0.at(points))
        amplitudes[element_order] = np.abs(np.fft.rfft(values)) / len(values)

    wavenumbers = np.arange(len(amplitudes[0]))

    fig, ax = plt.subplots()
    ax.semilogy(wavenumbers, amplitudes[0], label='Degree 0', marker='o')
    ax.semilogy(wavenumbers, amplitudes[1], label='Degree 1', marker='x')
    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Amplitude')
    ax.set_title('Fourier spectrum of initial buoyancy field')
    ax.legend()
    fig.savefig(f'{dirname}_buoyancy_spectrum.png')


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
        default=skamarock_klemp_incompressible_bouss_defaults['ncolumns']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=skamarock_klemp_incompressible_bouss_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=skamarock_klemp_incompressible_bouss_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=skamarock_klemp_incompressible_bouss_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=skamarock_klemp_incompressible_bouss_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=skamarock_klemp_incompressible_bouss_defaults['dirname']
    )
    parser.add_argument(
        '--element_order',
        help="The polynomial degree of the elements. The ncolumns/nlayers "
        "grid is halved when degree 1 is used, to keep the number of "
        "degrees of freedom constant relative to degree 0.",
        type=int,
        choices=[0, 1],
        default=skamarock_klemp_incompressible_bouss_defaults['element_order']
    )
    parser.add_argument(
        '--plot_spectrum',
        help="If set, instead of running the model this compares the "
        "Fourier spectrum of the initial buoyancy field between the "
        "degree 0 and degree 1 configurations.",
        action='store_true'
    )
    args, unknown = parser.parse_known_args()

    if args.plot_spectrum:
        plot_buoyancy_spectra(
            ncolumns=args.ncolumns, nlayers=args.nlayers, dirname=args.dirname
        )
    else:
        del args.plot_spectrum
        skamarock_klemp_incompressible_bouss(**vars(args))
