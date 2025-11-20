"""
This example uses the compressible Boussinesq equations to solve the vertical
slice gravity wave test case of Skamarock and Klemp, 1994:
``Efficiency and Accuracy of the Klemp-Wilhelmson Time-Splitting Technique'',
MWR.

Buoyancy is transported using SUPG, and the degree 1 elements are used.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import (
    as_vector, PeriodicIntervalMesh, ExtrudedMesh, sin, SpatialCoordinate,
    Function, pi
)
from gusto import (
    Domain, IO, OutputParameters, SemiImplicitQuasiNewton, SSPRK3, DGUpwind,
    TrapeziumRule, SUPGOptions, Divergence, Perturbation, CourantNumber,
    BoussinesqParameters, BoussinesqEquations, BoussinesqSolver, LinearTimesteppingSolver,
    boussinesq_hydrostatic_balance
)

skamarock_klemp_compressible_bouss_defaults = {
    'ncolumns': 300,
    'nlayers': 10,
    'dt': 6.0,
    'tmax': 3600.,
    'dumpfreq': 300,
    'dirname': 'skamarock_klemp_compressible_bouss'
}


def skamarock_klemp_compressible_bouss(
        ncolumns=skamarock_klemp_compressible_bouss_defaults['ncolumns'],
        nlayers=skamarock_klemp_compressible_bouss_defaults['nlayers'],
        dt=skamarock_klemp_compressible_bouss_defaults['dt'],
        tmax=skamarock_klemp_compressible_bouss_defaults['tmax'],
        dumpfreq=skamarock_klemp_compressible_bouss_defaults['dumpfreq'],
        dirname=skamarock_klemp_compressible_bouss_defaults['dirname']
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
    cs = 300.                 # Speed of sound (m/s)

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    element_order = 1

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    base_mesh = PeriodicIntervalMesh(ncolumns, domain_width)
    mesh = ExtrudedMesh(base_mesh, nlayers, layer_height=domain_height/nlayers)
    domain = Domain(mesh, dt, 'CG', element_order)

    # Equation
    parameters = BoussinesqParameters(mesh, cs=cs)
    eqns = BoussinesqEquations(domain, parameters)

    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_vtus=True, dump_nc=False,
    )
    # list of diagnostic fields, each defined in a class in diagnostics.py
    diagnostic_fields = [CourantNumber(), Divergence(), Perturbation('b')]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    b_opts = SUPGOptions()
    transported_fields = [
        TrapeziumRule(domain, "u"),
        SSPRK3(domain, "p"),
        SSPRK3(domain, "b", options=b_opts)
    ]
    transport_methods = [
        DGUpwind(eqns, "u"),
        DGUpwind(eqns, "p"),
        DGUpwind(eqns, "b", ibp=b_opts.ibp)
    ]

    # Linear solver
    solver_parameters = {
        'ksp_monitor_true_residual': None,
        'ksp_view': ':ksp_view.log',
        'ksp_error_if_not_converged': None,
        'mat_type': 'matfree',
        'ksp_type': 'preonly',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'schur',
        'pc_fieldsplit_schur_fact_type': 'full',
        'pc_fieldsplit_1_fields': '0,1',
        'pc_fieldsplit_0_fields': '2',  # eliminate temperature
        'fieldsplit_theta': {
            'ksp_type': 'preonly',
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'assembled_pc_type': 'bjacobi',
            'assembled_sub_pc_type': 'ilu',
        },
        'fieldsplit_1': {
            'ksp_monitor': None,
            'ksp_type': 'preonly',
            'pc_type': 'python',
            'pc_python_type': 'gusto.AuxiliaryPC',
            'aux': {
                'mat_type': 'matfree',
                'pc_type': 'python',
                'pc_python_type': 'firedrake.HybridizationPC',
                'hybridization': {
                    'ksp_type': 'cg',
                    'pc_type': 'gamg',
                    'ksp_rtol': 1e-8,
                    'mg_levels': {
                        'ksp_type': 'chebyshev',
                        'ksp_max_it': 2,
                        'pc_type': 'bjacobi',
                        'sub_pc_type': 'ilu'
                    },
                    'mg_coarse': {
                        'ksp_type': 'preonly',
                        'pc_type': 'lu',
                        'pc_factor_mat_solver_type': 'mumps',
                    },
                },
            },
        },
    }

    def trace_nullsp(T):
        return VectorSpaceBasis(constant=True)


    appctx = {
        'auxform': eqns.schur_complement_form(alpha=0.5),
        "trace_nullspace": trace_nullsp,
    }

    linear_solver = LinearTimesteppingSolver(
        eqns, alpha=0.5, options_prefix="boussinesq",
        solver_parameters=solver_parameters,
        appctx=appctx
    )

    # Time stepper
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transported_fields, transport_methods,
        linear_solver=linear_solver
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
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

    # setup constants
    b_pert = (
        deltab * sin(pi*z/domain_height)
        / (1 + (x - domain_width/2)**2 / pert_width**2)
    )
    # interpolate the expression to the function
    b0.interpolate(b_b + b_pert)

    p_b = Function(Vp)
    boussinesq_hydrostatic_balance(eqns, b_b, p_b)
    p0.assign(p_b)

    uinit = (as_vector([wind_initial, 0.0]))
    u0.project(uinit)

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
        default=skamarock_klemp_compressible_bouss_defaults['ncolumns']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=skamarock_klemp_compressible_bouss_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=skamarock_klemp_compressible_bouss_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=skamarock_klemp_compressible_bouss_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=skamarock_klemp_compressible_bouss_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=skamarock_klemp_compressible_bouss_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    skamarock_klemp_compressible_bouss(**vars(args))
