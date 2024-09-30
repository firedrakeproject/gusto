"""
This example uses the non-linear compressible Euler equations to solve the
vertical slice gravity wave test case of Skamarock and Klemp, 1994:
``Efficiency and Accuracy of the Klemp-Wilhelmson Time-Splitting Technique'',
MWR.

Potential temperature is transported using SUPG, and the degree 1 elements are
used.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from petsc4py import PETSc
PETSc.Sys.popErrorHandler()
import itertools
from firedrake import (
    as_vector, SpatialCoordinate, PeriodicIntervalMesh, ExtrudedMesh, exp, sin,
    Function, pi, COMM_WORLD, Constant, sqrt
)
import numpy as np
from gusto import (
    split_continuity_form, implicit, explicit, Timestepper, SDC, IMEX_Euler,
    time_derivative, transport, prognostic,
    Domain, IO, OutputParameters, SemiImplicitQuasiNewton, SSPRK3, DGUpwind,
    TrapeziumRule, SUPGOptions, CourantNumber, Perturbation, Gradient,
    CompressibleParameters, CompressibleEulerEquations, CompressibleSolver,
    compressible_hydrostatic_balance, logger, RichardsonNumber, MixedFSOptions,
    IMEX_SSP3
)

from time import perf_counter

skamarock_klemp_nonhydrostatic_defaults = {
    'ncolumns': 150,
    'nlayers': 10,
    'dt': 6.0,
    'tmax': 3600.,
    'dumpfreq': 300,
    'dirname': 'skamarock_klemp_nonhydrostatic2'
}


def skamarock_klemp_nonhydrostatic(
        ncolumns=skamarock_klemp_nonhydrostatic_defaults['ncolumns'],
        nlayers=skamarock_klemp_nonhydrostatic_defaults['nlayers'],
        dt=skamarock_klemp_nonhydrostatic_defaults['dt'],
        tmax=skamarock_klemp_nonhydrostatic_defaults['tmax'],
        dumpfreq=skamarock_klemp_nonhydrostatic_defaults['dumpfreq'],
        dirname=skamarock_klemp_nonhydrostatic_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Test case parameters
    # ------------------------------------------------------------------------ #

    domain_width = 3.0e5      # Width of domain (m)
    domain_height = 1.0e4     # Height of domain (m)
    Tsurf = 300.              # Temperature at surface (K)
    wind_initial = 20.        # Initial wind in x direction (m/s)
    pert_width = 5.0e3        # Width parameter of perturbation (m)
    deltaTheta = 1.0e-2       # Magnitude of theta perturbation (K)
    N = 0.01                  # Brunt-Vaisala frequency (1/s)

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    element_order = 1

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain -- 3D volume mesh
    base_mesh = PeriodicIntervalMesh(ncolumns, domain_width)
    mesh = ExtrudedMesh(base_mesh, nlayers, layer_height=domain_height/nlayers)
    domain = Domain(mesh, dt, "CG", element_order)

    # Equation
    theta_opts = SUPGOptions(field_name="theta")
    parameters = CompressibleParameters()
    eqns = CompressibleEulerEquations(domain, parameters)
    eqns = split_continuity_form(eqns)


    eqns.label_terms(lambda t: not any(t.has_label(time_derivative,transport)), implicit)
    eqns.label_terms(lambda t: t.has_label(transport), explicit)
    # eqns.label_terms(lambda t: t.get(prognostic) != 'theta' and t.has_label(transport), explicit)

    # I/O
    points_x = np.linspace(0., domain_width, 100)
    points_z = [domain_height/2.]
    points = np.array([p for p in itertools.product(points_x, points_z)])

    # Dumping point data using legacy PointDataOutput is not supported in parallel
    if COMM_WORLD.size == 1:
        output = OutputParameters(
            dirname=dirname, dumpfreq=dumpfreq, pddumpfreq=dumpfreq,
            dump_vtus=False, dump_nc=True,
            point_data=[('theta_perturbation', points)],
        )
    else:
        logger.warning(
            'Dumping point data using legacy PointDataOutput is not'
            ' supported in parallel\nDisabling PointDataOutput'
        )
        output = OutputParameters(
            dirname=dirname, dumpfreq=dumpfreq, pddumpfreq=dumpfreq,
            dump_vtus=True, dump_nc=True,
        )

    diagnostic_fields = [
        CourantNumber(), Gradient('u'), Perturbation('theta'),
        Gradient('theta_perturbation'), Perturbation('rho'),
        RichardsonNumber('theta', parameters.g/Tsurf), Gradient('theta')
    ]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # # Transport schemes
    # theta_opts = SUPGOptions()
    # transported_fields = [
    #     SSPRK3(domain, "u", subcycle_by_courant=0.2),
    #     SSPRK3(domain, "rho", subcycle_by_courant=0.2),
    #     SSPRK3(domain, "theta", subcycle_by_courant=0.2, options=theta_opts)
    # ]

    transport_methods = [
        DGUpwind(eqns, "u"),
        DGUpwind(eqns, "rho"),
        DGUpwind(eqns, "theta", ibp=theta_opts.ibp)
    ]

    nl_solver_parameters = {
        "snes_converged_reason": None,
        'ksp_ew': None,
        "ksp_ew_version": 1,
        "ksp_ew_threshold": 1e-5,
        "ksp_ew_rtol0": 1e-3,
        "mat_type": "matfree",
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-12,
        'ksp_atol': 1e-12,
        "ksp_converged_reason": None,
        "ksp_max_it": 400,
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC","assembled": {
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMStarPC",
            "pc_star": {
                "construct_dim": 0,
                "sub_sub": {
                    "pc_type": "lu",
                    "pc_factor_mat_ordering_type": "rcm",
                    "pc_factor_reuse_ordering": None,
                    "pc_factor_reuse_fill": None,
                    "pc_factor_fill": 1.2
                }
            },
        },}
    

    nl_solver_parameters = {"pc_type": "fieldsplit",
                'mat_type': 'matfree',
                'ksp_type': 'gmres',
                'snes_rtol': 1.0e-6,
                'snes_atol': 1.0e-6,
                'ksp_rtol': 1.0e-6,
                'ksp_atol': 1.0e-8,
                "pc_fieldsplit_type": "schur",
                'pc_fieldsplit_schur_precondition': 'FULL',
                'snes_monitor': None,
                'ksp_monitor_true_residual': None,
                'snes_converged_reason': None,
                'ksp_converged_reason': None,
                # first split contains first two fields, second
                # contains the third
                "pc_fieldsplit_0_fields": "0, 1",
                "pc_fieldsplit_1_fields": "2",
                "fieldsplit_0": {'ksp_type': 'preonly',
                                'pc_type': 'python',
                                 'ksp_rtol': 1.0e-6,
                                'ksp_atol': 1.0e-8,
                                'ksp_monitor_true_residual': None,
                                'ksp_converged_reason': None,
                                'pc_python_type': 'firedrake.HybridizationPC',
                'hybridization': {'ksp_type': 'fgmres',
                                   'ksp_rtol': 1.0e-6,
                                    'ksp_atol': 1.0e-8,
                                 'ksp_monitor_true_residual': None,
                                'ksp_converged_reason': None,
                                         'pc_type': 'bjacobi',
                                         'sub_pc_type': 'ilu'}},
                    "fieldsplit_1": {'ksp_type': 'cg',
                                      'ksp_rtol': 1.0e-6,
                                     'ksp_atol': 1.0e-8,
                                'ksp_monitor_true_residual': None,
                                'ksp_converged_reason': None,
                             'pc_type': 'bjacobi',
                             'sub_pc_type': 'ilu'}}

    nl_solver_parameters = {
    "pc_type": "fieldsplit",
    'mat_type': 'matfree',
    'ksp_type': 'gmres',
    'snes_rtol': 1.0e-6,
    'snes_atol': 1.0e-8,  # Tighter tolerance
    'ksp_rtol': 1.0e-6,   # Tighter relative tolerance
    'ksp_atol': 1.0e-8,   # Tighter absolute tolerance
    "pc_fieldsplit_type": "schur",
    'pc_fieldsplit_schur_fact_type': 'FULL',
    'snes_monitor': '',
    'ksp_monitor_true_residual': '',
    'snes_converged_reason': '',
    'ksp_converged_reason': '',
    # First split contains first two fields, second contains the third
    "pc_fieldsplit_0_fields": "0, 1",
    "pc_fieldsplit_1_fields": "2",
    "fieldsplit_0": {
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.HybridizationPC',
        'hybridization': {
               'ksp_type': 'fgmres',
                                            'ksp_rtol': 1.0e-8,
                                            'ksp_atol': 1.0e-8,
                                            'ksp_max_it': 100,
                                            'pc_type': 'gamg',
                                            'pc_gamg_sym_graph': None,
                                            'mg_levels': {'ksp_type': 'gmres',
                                                        'ksp_max_it': 5,
                                                        'pc_type': 'bjacobi',
                                                        'sub_pc_type': 'ilu'}}

    },
    "fieldsplit_1": {
        'ksp_type': 'cg',
        'ksp_rtol': 1.0e-6,
        'ksp_atol': 1.0e-8,
        'pc_type': 'asm',
        'sub_pc_type': 'ilu',

    }
}
    # {'ksp_type': 'fgmres',
    #                                         'ksp_rtol': 1.0e-8,
    #                                         'ksp_atol': 1.0e-8,
    #                                         'ksp_max_it': 100,
    #                                         'pc_type': 'gamg',
    #                                         'pc_gamg_sym_graph': None,
    #                                         'mg_levels': {'ksp_type': 'gmres',
    #                                                     'ksp_max_it': 5,
    #                                                     'pc_type': 'bjacobi',
    #                                                     'sub_pc_type': 'ilu'}}
    # nl_solver_parameters = {'mat_type': 'matfree',
    #           'ksp_type': 'preonly',
    #           'pc_type': 'python',
    #           'pc_python_type': 'firedrake.HybridizationPC',
    #           'hybridization': {'ksp_type': 'cg',
    #                             'pc_type': 'none',
    #                             'ksp_rtol': 1e-12,
    #                             'mat_type': 'matfree',
    #                             'localsolve': {'ksp_type': 'preonly',
    #                                            'pc_type': 'fieldsplit',
    #                                            'pc_fieldsplit_type': 'schur',
    #                                            'fieldsplit_1': {'ksp_type': 'default',
    #                                                             'pc_type': 'jacobi'}}}}

    nl_solver_parameters = {
    "snes_converged_reason": None,
    "snes_rtol": 1e-5,
    "snes_atol": 1e-5,
    "ksp_type": "fgmres",
    "ksp_rtol": 1e-6,  # Relaxed relative tolerance
    "ksp_atol": 1e-6,  # Relaxed absolute tolerance
    "ksp_max_it": 500,  # Increased max iterations
    "ksp_ew": None,
    "ksp_ew_version": 1,
    "ksp_ew_threshold": 1e-5,
    "ksp_ew_rtol0": 1e-3,
    "mat_type": "matfree",
    "ksp_converged_reason": None,
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled": {
        "pc_type": "python",
        "pc_python_type": "firedrake.ASMStarPC",
        "pc_star": {
            "construct_dim": 0,
            "pc_asm_overlap": 3,  # Increased overlap for ASM
            "sub_sub": {
                "pc_type": "ilu",  # Switch from LU to ILU for better performance
                "pc_factor_levels": 3,  # ILU fill level
                "pc_factor_reuse_ordering": True,  # Reuse LU ordering for efficiency
                "pc_factor_fill": 1.2,
            }
        },
    },
}
    
    nl_solver_parameters = {
    # Nonlinear Solver (for implicit part)
    "snes_type": "newtonls",  # Nonlinear solver: Newton with line search
    "snes_rtol": 1e-6,  # Relative tolerance for SNES
    "snes_atol": 1e-8,  # Absolute tolerance for SNES
    "snes_max_it": 50,  # Maximum nonlinear iterations
    "snes_monitor": None,  # Monitor the nonlinear residuals
    
    # Linear Solver (Krylov solver)
    "ksp_type": "fgmres",  # Flexible GMRES, useful for non-symmetric systems
    "ksp_rtol": 1e-8,  # Relative tolerance for Krylov solver
    "ksp_atol": 1e-10,  # Absolute tolerance for Krylov solver
    "ksp_max_it": 100,  # Maximum iterations for Krylov solver
    "ksp_monitor_true_residual": None,  # Monitor true residuals

    # Preconditioner (Field-Split)
    "pc_type": "fieldsplit",  # Field-splitting preconditioner for 3 fields
    "pc_fieldsplit_type": "schur",  # Use Schur complement to decouple fields
    "pc_fieldsplit_schur_fact_type": "diag",  # Diagonal Schur complement approximation
    "pc_fieldsplit_0_fields": "0, 1",  # First block: velocity and density
    "pc_fieldsplit_1_fields": "2",  # Second block: theta (potential temperature)

    "fieldsplit_0":{  
        "ksp_type": "fgmres",  # GMRES for robustness
        "pc_type": "python",
        "ksp_rtol": 1e-5,  # Adjust tolerance
        "ksp_atol": 1e-7,
        "pc_python_type": "firedrake.AssembledPC",
        "assembled": {
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMStarPC",
            "pc_star": {
                "construct_dim": 0,
                "pc_asm_overlap": 2,  # Increased overlap for ASM
                "sub_sub": {
                    "pc_type": "ilu",  # Switch from LU to ILU for better performance
                    "pc_factor_levels": 2,  # ILU fill level
                    "pc_factor_reuse_ordering": True,  # Reuse LU ordering for efficiency
                    "pc_factor_fill": 1.2,
                }
            },
        },},
    # Theta Block (Field Split 1)
    "fieldsplit_1": {
        "ksp_type": "cg",  # Use Conjugate Gradient for theta (potential temperature)
        "pc_type": "jacobi",  # Simple Jacobi preconditioner for theta
        "ksp_rtol": 1e-5,
        "ksp_atol": 1e-7
    },
}
    nl_solver_parameters = {
    # Nonlinear Solver
    "snes_type": "newtonls",  # Newton's method with line search
    "snes_rtol": 1e-5,  # Loosened nonlinear relative tolerance
    "snes_atol": 1e-6,  # Loosened nonlinear absolute tolerance
    "snes_max_it": 50,  # Max nonlinear iterations
    "snes_linesearch_type": "basic",  # Basic line search

    # Linear Solver (Krylov Solver)
    "ksp_type": "fgmres",  # Use CG instead of FGMRES for symmetric systems (velocity/density)
    "ksp_rtol": 1e-5,  # Looser Krylov relative tolerance
    "ksp_atol": 1e-8,  # Adjust Krylov absolute tolerance
    "ksp_max_it": 100,  # Max Krylov iterations
    "ksp_monitor_true_residual": None,  # Monitor true residuals

    # Preconditioner (Field-Split)
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",  # Use Schur complement to decouple fields
    "pc_fieldsplit_schur_fact_type": "lower",  # Use lower triangular Schur complement

    # Velocity and Density Block (Fieldsplit 0)
    "pc_fieldsplit_0_fields": "0, 1",  # Velocity and density
    "fieldsplit_0":{  
        "ksp_type": "fgmres",  # GMRES for robustness
        "pc_type": "python",
        "ksp_rtol": 1e-5,  # Adjust tolerance
        "ksp_atol": 1e-7,
        "pc_python_type": "firedrake.AssembledPC",
        "assembled": {
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMStarPC",
            "pc_star": {
                "construct_dim": 0,
                "pc_asm_overlap": 1,  # Increased overlap for ASM
                "sub_sub": {
                    "pc_type": "ilu",  # Switch from LU to ILU for better performance
                    "pc_factor_levels": 1,  # ILU fill level
                    "pc_factor_reuse_ordering": True,  # Reuse LU ordering for efficiency
                    "pc_factor_fill": 1.2,
                }
            },
        },},
    # Theta Block (Fieldsplit 1)
    "pc_fieldsplit_1_fields": "2",  # Theta (potential temperature)
    "fieldsplit_1": {
        "ksp_type": "cg",  # Use CG for the Theta field
        "pc_type": "jacobi",  # Simple Jacobi preconditioner for theta
        "ksp_rtol": 1e-5,  # Adjusted Krylov tolerance
        "ksp_atol": 1e-8,  # Adjusted Krylov tolerance
    },
}

    
#     nl_solver_parameters = {
#     "snes_type": "nasm",
#     "snes_nasm_type": "basic",
#     "snes_nasm_overlap": 2,
#     "mat_partitioning_type": "metis",
#     "snes_converged_reason": None,
#     "snes_max_it": 150,

#     # Solver for each subdomain
#     "snes_nasm_subsnes": {
#         "snes_type": "newtontr",
#         "ksp_type": "gmres",       # Use iterative GMRES solver for each subdomain
#         "pc_type": "hypre",        # Use multigrid preconditioner for each subdomain
#         "pc_hypre_type": "boomeramg",
#         "ksp_rtol": 1e-5,
#         "ksp_max_it": 100
#     }
# }


    nl_solver_parameters = {
    "snes_converged_reason": None,
    'ksp_ew': None,
    "ksp_ew_version": 1,
    "ksp_ew_threshold": 1e-5,
    "ksp_ew_rtol0": 1e-3,
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "ksp_rtol": 1e-5,
    'ksp_atol': 1e-7,
    "ksp_converged_reason": None,
    "ksp_max_it": 400,
    "pc_type": "python",
    "ksp_rtol": 1e-5,  # Adjust tolerance
    "ksp_atol": 1e-7,
    "pc_python_type": "firedrake.AssembledPC",
    "assembled": {
        "pc_type": "python",
        "pc_python_type": "firedrake.ASMStarPC",
        "pc_star": {
            "construct_dim": 0,
            "pc_asm_overlap": 2,  # Increased overlap for ASM
            "sub_sub": {
                "pc_type": "ilu",  # Switch from LU to ILU for better performance
                "pc_factor_levels": 2,  # ILU fill level
                "pc_factor_reuse_ordering": True,  # Reuse LU ordering for efficiency
                "pc_factor_fill": 1.2,
            }
        },
    },}
    
#     nl_solver_parameters = {
#     # Nonlinear solver: Newton linesearch
#     "snes_type": "newtonls",
#     "snes_rtol": 1e-4,
#     "snes_atol": 1e-5,
#     "snes_max_it": 50,

#     # Linear solver: FGMRES with schur complement preconditioner
#     "ksp_type": "fgmres",
#     "ksp_rtol": 1e-8,
#     "ksp_atol": 1e-7,
#     "pc_type": "fieldsplit",
#     "ksp_max_it": 200,
#     "pc_fieldsplit_type": "schur",
#     "pc_fieldsplit_schur_fact_type": "full",
#     "pc_fieldsplit_schur_precondition": "selfp",

#     # u, rho block uses Addidtive Schwarz preconditioner with ASMStar subdomain solver
#     "pc_fieldsplit_0_fields": "0, 2",
#     "fieldsplit_0":{  
#         "ksp_type": "gmres",
#         "pc_type": "python",
#         "ksp_rtol": 1e-8,
#         "ksp_atol": 1e-7,
#         "ksp_max_it": 200,
#         "pc_python_type": "firedrake.AssembledPC",
#         "assembled": {
#             "pc_type": "python",
#             "pc_python_type": "firedrake.ASMStarPC",
#             "pc_star": {
#                 "construct_dim": 0,
#                 "pc_asm_overlap": 2,
#                 "sub_sub": {
#                     "pc_type": "ilu",
#                     "pc_factor_levels": 3,
#                     "pc_factor_reuse_ordering": True,
#                     "pc_factor_fill": 1.2,
#                 }
#             },
#         },},
#     # theta block, simple cg with block jacobi preconditioner
#     "pc_fieldsplit_1_fields": "1",
#     "fieldsplit_1": {
#         "ksp_type": "cg",
#         "ksp_max_it": 200,  # Maximum iterations for the linear solver
#         "pc_type": "bjacobi",
#         "sub_pc_type": "ilu",
#         "ksp_rtol": 1e-5,
#         "ksp_atol": 1e-7,
#     },
# }


    # IMEX time stepper
    scheme = IMEX_SSP3(domain, options = theta_opts, solver_parameters=nl_solver_parameters)
    #Time stepper
    stepper = Timestepper(eqns, scheme, io, transport_methods)
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
    g = parameters.g

    x, z = SpatialCoordinate(mesh)

    # N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
    thetab = Tsurf*exp(N**2*z/g)

    theta_b = Function(Vt).interpolate(thetab)
    rho_b = Function(Vr)

    # Calculate hydrostatic exner
    compressible_hydrostatic_balance(eqns, theta_b, rho_b)

    theta_pert = (
        deltaTheta * sin(pi*z/domain_height)
        / (1 + (x - domain_width/2)**2 / pert_width**2)
    )
    theta0.interpolate(theta_b + theta_pert)
    rho0.assign(rho_b)
    u0.project(as_vector([wind_initial, 0.0]))

    stepper.set_reference_profiles([('rho', rho_b), ('theta', theta_b)])

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #
    t1 = perf_counter()
    stepper.run(t=0, tmax=tmax)
    t2 = perf_counter()
    print("Elapsed time:", t2-t1)

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
        default=skamarock_klemp_nonhydrostatic_defaults['ncolumns']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=skamarock_klemp_nonhydrostatic_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=skamarock_klemp_nonhydrostatic_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=skamarock_klemp_nonhydrostatic_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=skamarock_klemp_nonhydrostatic_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=skamarock_klemp_nonhydrostatic_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    skamarock_klemp_nonhydrostatic(**vars(args))
