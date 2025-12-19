"""
This module provides PETSc dictionaries for nonlinear and linear problems.

The solvers provided here are used for solving linear problems on mixed
finite element spaces.
"""

from gusto.equations import (CompressibleEulerEquations, BoussinesqEquations,
                             ShallowWaterEquations, ShallowWaterEquations_1d
                             )
from firedrake import (
    VectorSpaceBasis
)

__all__ = ["hybridised_solver_parameters"]


def hybridised_solver_parameters(equation, alpha=0.5, tau_values=None,
                                 nonlinear=False):
    """
    Returns PETSc solver settings for hybridised solver for mixed finite
    element problems.

    Parameters
    ----------
    equation (:class:`PrognosticEquation`): the model's equation.
    alpha : float, optional
        The implicitness parameter for the time discretisation. Default is 0.5.
    tau_values : dict, optional
        A dictionary of stabilization parameters for the hybridization.
        Default is None.
    nonlinear : bool, optional
        Whether the problem is nonlinear. Default is False.
    Returns
    -------
    settings : dict
        A dictionary containing the PETSc solver settings.
    appctx : dict
        A dictionary containing the application context for the solver.
    """

    def trace_nullsp(T):
        return VectorSpaceBasis(constant=True)

    # We chose our solver settings based on the equation being solved.

    # ============================================================================ #
    # Compressible Euler
    # ============================================================================ #

    if isinstance(equation, CompressibleEulerEquations):
        # Compressible Euler equations - (u, rho, theta) system. We use a bespoke
        # hybridization preconditioner for this system owing to the nonlinear pressure
        # gradient term.
        settings = {
            'ksp_monitor': None,
            'ksp_type': 'preonly',
            'mat_type': 'matfree',
            'pc_type': 'python',
            'pc_python_type': 'gusto.CompressibleHybridisedSCPC',
        }

        # We pass the implicit weighting (alpha) and tau_values to the preconditioner,
        # as well as the equation to construct the hybridized system.
        # Provide callback for the nullspace of the trace system with trace_nullsp.
        appctx = {
            'equations': equation,
            'alpha': alpha,
            'trace_nullspace': trace_nullsp
        }
        if tau_values is not None:
            appctx['tau_values'] = tau_values

    # ============================================================================ #
    # Boussinesq
    # ============================================================================ #

    elif isinstance(equation, BoussinesqEquations):
        # Boussinesq equations - (u, p, theta) system. We use a fieldsplit preconditioner
        # to first eliminate theta via Schur complement, then hybridize the (u, p) system.
        settings = {
            'ksp_monitor_true_residual': None,
            'ksp_view': ':ksp_view.log',
            'ksp_error_if_not_converged': None,
            'mat_type': 'matfree',
            'ksp_type': 'preonly',
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'schur',
            'pc_fieldsplit_schur_fact_type': 'full',
            'pc_fieldsplit_1_fields': '0,1',  # (u, p)
            'pc_fieldsplit_0_fields': '2',   # eliminate theta
            'fieldsplit_theta': {  # theta solve, simple block jacobi with ilu on blocks
                'ksp_type': 'preonly',
                'pc_type': 'python',
                'pc_python_type': 'firedrake.AssembledPC',
                'assembled_pc_type': 'bjacobi',
                'assembled_sub_pc_type': 'ilu',
            },
            'fieldsplit_1': {  # (u, p) solve, hybridized
                'ksp_monitor': None,
                'ksp_type': 'preonly',
                'pc_type': 'python',
                'pc_python_type': 'gusto.AuxiliaryPC',
                'aux': {
                    'mat_type': 'matfree',
                    'pc_type': 'python',
                    'pc_python_type': 'firedrake.HybridizationPC',  # Uses Firedrake's
                    'hybridization': {                              # hybridization PC
                        'ksp_type': 'cg',
                        'pc_type': 'gamg',  # AMG for trace system
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
        # We pass the Schur complement form to eliminate theta to the AuxiliaryPC.
        # This Schur complement form is defined in the BoussinesqEquations class.
        # Provide callback for the nullspace of the trace system with trace_nullsp.
        appctx = {
            'auxform': equation.schur_complement_form(alpha=alpha),
            "trace_nullspace": trace_nullsp,
        }

    # ============================================================================ #
    # Shallow Water
    # ============================================================================ #

    elif isinstance(equation, (ShallowWaterEquations, ShallowWaterEquations_1d)):
        # We have many different variations of the shallow water equations,
        # we need to work out which variables are being solved for.
        fields = equation.field_names

        if fields == ['u', 'D']:
            # (u, D) system - dry shallow water. No elimination via Schur complement,
            # just hybridization of the full system.
            settings = {
                'ksp_type': 'preonly',
                'mat_type': 'matfree',
                'pc_type': 'python',
                'pc_python_type': 'firedrake.HybridizationPC',  # Uses Firedrake's
                'hybridization': {                              # hybridization PC
                    'ksp_type': 'cg',
                    'pc_type': 'gamg',  # AMG for trace system
                    'ksp_rtol': 1e-8,
                    'mg_levels': {
                        'ksp_type': 'chebyshev',
                        'ksp_max_it': 2,
                        'pc_type': 'bjacobi',
                        'sub_pc_type': 'ilu'
                    }
                }
            }
            # Provide callback for the nullspace of the trace system with trace_nullsp.
            appctx = {
                "trace_nullspace": trace_nullsp,
            }

        # ============================================================================ #

        elif fields == ['u', 'D', 'b']:
            # (u, D, b) system - thermal shallow water
            settings = {
                'ksp_monitor_true_residual': None,
                'ksp_view': ':ksp_view.log',
                'ksp_error_if_not_converged': None,
                'mat_type': 'matfree',
                'ksp_type': 'preonly',
                'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'schur',
                'pc_fieldsplit_schur_fact_type': 'full',
                'pc_fieldsplit_1_fields': '0,1',  # (u, D) system
                'pc_fieldsplit_0_fields': '2',   # eliminate of b by Schur complement
                'fieldsplit_L2': {   # b solve, simple block jacobi with ilu on blocks
                    'ksp_monitor': None,
                    'ksp_type': 'preonly',
                    'pc_type': 'python',
                    'pc_python_type': 'firedrake.AssembledPC',
                    'assembled_pc_type': 'bjacobi',
                    'assembled_sub_pc_type': 'ilu',
                },
                'fieldsplit_1': {  # (u, D) solve, hybridized
                    'ksp_monitor': None,
                    'ksp_type': 'preonly',
                    'pc_type': 'python',
                    'pc_python_type': 'gusto.AuxiliaryPC',
                    'aux': {
                        'mat_type': 'matfree',
                        'pc_type': 'python',
                        'pc_python_type': 'firedrake.HybridizationPC',  # Uses Firedrake's
                        'hybridization': {                              # hybridization PC
                            'ksp_type': 'cg',
                            'pc_type': 'gamg',  # AMG for trace system
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
            # We pass the Schur complement form to eliminate b to the AuxiliaryPC.
            # This Schur complement form is defined in the ShallowWaterEquations class.
            # Provide callback for the nullspace of the trace system with trace_nullsp.
            appctx = {
                'auxform': equation.schur_complement_form(alpha=alpha),
                "trace_nullspace": trace_nullsp,
            }

        # ============================================================================ #

        elif fields[0:2] == ['u', 'D'] and len(fields) > 2 and fields[2] not in ['b', 'b_e']:

            # (u, D, scalars) system - moist convective shallow water

            # Create scalars list excluding u and D. Scalars are moisture variables and
            # are not solved for in the linear solver.
            scalars = ",".join(str(idx) for idx, name in enumerate(fields) if name not in ['u', 'D'])
            settings = {
                'mat_type': 'matfree',
                'ksp_type': 'preonly',
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "additive",  # additive fieldsplit is used
                                                   # to seperate scalars
                "pc_fieldsplit_0_fields": "0,1",  # (u, D) system to be hybridized
                "pc_fieldsplit_1_fields": scalars,  # do nothing for scalar fields
                "fieldsplit_0": {
                    'ksp_monitor_true_residual': None,
                    'ksp_type': 'preonly',
                    'pc_type': 'python',
                    'pc_python_type': 'firedrake.HybridizationPC',  # Uses Firedrake's
                    'hybridization': {                              # hybridization PC
                        'ksp_type': 'cg',
                        'pc_type': 'gamg',   # AMG for trace system
                        'ksp_rtol': 1e-8,
                        'mg_levels': {
                            'ksp_type': 'chebyshev',
                            'ksp_max_it': 2,
                            'pc_type': 'bjacobi',
                            'sub_pc_type': 'ilu'
                        }
                    }
                },
                "fieldsplit_1": {  # do nothing for scalar fields - the rhs is 0
                    'ksp_type': 'preonly',
                    'pc_type': 'none',
                },
            }

            # Provide callback for the nullspace of the trace system with trace_nullsp.
            appctx = {"trace_nullspace": trace_nullsp}

        # ============================================================================ #

        elif (fields[0:3] == ['u', 'D', 'b'] or fields[0:3] == ['u', 'D', 'b_e']) and len(fields) > 3:
            # (u, D, b, scalars) system - moist thermal shallow water
            # Create scalars list excluding u, D and b/b_e. Scalars are moisture
            # variables and are not solved for in the linear solver.
            scalars = ",".join(str(idx) for idx, name in enumerate(fields) if name not in ['u', 'D', 'b', 'b_e'])
            settings = {
                'ksp_monitor_true_residual': None,
                'ksp_view': ':ksp_view.log',
                'ksp_error_if_not_converged': None,
                'mat_type': 'matfree',
                'ksp_type': 'preonly',
                'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'additive',  # additive fieldsplit to seperate scalars
                'pc_fieldsplit_0_fields': '0,1,2',  # (u, D, b/b_e) system
                'pc_fieldsplit_1_fields': scalars,  # do nothing for scalar field
                'fieldsplit_0': {  # (u, D, b/b_e) solve with Schur complement elimination of b/b_e
                    'pc_type': 'fieldsplit',
                    'pc_fieldsplit_type': 'schur',
                    'pc_fieldsplit_schur_fact_type': 'full',
                    'pc_fieldsplit_1_fields': '0,1',
                    'pc_fieldsplit_0_fields': '2',   # eliminate b/b_e
                    'fieldsplit_L2': {  # b/b_e solve, simple block jacobi with ilu on blocks
                        'ksp_type': 'preonly',
                        'pc_type': 'python',
                        'pc_python_type': 'firedrake.AssembledPC',
                        'assembled_pc_type': 'bjacobi',
                        'assembled_sub_pc_type': 'ilu',
                    },
                    'fieldsplit_1': {  # (u, D) solve, hybridized
                        'ksp_monitor_true_residual': None,
                        'ksp_monitor': None,
                        'ksp_type': 'preonly',
                        'pc_type': 'python',
                        'pc_python_type': 'gusto.AuxiliaryPC',
                        'aux': {
                            'mat_type': 'matfree',
                            'pc_type': 'python',
                            'pc_python_type': 'firedrake.HybridizationPC',  # Uses Firedrake's hybridization PC
                            'hybridization': {
                                'ksp_type': 'cg',
                                'pc_type': 'gamg',   # AMG for trace system
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
                },
                'fieldsplit_L2': {  # do nothing for scalar field - the rhs is 0
                    'ksp_type': 'preonly',
                    'pc_type': 'none',
                },
            }
            # We pass the Schur complement form to eliminate b/b_e to the AuxiliaryPC.
            # This Schur complement form is defined in the ShallowWaterEquations class.
            # Provide callback for the nullspace of the trace system with trace_nullsp.
            appctx = {'auxform': equation.schur_complement_form(alpha=alpha),
                      "trace_nullspace": trace_nullsp,
                      }

        if nonlinear:
            # Nonlinear solver settings.
            settings['ksp_type'] = 'fgmres'
            settings['snes_type'] = 'newtonls'
            settings['snes_rtol'] = 1e-8
            settings['snes_max_it'] = 50
            settings['snes_monitor'] = None
            settings['ksp_rtol'] = 1e-8
            settings['ksp_max_it'] = 100
            settings['ksp_atol'] = 1e-8
            settings['snes_rtol'] = 1e-4
            settings['snes_atol'] = 1e-6
            settings['snes_converged_reason'] = None
            settings['snes_view'] = None
    return settings, appctx
