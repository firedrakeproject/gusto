"""
This module provides PETSc dictionaries for nonlinear and linear problems.

The solvers provided here are used for solving linear problems on mixed
finite element spaces.
"""

from gusto.equations import CompressibleEulerEquations, BoussinesqEquations, \
                            ShallowWaterEquations, ShallowWaterEquations_1d
from firedrake import (
    VectorSpaceBasis, COMM_WORLD
)

__all__ = ["HybridisedSolverParameters"]

def HybridisedSolverParameters(equation, alpha=0.5, tau_values=None, nonlinear=False):
    """
    Returns PETSc solver settings for hybridised solver for mixed finite element problems.

    Parameters
    ----------
    equation : str
        The type of equation being solved. Options are "CompressibleEuler", "Boussinesq", "ThermalSW", "ConvectiveSW", "ShallowWater".
    nonlinear : bool, optional
        Whether the problem is nonlinear. Default is False.
    alpha : float, optional
        The implicitness parameter for the time discretisation. Default is 0.5.
    tau_values : dict, optional
        A dictionary of stabilization parameters for the hybridization. Default is None.
    Returns
    -------
    settings : dict
        A dictionary containing the PETSc solver settings.
    appctx : dict
        A dictionary containing the application context for the solver.
    """
    def trace_nullsp(T):
            return VectorSpaceBasis(constant=True)
    if isinstance(equation, CompressibleEulerEquations):
        settings =  {'ksp_monitor': None,
                        'ksp_type': 'preonly',
                        'mat_type': 'matfree',
                        'pc_type': 'python',
                        'pc_python_type': 'gusto.CompressibleHybridisedSCPC'}


        appctx = {
            'equations': equation,
            'alpha': alpha,
            'trace_nullspace': trace_nullsp
        }
        if tau_values is not None:
            appctx['tau_values'] = tau_values
    elif isinstance(equation, BoussinesqEquations):
        settings = {
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

        appctx = {
            'auxform': equation.schur_complement_form(alpha=alpha),
            "trace_nullspace": trace_nullsp,
        }
    elif isinstance(equation, (ShallowWaterEquations, ShallowWaterEquations_1d)):
        # We need to work out which variables are being solved for.
        fields = equation.field_names

        if fields == ['u', 'D']:
            # (u, D) system - dry shallow water
            settings = {
                'ksp_type': 'preonly',
                'mat_type': 'matfree',
                'pc_type': 'python',
                'pc_python_type': 'firedrake.HybridizationPC',
                'hybridization': {'ksp_type': 'cg',
                                'pc_type': 'gamg',
                                'ksp_rtol': 1e-8,
                                'mg_levels': {'ksp_type': 'chebyshev',
                                                'ksp_max_it': 2,
                                                'pc_type': 'bjacobi',
                                                'sub_pc_type': 'ilu'}}
            }

            appctx = {
                "trace_nullspace": trace_nullsp,
            }
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
                'pc_fieldsplit_1_fields': '0,1',
                'pc_fieldsplit_0_fields': '2',  # eliminate temperature
                'fieldsplit_L2': {
                    'ksp_monitor': None,
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
            appctx = {
                'auxform': equation.schur_complement_form(alpha=alpha),
                "trace_nullspace": trace_nullsp,
            }
        elif fields[0:2] == ['u', 'D'] and len(fields) > 2 and fields[2] not in ['b', 'b_e']:

            # (u, D, scalars) system - moist convective shallow water

            # Create scalars list
            scalars = ",".join(str(idx) for idx, name in enumerate(fields) if name not in ['u', 'D'])
            settings = {
                'mat_type': 'matfree',
                'ksp_type': 'preonly',
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "additive",
                "pc_fieldsplit_0_fields": "0,1",    # (u, D)
                "pc_fieldsplit_1_fields": scalars,  # (scalars,)
                "fieldsplit_0": {  # hybridisation on the (u,D) system
                    'ksp_monitor_true_residual': None,
                    'ksp_type': 'preonly',
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
                        }
                    }
                },
                "fieldsplit_1": {  # Don't touch the transported fields
                    "ksp_type": "preonly",
                    "pc_type": "none"
                },
            }
            def trace_nullsp(T):
                return VectorSpaceBasis(constant=True, comm=COMM_WORLD)
            appctx = {"trace_nullspace": trace_nullsp}
        elif (fields[0:3] == ['u', 'D', 'b'] or fields[0:3] == ['u', 'D', 'b_e']) and len(fields) > 3:
            # (u, D, b, scalars) system - moist thermal shallow water
            scalars = ",".join(str(idx) for idx, name in enumerate(fields) if name not in ['u', 'D', 'b', 'b_e'])
            settings = {
                'ksp_monitor_true_residual': None,
                'ksp_view': ':ksp_view.log',
                'ksp_error_if_not_converged': None,
                'mat_type': 'matfree',
                'ksp_type': 'preonly',
                'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'additive',
                'pc_fieldsplit_0_fields': '0,1,2',
                'pc_fieldsplit_1_fields': scalars,  # do nothing for scalar field
                'fieldsplit_0': {
                    'pc_type': 'fieldsplit',
                    'pc_fieldsplit_type': 'schur',
                    'pc_fieldsplit_schur_fact_type': 'full',
                    'pc_fieldsplit_1_fields': '0,1',
                    'pc_fieldsplit_0_fields': '2',  # eliminate temperature
                    'fieldsplit_L2': {
                        'ksp_monitor': None,
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
                },
                'fieldsplit_L2': {
                    'ksp_type': 'preonly',
                    'pc_type': 'none',
                },
            }


            appctx = {'auxform': equation.schur_complement_form(alpha=alpha)}

    return settings, appctx