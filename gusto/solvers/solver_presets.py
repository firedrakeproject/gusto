"""
This module provides PETSc dictionaries for linear problems.

The solvers provided here are used for solving linear problems on mixed
finite element spaces.
"""

from gusto.equations import (
    CompressibleEulerEquations, BoussinesqEquations, ShallowWaterEquations,
    ShallowWaterEquations_1d, ThermalShallowWaterEquations
)
from gusto.core.logging import logger, DEBUG
from firedrake import VectorSpaceBasis

__all__ = ["hybridised_solver_parameters", "monolithic_solver_parameters"]


def hybridised_solver_parameters(equation, alpha=0.5, tau_values=None):
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
    Returns
    -------
    settings : dict
        A dictionary containing the PETSc solver settings.
    appctx : dict
        A dictionary containing the application context for the solver.
    """

    # Callback function for the nullspace of the trace system
    def trace_nullsp(T):
        return VectorSpaceBasis(constant=True)

    # Prepare some generic variables to help with logic below ------------------

    # Create list of "tracer" variables that are not solved for by the linear
    # solver -- i.e. the list of variables excluding the "solver prognostics".
    # For example, these could be moisture variables
    # TODO: this logic should directly use the list of "solver_prognostics"
    num_tracers = len(equation.active_tracers)
    tracer_names = [tracer.name for tracer in equation.active_tracers]
    scalars = ",".join(
        str(idx) for idx, name in enumerate(equation.field_names)
        if name in tracer_names
    )
    is_shallow_water = isinstance(equation, (
        ShallowWaterEquations, ShallowWaterEquations_1d
    ))

    # ------------------------------------------------------------------------ #
    # We chose our solver settings based on the equation being solved...

    # ======================================================================== #
    # Compressible Euler
    # ======================================================================== #

    if isinstance(equation, CompressibleEulerEquations):
        # Compressible Euler equations - (u, rho, theta) system. We use a
        # bespoke hybridization preconditioner for this system owing to the
        # nonlinear pressure gradient term.
        settings = {
            'ksp_monitor': None,
            'ksp_type': 'preonly',
            'mat_type': 'matfree',
            'pc_type': 'python',
            'pc_python_type': 'gusto.CompressibleHybridisedSCPC',
        }

        # We pass the implicit weighting parameter (alpha) and tau_values to the
        # preconditioner, and the equation to construct the hybridized system.
        # Provide callback for nullspace of the trace system with trace_nullsp
        appctx = {
            'equations': equation,
            'alpha': alpha,
            'trace_nullspace': trace_nullsp
        }
        if tau_values is not None:
            appctx['tau_values'] = tau_values

    # ======================================================================== #
    # Boussinesq
    # ======================================================================== #

    elif isinstance(equation, BoussinesqEquations):
        # Boussinesq equations - (u, p, theta) system. We use a fieldsplit
        # preconditioner to first eliminate theta via Schur complement, then
        # hybridize the (u, p) system.
        settings = {
            'ksp_error_if_not_converged': None,
            'mat_type': 'matfree',
            'ksp_type': 'preonly',
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'schur',
            'pc_fieldsplit_schur_fact_type': 'full',
            'pc_fieldsplit_1_fields': '0,1',  # (u, p)
            'pc_fieldsplit_0_fields': '2',  # eliminate theta
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
        # To eliminate theta, we pass the Schur complement form to the
        # AuxiliaryPC. This Schur complement form is defined in the
        # BoussinesqEquations class.
        # Provide callback for nullspace of the trace system with trace_nullsp
        appctx = {
            'auxform': equation.schur_complement_form(alpha=alpha),
            'trace_nullspace': trace_nullsp,
        }

    # ======================================================================== #
    # Moist Thermal Shallow Water
    # ======================================================================== #

    elif isinstance(equation, ThermalShallowWaterEquations) and num_tracers > 0:
        # (u, D, b, scalars) system - moist thermal shallow water
        settings = {
            'ksp_error_if_not_converged': None,
            'mat_type': 'matfree',
            'ksp_type': 'preonly',
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'additive',  # additive fieldsplit to separate scalars
            'pc_fieldsplit_0_fields': '0,1,2',  # (u, D, b/b_e) system
            'pc_fieldsplit_1_fields': scalars,  # do nothing for scalar field
            'fieldsplit_0': {  # (u, D, b/b_e) solve with Schur complement elimination of b/b_e
                'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'schur',
                'pc_fieldsplit_schur_fact_type': 'full',
                'pc_fieldsplit_1_fields': '0,1',
                'pc_fieldsplit_0_fields': '2',  # eliminate b/b_e
                'fieldsplit_L2': {  # b/b_e solve, simple block jacobi with ilu on blocks
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
                        'pc_python_type': 'firedrake.HybridizationPC',  # Uses Firedrake's hybridization PC
                        'hybridization': {
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
            },
            'fieldsplit_L2': {  # do nothing for scalar field - the rhs is 0
                'ksp_type': 'preonly',
                'pc_type': 'none',
            },
        }
        # To eliminate b/b_e, we pass the Schur complement form to the
        # AuxiliaryPC. This Schur complement form is defined in the
        # ThermalShallowWaterEquations class.
        # Provide callback for nullspace of the trace system with trace_nullsp
        appctx = {
            'auxform': equation.schur_complement_form(alpha=alpha),
            'trace_nullspace': trace_nullsp,
        }

    # ======================================================================== #
    # Thermal Shallow Water (Dry)
    # ======================================================================== #

    elif isinstance(equation, ThermalShallowWaterEquations):
        # (u, D, b) system - thermal shallow water
        settings = {
            'ksp_error_if_not_converged': None,
            'mat_type': 'matfree',
            'ksp_type': 'preonly',
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'schur',
            'pc_fieldsplit_schur_fact_type': 'full',
            'pc_fieldsplit_1_fields': '0,1',  # (u, D) system
            'pc_fieldsplit_0_fields': '2',  # eliminate of b by Schur complement
            'fieldsplit_L2': {  # b solve, simple block jacobi with ilu on blocks
                'ksp_type': 'preonly',
                'pc_type': 'python',
                'pc_python_type': 'firedrake.AssembledPC',
                'assembled_pc_type': 'bjacobi',
                'assembled_sub_pc_type': 'ilu',
            },
            'fieldsplit_1': {  # (u, D) solve, hybridized
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
        # To eliminate b we pass the Schur complement form to the AuxiliaryPC.
        # This Schur complement form is defined in the
        # ThermalShallowWaterEquations class.
        # Provide callback for nullspace of the trace system with trace_nullsp
        appctx = {
            'auxform': equation.schur_complement_form(alpha=alpha),
            'trace_nullspace': trace_nullsp,
        }

    # ======================================================================== #
    # Moist Shallow Water (non-thermal)
    # ======================================================================== #

    elif is_shallow_water and num_tracers > 0:
        # (u, D, scalars) system - moist convective shallow water
        settings = {
            'mat_type': 'matfree',
            'ksp_type': 'preonly',
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "additive",  # additive fieldsplit is used
                                               # to separate scalars
            "pc_fieldsplit_0_fields": "0,1",  # (u, D) system to be hybridized
            "pc_fieldsplit_1_fields": scalars,  # do nothing for scalar fields
            "fieldsplit_0": {
                'ksp_type': 'preonly',
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
            },
            "fieldsplit_1": {  # do nothing for scalar fields - the rhs is 0
                'ksp_type': 'preonly',
                'pc_type': 'none',
            },
        }

        # Provide callback for the nullspace of trace system with trace_nullsp
        appctx = {'trace_nullspace': trace_nullsp}

    # ======================================================================== #
    # Shallow Water (standard)
    # ======================================================================== #

    elif is_shallow_water and num_tracers == 0:
        # (u, D) system - dry shallow water
        # No elimination via Schur complement, just hybridization of the
        # full system.
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
            'trace_nullspace': trace_nullsp,
        }

    else:
        raise NotImplementedError(
            "Hybridised solver presets not implemented for equation "
            f"type {type(equation)}"
        )

    # ======================================================================== #
    # Generic settings
    # ======================================================================== #

    if logger.isEnabledFor(DEBUG):
        settings["ksp_monitor_true_residual"] = None
        # TODO: the following output is not picked up by the Gusto logger
        fieldsplit_keys = [f'fieldsplit_{i}' for i in ['0', '1', 'L2']]
        for key in fieldsplit_keys:
            if key in settings:
                settings[key]['ksp_monitor_true_residual'] = None

    return settings, appctx


def monolithic_solver_parameters():
    """
    Returns PETSc solver settings for monolithic solver for mixed finite
    element problems.

    Returns
    -------
    settings : dict
        A dictionary containing the PETSc solver settings.
    """
    settings = {
        "snes_type": "ksponly",
        "mat_type": "matfree",
        "ksp_type": "gmres",
        "ksp_converged_reason": None,
        "ksp_atol": 1e-5,
        "ksp_rtol": 1e-5,
        "ksp_max_it": 400,
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_pc_star_sub_sub_pc_type": "lu",
        "assembled_pc_type": "python",
        "assembled_pc_python_type": "firedrake.ASMStarPC",
        "assembled_pc_star_construct_dim": 0,
        "assembled_pc_star_sub_sub_pc_factor_mat_ordering_type": "rcm",
        "assembled_pc_star_sub_sub_pc_factor_reuse_ordering": None,
        "assembled_pc_star_sub_sub_pc_factor_reuse_fill": None,
        "assembled_pc_star_sub_sub_pc_factor_fill": 1.2
    }

    return settings
