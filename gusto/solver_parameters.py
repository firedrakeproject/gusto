lines_parameters = {
    "snes_converged_reason": None,
    "mat_type": "matfree",
    "ksp_type": "gmres",
    "ksp_converged_reason": None,
    "ksp_atol": 1e-8,
    "ksp_rtol": 1e-8,
    "ksp_max_it": 400,
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "python",
    "assembled_pc_python_type": "firedrake.ASMStarPC",
    "assembled_pc_star_construct_dim": 0,
    "assembled_pc_star_sub_sub_pc_factor_mat_ordering_type": "rcm",
    "assembled_pc_star_sub_sub_pc_factor_reuse_ordering": None,
    "assembled_pc_star_sub_sub_pc_factor_reuse_fill": None,
    "assembled_pc_star_sub_sub_pc_factor_fill": 1.2,
}


hybridized_parameters = {
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

