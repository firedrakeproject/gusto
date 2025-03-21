"""
This module provides some parameters sets that are good defaults
for particular kinds of system.
"""
from gusto.core.function_spaces import is_cg

__all__ = ['mass_parameters', 'hydrostatic_parameters']


def mass_parameters(V, spaces=None, ignore_vertical=True):
    """
    PETSc solver parameters for mass matrices.

    Currently this sets to a monolithic CG+ILU.

    TODO: implement field-by-field parameters that choose
          preonly for discontinuous fields and CG for continuous
          fields - see docstring below.

    ================= FUTURE DOCSTRING =================
    Any fields which are discontinuous will have block diagonal
    mass matrices, so are solved directly using:
        'ksp_type': 'preonly'
        'pc_type': 'ilu'

    All continuous fields are solved with CG, with the preconditioner
    being ILU independently on each field. By solving all continuous fields
    "monolithically", the total number of inner products is minimised, which
    is beneficial for scaling to large core counts because it minimises the
    total number of MPI_Allreduce calls.
        'ksp_type': 'cg'
        'pc_type': 'fieldsplit'
        'pc_fieldsplit_type': 'additive'
        'fieldsplit_ksp_type': 'preonly'
        'fieldsplit_pc_type': 'ilu'

    Args:
        spaces: Optional `Spaces` object. If present, any subspace
            of V that came from the `Spaces` object will use the
            continuity information from `spaces`.
            If not present, continuity is checked with `is_cg`.

        ignore_vertical: whether to include the vertical direction when checking
            field continuity on extruded meshes. If True, only the horizontal
            continuity will be considered, e.g. the standard theta space will
            be treated as discontinuous.
    """
    return {
        'ksp_type': 'cg',
        'pc_type': 'bjacobi',
        'sub_pc_type': 'ilu'
    }

    extruded = hasattr(V.mesh, "_base_mesh")

    continuous_fields = set()
    for i, Vsub in enumerate(V.subfunctions):
        # field = Vsub.name or str(i)
        field = str(i)

        if spaces is not None:
            continuous = spaces.continuity.get(field, is_cg(Vsub))
        else:
            continuous = is_cg(Vsub)

        # For extruded meshes the continuity is recorded
        # separately for the horizontal and vertical directions.
        if extruded and spaces is not None:
            if ignore_vertical:
                continuous = continuous['horizontal']
            else:
                continuous = (continuous['horizontal']
                              or continuous['vertical'])

        if continuous:
            continuous_fields.add(field)

    if len(V.subfunctions) == 1:
        parameters = {
            'ksp_type': 'cg' if all(continuous_fields) else 'preonly',
            'pc_type': 'bjacobi',
            'sub_pc_type': 'ilu',
        }
    else:

        parameters = {
            'ksp_type': 'preonly',
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'additive',
            'pc_fieldsplit_0_fields': ','.join(continuous_fields),
            'fieldsplit': {
                'ksp_type': 'preonly',
                'pc_type': 'bjacobi',
                'sub_pc_type': 'ilu'
            },
            'fieldsplit_0_ksp_type': 'cg',
        }

    return parameters


hydrostatic_parameters = {
    'mat_type': 'matfree',
    'ksp_type': 'preonly',
    'pc_type': 'python',
    'pc_python_type': 'firedrake.SCPC',
    # Velocity mass operator is singular in the hydrostatic case.
    # So for reconstruction, we eliminate rho into u
    'pc_sc_eliminate_fields': '1, 0',
    'condensed_field': {
        'ksp_type': 'fgmres',
        'ksp_rtol': 1.0e-8,
        'ksp_atol': 1.0e-8,
        'ksp_max_it': 100,
        'pc_type': 'gamg',
        'pc_gamg_sym_graph': True,
        'mg_levels': {
            'ksp_type': 'gmres',
            'ksp_max_it': 5,
            'pc_type': 'bjacobi',
            'sub_pc_type': 'ilu'
        }
    }
}
