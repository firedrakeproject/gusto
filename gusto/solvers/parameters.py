"""
This module provides some parameters sets that are good defaults
for particular kinds of system.
"""
from gusto.time_discretisation.wrappers import is_cg

__all__ = ['mass_parameters']


def mass_parameters(V, spaces=None, ignore_vertical=True):
    """
    PETSc solver parameters for mass matrices.

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

    extruded = hasattr(V.mesh, "_base_mesh")

    continuous_fields = set()
    for i, Vsub in V.subfunctions:
        field = Vsub.name or str(i)

        if spaces is not None:
            continuous = spaces.continuity.get(field, is_cg(Vsub))
        else:
            continuous = is_cg(Vsub)

        # For extruded meshes the continuity is recorded
        # separately for the horizontal and vertical directions.
        if extruded:
            if ignore_vertical:
                continuous = continuous['horizontal']
            else:
                continuous = (continuous['horizontal']
                              and continuous['vertical'])

        if continuous:
            continuous_fields.add(field)

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
