"""
A test of the ClipZero kernel, which is used to enforce non-negativity.
"""

from firedrake import UnitSquareMesh, Function, FunctionSpace
from gusto import kernels
import numpy as np


def test_clip_zero():

    # ------------------------------------------------------------------------ #
    # Set up meshes and spaces
    # ------------------------------------------------------------------------ #

    mesh = UnitSquareMesh(3, 3)

    DG1 = FunctionSpace(mesh, "DG", 1)

    field = Function(DG1)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    field.dat.data[:] = -3.0

    # ------------------------------------------------------------------------ #
    # Apply kernel
    # ------------------------------------------------------------------------ #

    kernel = kernels.ClipZero(DG1)
    kernel.apply(field, field)

    # ------------------------------------------------------------------------ #
    # Check values
    # ------------------------------------------------------------------------ #

    assert np.all(field.dat.data >= 0.0), \
        'ClipZero kernel is not enforcing non-negativity'
