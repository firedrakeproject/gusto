"""
A test of the MaxKernel kernel, which finds the global maximum of a field.
"""

from firedrake import UnitSquareMesh, Function, FunctionSpace, SpatialCoordinate
from gusto import kernels
import numpy as np


def test_max_kernel():

    # ------------------------------------------------------------------------ #
    # Set up meshes and spaces
    # ------------------------------------------------------------------------ #

    mesh = UnitSquareMesh(3, 3)

    DG1 = FunctionSpace(mesh, "DG", 1)

    field = Function(DG1)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    x, y = SpatialCoordinate(mesh)

    # Some random expression
    init_expr = (20./3.)*x*y + 300.
    field.interpolate(init_expr)

    # Set a maximum value
    max_val = 40069.18
    field.dat.data[5] = max_val

    # ------------------------------------------------------------------------ #
    # Apply kernel
    # ------------------------------------------------------------------------ #

    kernel = kernels.MaxKernel()
    new_max = kernel.apply(field)

    # ------------------------------------------------------------------------ #
    # Check values
    # ------------------------------------------------------------------------ #

    assert np.isclose(new_max, max_val), 'maximum kernel is not correct'
