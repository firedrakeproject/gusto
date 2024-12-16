"""
A test of the MinKernel kernel, which finds the global minimum of a field.
"""

from firedrake import UnitSquareMesh, Function, FunctionSpace, SpatialCoordinate
from gusto import kernels
import numpy as np


def test_min_kernel():

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

    # Set a minimum value
    min_val = -400.18
    field.dat.data[5] = min_val

    # ------------------------------------------------------------------------ #
    # Apply kernel
    # ------------------------------------------------------------------------ #

    kernel = kernels.MinKernel()
    new_min = kernel.apply(field)

    # ------------------------------------------------------------------------ #
    # Check values
    # ------------------------------------------------------------------------ #

    assert np.isclose(new_min, min_val), 'Minimum kernel is not correct'
