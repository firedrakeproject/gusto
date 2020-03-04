"""
A test of the Average kernel used for the Averager.
"""

from firedrake import (Function, RectangleMesh, as_vector,
                       VectorFunctionSpace, FiniteElement, dx)
from firedrake.parloops import par_loop, READ, INC

from gusto import kernels
import numpy as np
import pytest

def test_average():

    L = 3.0
    n = 3

    mesh = RectangleMesh(n, n, L, L, quadrilateral=True)


    cell = mesh.ufl_cell().cellname()
    DG1_elt = FiniteElement("DG", cell, 1, variant="equispaced")
    vec_DG1 = VectorFunctionSpace(mesh, DG1_elt)
    vec_CG1 = VectorFunctionSpace(mesh, "CG", 1)

    # We will fill DG_field with values, and average them to CG_field
    DG_field = Function(vec_DG1)
    CG_field = Function(vec_CG1)
    weights = Function(vec_CG1)

    # The numbering of DoFs for DG1 and CG1 near the origin in this mesh is
    #  |      DG1                |     CG1     |
    #  |-------|------|--        6------7------10
    #  |10   11|18  19|          |      |      |
    #  |       |      |          |      |      |
    #  |8     9|16  17|          |      |      |
    #  |-------|------|--        1------2------4
    #  |1     3|5    7|          |      |      |
    #  |       |      |          |      |      |
    #  |0     2|4    6|          |      |      |
    #  |-------|------|--        0------3------5---

    # Let us focus on the point at (1,1)
    # For DG1 this is the DoFs numbered 3, 5, 9 and 16
    # For CG1 this is the DoF numbered 2
    # The test is if at CG_field[2] we get the average of the corresponding DG_field values
    # We do it for both components of the vector field

    DG_field.dat.data[3] = [6.0, -3.0]
    DG_field.dat.data[5] = [-7.0, -6.0]
    DG_field.dat.data[9] = [0.0, 3.0]
    DG_field.dat.data[16] = [-9.0, -1.0]

    true_values = [0.25 * (6.0 - 7.0 + 0.0 - 9.0),
                   0.25 * (-3.0 - 6.0 + 3.0 - 1.0)]

    weights = Function(vec_CG1).interpolate(as_vector([4.0, 4.0]))
    weights.dat.data[2] = [4.0, 4.0]

    kernel = kernels.Average(vec_CG1)
    par_loop(kernel, dx,
             {"vo": (CG_field, INC),
              "v": (DG_field, READ),
              "w": (weights, READ)},
             is_loopy_kernel=True)

    tolerance = 1e-12
    assert abs(CG_field.dat.data[2][0] - true_values[0]) < tolerance
    assert abs(CG_field.dat.data[2][1] - true_values[1]) < tolerance
