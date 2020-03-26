"""
A test of the Average kernel used for the Averager.
"""

from firedrake import (IntervalMesh, Function, RectangleMesh,
                       VectorFunctionSpace, FiniteElement, dx)
from firedrake.parloops import par_loop, READ, INC

from gusto import kernels
import pytest


@pytest.fixture
def mesh(geometry):

    L = 3.0
    n = 3

    if geometry == "1D":
        m = IntervalMesh(n, L)
    elif geometry == "2D":
        m = RectangleMesh(n, n, L, L, quadrilateral=True)

    return m


def setup_values(geometry, DG_field, weights):

    if geometry == "1D":
        # The numbering of DoFs for DG1 and CG1 near the origin in this mesh is
        #  |      DG1                |      |      |
        #  |1-----0|3----2|--        0------1------2

        # Let us focus on the point at (1,1)
        # For DG1 this is the DoFs numbered 0 and 3
        # For CG1 this is the DoF numbered 1
        # The test is if at CG_field[1] we get the average of the corresponding DG_field values

        DG_field.dat.data[0] = 6.0
        DG_field.dat.data[3] = -10.0

        true_values = 0.5 * (6.0 - 10.0)

        weights.dat.data[1] = 2.0

    elif geometry == "2D":
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

        weights.dat.data[2] = [4.0, 4.0]

    return DG_field, weights, true_values


@pytest.mark.parametrize("geometry", ["1D", "2D"])
def test_average(geometry, mesh):

    cell = mesh.ufl_cell().cellname()
    DG1_elt = FiniteElement("DG", cell, 1, variant="equispaced")
    vec_DG1 = VectorFunctionSpace(mesh, DG1_elt)
    vec_CG1 = VectorFunctionSpace(mesh, "CG", 1)

    # We will fill DG_field with values, and average them to CG_field
    DG_field = Function(vec_DG1)
    CG_field = Function(vec_CG1)
    weights = Function(vec_CG1)

    DG_field, weights, true_values = setup_values(geometry, DG_field, weights)

    kernel = kernels.Average(vec_CG1)
    par_loop(kernel, dx,
             {"vo": (CG_field, INC),
              "v": (DG_field, READ),
              "w": (weights, READ)},
             is_loopy_kernel=True)

    tolerance = 1e-12
    if geometry == "1D":
        assert abs(CG_field.dat.data[1] - true_values) < tolerance
    elif geometry == "2D":
        assert abs(CG_field.dat.data[2][0] - true_values[0]) < tolerance
        assert abs(CG_field.dat.data[2][1] - true_values[1]) < tolerance
