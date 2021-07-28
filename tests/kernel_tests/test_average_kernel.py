"""
A test of the Average kernel used for the Averager.
"""

from firedrake import (IntervalMesh, Function, RectangleMesh, SpatialCoordinate,
                       VectorFunctionSpace, FiniteElement)

from gusto import kernels
import numpy as np
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


def setup_values(geometry, DG0_field, weights):

    x = SpatialCoordinate(weights.function_space().mesh())
    coords_CG1 = Function(weights.function_space()).interpolate(x)
    coords_DG0 = Function(DG0_field.function_space()).interpolate(x)

    if geometry == "1D":
        # Let us focus on the point at x = 1.0
        # The test is if at CG_field[CG_index] we get the average of the corresponding DG_field values
        CG_index = set_val_at_point(coords_CG1, 1.0)
        set_val_at_point(coords_DG0, 0.5, DG0_field, 6.0)
        set_val_at_point(coords_DG0, 1.5, DG0_field, -10.0)
        set_val_at_point(coords_CG1, 1.0, weights, 2.0)

        true_values = 0.5 * (6.0 - 10.0)

    elif geometry == "2D":
        # Let us focus on the point at (1,1)
        # The test is if at CG_field[CG_index] we get the average of the corresponding DG_field values
        # We do it for both components of the vector field

        CG_index = set_val_at_point(coords_CG1, [1.0, 1.0])
        set_val_at_point(coords_CG1, [1.0, 1.0], weights, [4.0, 4.0])
        set_val_at_point(coords_DG0, [0.5, 0.5], DG0_field, [6.0, -3.0])
        set_val_at_point(coords_DG0, [1.5, 0.5], DG0_field, [-7.0, -6.0])
        set_val_at_point(coords_DG0, [0.5, 1.5], DG0_field, [0.0, 3.0])
        set_val_at_point(coords_DG0, [1.5, 1.5], DG0_field, [-9.0, -1.0])

        true_values = [0.25 * (6.0 - 7.0 + 0.0 - 9.0),
                       0.25 * (-3.0 - 6.0 + 3.0 - 1.0)]

    return DG0_field, weights, true_values, CG_index


def set_val_at_point(coord_field, coords, field=None, new_value=None):
    """
    Finds the DoF of a field at a particular coordinate. If new_value is
    provided then it also assigns the coefficient for the field there to be
    new_value. Otherwise the DoF index is returned.
    """
    num_points = len(coord_field.dat.data[:])
    point_found = False

    for i in range(num_points):
        # Do the coordinates at the ith point match our desired coords?
        if np.allclose(coord_field.dat.data[i], coords, rtol=1e-14):
            point_found = True
            point_index = i
            if field is not None and new_value is not None:
                field.dat.data[i] = new_value
            break

    if not point_found:
        raise ValueError('Your coordinates do not appear to match the coordinates of a DoF')

    if field is None or new_value is None:
        return point_index


@pytest.mark.parametrize("geometry", ["1D", "2D"])
def test_average(geometry, mesh):

    cell = mesh.ufl_cell().cellname()
    DG1_elt = FiniteElement("DG", cell, 1, variant="equispaced")
    vec_DG1 = VectorFunctionSpace(mesh, DG1_elt)
    vec_DG0 = VectorFunctionSpace(mesh, "DG", 0)
    vec_CG1 = VectorFunctionSpace(mesh, "CG", 1)

    # We will fill DG1_field with values, and average them to CG_field
    # First need to put the values into DG0 and then interpolate
    DG0_field = Function(vec_DG0)
    DG1_field = Function(vec_DG1)
    CG_field = Function(vec_CG1)
    weights = Function(vec_CG1)

    DG0_field, weights, true_values, CG_index = setup_values(geometry, DG0_field, weights)

    DG1_field.interpolate(DG0_field)
    kernel = kernels.Average(vec_CG1)
    kernel.apply(CG_field, weights, DG1_field)

    tolerance = 1e-12
    if geometry == "1D":
        assert abs(CG_field.dat.data[CG_index] - true_values) < tolerance
    elif geometry == "2D":
        assert abs(CG_field.dat.data[CG_index][0] - true_values[0]) < tolerance
        assert abs(CG_field.dat.data[CG_index][1] - true_values[1]) < tolerance
