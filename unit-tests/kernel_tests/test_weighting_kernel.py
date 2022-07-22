"""
A test of the AverageWeightings kernel used for the Averager.
"""

from firedrake import (IntervalMesh, Function, RectangleMesh,
                       VectorFunctionSpace, SpatialCoordinate)

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


def setup_values(geometry, true_field):

    # The true values can be determined by the number of elements
    # that the DoF is shared between.

    x = SpatialCoordinate(true_field.function_space().mesh())
    coords_CG1 = Function(true_field.function_space()).interpolate(x)

    if geometry == "1D":
        # List coords of DoFs
        edge_coords = [0.0, 3.0]
        internal_coords = [1.0, 2.0]

        for coord in edge_coords:
            set_val_at_point(coords_CG1, coord, true_field, 1.0)
        for coord in internal_coords:
            set_val_at_point(coords_CG1, coord, true_field, 2.0)

    elif geometry == "2D":

        # List coords of DoFs
        corner_coords = [[0.0, 0.0], [0.0, 3.0], [3.0, 0.0], [3.0, 3.0]]
        edge_coords = [[0.0, 1.0], [0.0, 2.0], [3.0, 1.0], [3.0, 2.0],
                       [1.0, 0.0], [2.0, 0.0], [1.0, 3.0], [2.0, 3.0]]
        internal_coords = [[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]]

        for coord in corner_coords:
            set_val_at_point(coords_CG1, coord, true_field, 1.0)
        for coord in edge_coords:
            set_val_at_point(coords_CG1, coord, true_field, 2.0)
        for coord in internal_coords:
            set_val_at_point(coords_CG1, coord, true_field, 4.0)

    return true_field


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

    vec_CG1 = VectorFunctionSpace(mesh, "CG", 1)

    # We will fill DG_field with values, and average them to CG_field
    weights = Function(vec_CG1)
    true_values = Function(vec_CG1)

    true_values = setup_values(geometry, true_values)

    kernel = kernels.AverageWeightings(vec_CG1)
    kernel.apply(weights)

    tolerance = 1e-12
    if geometry == "1D":
        for i, (weight, true) in enumerate(zip(weights.dat.data[:], true_values.dat.data[:])):
            assert abs(weight - true) < tolerance, "Weight not correct at position %i" % i
    elif geometry == "2D":
        for i, (weight, true) in enumerate(zip(weights.dat.data[:], true_values.dat.data[:])):
            for weight_j, true_j in zip(weight, true):
                assert abs(weight_j - true_j) < tolerance, "Weight not correct at position %i" % i
