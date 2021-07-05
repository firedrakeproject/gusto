"""
A test of the Gaussian elimination kernel used for the BoundaryRecoverer.
"""

from firedrake import (IntervalMesh, FunctionSpace, Function, RectangleMesh,
                       VectorFunctionSpace, FiniteElement, SpatialCoordinate)

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


def setup_values(geometry, field_init, field_true, act_coords, eff_coords):

    x = SpatialCoordinate(field_init.function_space().mesh())
    act_coords.interpolate(x)

    if geometry == "1D":
        # We consider the cell on the left boundary: with act coords (0) and (1)
        # The eff coordinates are (0.5) and (1)
        # Consider a true field with values 2 and -1 at these DoFs
        # This would be described by a field with f = 2 - 3*x
        # The initial values at the eff coords would then be 0.5 and -1

        set_val_at_point_DG(act_coords, 0.0, field_init, 0.5)
        set_val_at_point_DG(act_coords, 1.0, field_init, -1.0)
        set_val_at_point_DG(act_coords, 0.0, field_true, 2.0)
        set_val_at_point_DG(act_coords, 1.0, field_true, -1.0)
        set_val_at_point_DG(act_coords, 0.0, eff_coords, 0.5)
        set_val_at_point_DG(act_coords, 1.0, eff_coords, 1.0)

    elif geometry == "2D":
        # We consider the unit-square cell: with act coords (0,0), (1,0), (0,1) and (1,1)
        # The eff coordinates are (0.5,0.5), (1, 0.5), (0.5, 1) and (1, 1)
        # Consider a true field with values 2, -1, -3 and 1 at these DoFs
        # This would be described by a field with f = 2 - 3*x - 5*y + 7*x*y
        # The initial values at the eff coords would then be -0.25, 0, -1, 1

        set_val_at_point_DG(act_coords, [0.0, 0.0], field_init, -0.25)
        set_val_at_point_DG(act_coords, [1.0, 0.0], field_init, 0.0)
        set_val_at_point_DG(act_coords, [0.0, 1.0], field_init, -1.0)
        set_val_at_point_DG(act_coords, [1.0, 1.0], field_init, 1.0)
        set_val_at_point_DG(act_coords, [0.0, 0.0], field_true, 2.0)
        set_val_at_point_DG(act_coords, [1.0, 0.0], field_true, -1.0)
        set_val_at_point_DG(act_coords, [0.0, 1.0], field_true, -3.0)
        set_val_at_point_DG(act_coords, [1.0, 1.0], field_true, 1.0)
        set_val_at_point_DG(act_coords, [0.0, 0.0], eff_coords, [0.5, 0.5])
        set_val_at_point_DG(act_coords, [1.0, 0.0], eff_coords, [1.0, 0.5])
        set_val_at_point_DG(act_coords, [0.0, 1.0], eff_coords, [0.5, 1.0])
        set_val_at_point_DG(act_coords, [1.0, 1.0], eff_coords, [1.0, 1.0])

    return field_init, field_true, act_coords, eff_coords


def set_val_at_point_DG(coord_field, coords, field=None, new_value=None):
    """
    Finds the DoFs of a field at a particular coordinate. If new_value is
    provided then it also assigns all the coefficients for the field at this
    coordinate to be new_value. Otherwise the list of DoF indices is returned.
    """
    num_points = len(coord_field.dat.data[:])
    point_indices = []
    for i in range(num_points):
        # Do the coordinates at the ith point match our desired coords?
        if np.allclose(coord_field.dat.data[i], coords, rtol=1e-14):
            point_indices.append(i)
            if field is not None and new_value is not None:
                field.dat.data[i] = new_value

    if len(point_indices) == 0:
        raise ValueError('Your coordinates do not appear to match the coordinates of a DoF')

    if field is None or new_value is None:
        return point_indices


@pytest.mark.parametrize("geometry", ["1D", "2D"])
def test_gaussian_elimination(geometry, mesh):

    cell = mesh.ufl_cell().cellname()
    DG1_elt = FiniteElement("DG", cell, 1, variant="equispaced")
    DG1 = FunctionSpace(mesh, DG1_elt)
    vec_DG1 = VectorFunctionSpace(mesh, DG1_elt)

    act_coords = Function(vec_DG1)
    eff_coords = Function(vec_DG1)
    field_init = Function(DG1)
    field_true = Function(DG1)
    field_final = Function(DG1)

    # We now include things for the num of exterior values, which may be removed
    DG0 = FunctionSpace(mesh, "DG", 0)
    num_ext = Function(DG0)
    num_ext.dat.data[0] = 1.0

    # Get initial and true conditions
    field_init, field_true, act_coords, eff_coords = setup_values(geometry, field_init,
                                                                  field_true, act_coords,
                                                                  eff_coords)

    kernel = kernels.GaussianElimination(DG1)
    kernel.apply(field_init, field_final, act_coords, eff_coords, num_ext)

    tolerance = 1e-12
    assert abs(field_true.dat.data[0] - field_final.dat.data[0]) < tolerance
    assert abs(field_true.dat.data[1] - field_final.dat.data[1]) < tolerance

    if geometry == "2D":
        assert abs(field_true.dat.data[2] - field_final.dat.data[2]) < tolerance
        assert abs(field_true.dat.data[3] - field_final.dat.data[3]) < tolerance
