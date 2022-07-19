"""
A test of the PhysicsRecoveryTop and PhysicsRecoveryBottom kernels,
which are used for the BoundaryRecoverer with the physics boundary
recovery method.
"""

from firedrake import (IntervalMesh, Function, BrokenElement, VectorElement,
                       FunctionSpace, FiniteElement, ExtrudedMesh,
                       interval, TensorProductElement, SpatialCoordinate)
from gusto import kernels
import numpy as np
import pytest


def setup_values(boundary, initial_field, true_field):
    # Initial field is in Vt
    # True field is in Vt_brok
    mesh = initial_field.function_space().mesh()
    x = SpatialCoordinate(mesh)
    Vec_Vt = FunctionSpace(mesh, VectorElement(initial_field.function_space().ufl_element()))
    Vec_Vt_brok = FunctionSpace(mesh, VectorElement(true_field.function_space().ufl_element()))
    coords_Vt = Function(Vec_Vt).interpolate(x)
    coords_Vt_brok = Function(Vec_Vt_brok).interpolate(x)

    # The DoFs of the Vt and Vt_brok for the mesh are laid out as follows:
    # that we use are numbered as follows:
    #
    #       Vt_brok                   Vt
    #  ------------------    ---o-----o-----o---
    # |  o  |  o  |  o  |    |     |     |     |
    # |     |     |     |    |     |     |     |
    # |  o  |  o  |  o  |    |     |     |     |
    #  -----|-----|------    ---o-----o-----o---
    # |  o  |  o  |  o  |    |     |     |     |
    # |     |     |     |    |     |     |     |
    # |  o  |  o  |  o  |    |     |     |     |
    #  -----|-----|------    ---o-----o-----o---
    # |  o  |  o  |  o  |    |     |     |     |
    # |     |     |     |    |     |     |     |
    # |  o  |  o  |  o  |    |     |     |     |
    # -------------------    ---o-----o-----o---

    # Set initial and true values for the central column, top and bottom layers
    if boundary == "top":
        set_val_at_point(coords_Vt, [1.5, 2.0], initial_field, 1.0)
        set_val_at_point(coords_Vt, [1.5, 3.0], initial_field, 2.0)
        set_val_at_point(coords_Vt_brok, [1.5, 3.0], true_field, 3.0)
        boundary_index = set_val_at_point(coords_Vt_brok, [1.5, 3.0])
    elif boundary == "bottom":
        set_val_at_point(coords_Vt, [1.5, 0.0], initial_field, 1.0)
        set_val_at_point(coords_Vt, [1.5, 1.0], initial_field, 2.0)
        set_val_at_point(coords_Vt_brok, [1.5, 0.0], true_field, 0.0)
        boundary_index = set_val_at_point(coords_Vt_brok, [1.5, 0.0])

    return initial_field, true_field, boundary_index


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


@pytest.mark.parametrize("boundary", ["top", "bottom"])
def test_physics_recovery_kernels(boundary):

    m = IntervalMesh(3, 3)
    mesh = ExtrudedMesh(m, layers=3, layer_height=1.0)

    cell = m.ufl_cell().cellname()
    hori_elt = FiniteElement("DG", cell, 0)
    vert_elt = FiniteElement("CG", interval, 1)
    theta_elt = TensorProductElement(hori_elt, vert_elt)
    Vt = FunctionSpace(mesh, theta_elt)
    Vt_brok = FunctionSpace(mesh, BrokenElement(theta_elt))

    initial_field = Function(Vt)
    true_field = Function(Vt_brok)
    new_field = Function(Vt_brok)

    initial_field, true_field, boundary_index = setup_values(boundary, initial_field, true_field)

    kernel = kernels.PhysicsRecoveryTop() if boundary == "top" else kernels.PhysicsRecoveryBottom()
    kernel.apply(new_field, initial_field)

    tolerance = 1e-12
    assert abs(true_field.dat.data[boundary_index] - new_field.dat.data[boundary_index]) < tolerance, \
        "Value at %s from physics recovery is not correct" % boundary
