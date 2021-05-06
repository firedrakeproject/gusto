"""
A test of the PhysicsRecoveryTop and PhysicsRecoveryBottom kernels,
which are used for the BoundaryRecoverer with the physics boundary
recovery method.
"""

from firedrake import (IntervalMesh, Function, BrokenElement,
                       FunctionSpace, FiniteElement, ExtrudedMesh,
                       interval, TensorProductElement)
from gusto import kernels
import pytest


def setup_values(boundary, initial_field, true_field):
    # Initial field is in Vt
    # True field is in Vt_brok

    # The DoFs of the Vt and Vt_brok for the mesh
    # that we use are numbered as follows:
    #
    #       Vt_brok                   Vt
    #  ------------------    ---3-----7-----11--
    # |  5  |  11 |  17 |    |     |     |     |
    # |     |     |     |    |     |     |     |
    # |  4  |  10 |  16 |    |     |     |     |
    #  -----|-----|------    ---2-----6-----10--
    # |  3  |  9  |  15 |    |     |     |     |
    # |     |     |     |    |     |     |     |
    # |  2  |  8  |  14 |    |     |     |     |
    #  -----|-----|------    ---1-----5-----9---
    # |  1  |  7  |  13 |    |     |     |     |
    # |     |     |     |    |     |     |     |
    # |  0  |  6  |  12 |    |     |     |     |
    # -------------------    ---0-----4-----8---

    # We put in values for the top/bottom boundaries
    if boundary == "top":
        initial_field.dat.data[6] = 1.0
        initial_field.dat.data[7] = 2.0
        true_field.dat.data[11] = 3.0
    elif boundary == "bottom":
        initial_field.dat.data[4] = 1.0
        initial_field.dat.data[5] = 2.0
        true_field.dat.data[6] = 0.0

    return initial_field, true_field


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

    initial_field, true_field = setup_values(boundary, initial_field, true_field)

    kernel = kernels.PhysicsRecoveryTop() if boundary == "top" else kernels.PhysicsRecoveryBottom()
    kernel.apply(new_field, initial_field)

    tolerance = 1e-12
    index = 11 if boundary == "top" else 6
    assert abs(true_field.dat.data[index] - new_field.dat.data[index]) < tolerance, \
        "Value at %s from physics recovery is not correct" % boundary
