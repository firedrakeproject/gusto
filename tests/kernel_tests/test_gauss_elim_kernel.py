"""
A test of the Gaussian elimination kernel used for the BoundaryRecoverer.
"""

from firedrake import (IntervalMesh, FunctionSpace, Function, RectangleMesh,
                       VectorFunctionSpace, FiniteElement)

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


def setup_values(geometry, field_init, field_true,
                 act_coords, eff_coords):

    if geometry == "1D":
        # We consider the cell on the left boundary: with act coords (0) and (1)
        # The eff coordinates are (0.5) and (1)
        # Consider a true field with values 2 and -1 at these DoFs
        # This would be described by a field with f = 2 - 3*x
        # The initial values at the eff coords would then be 0.5 and -1

        field_init.dat.data[0] = 0.5
        field_init.dat.data[1] = -1.0
        field_true.dat.data[0] = 2.0
        field_true.dat.data[1] = -1.0
        act_coords.dat.data[0] = 0.0
        act_coords.dat.data[1] = 1.0
        eff_coords.dat.data[0] = 0.5
        eff_coords.dat.data[1] = 1.0

    elif geometry == "2D":
        # We consider the unit-square cell: with act coords (0,0), (1,0), (0,1) and (1,1)
        # The eff coordinates are (0.5,0.5), (1, 0.5), (0.5, 1) and (1, 1)
        # Consider a true field with values 2, -1, -3 and 1 at these DoFs
        # This would be described by a field with f = 2 - 3*x - 5*y + 7*x*y
        # The initial values at the eff coords would then be -0.25, 0, -1, 1

        field_init.dat.data[0] = -0.25
        field_init.dat.data[1] = 0.0
        field_init.dat.data[2] = -1.0
        field_init.dat.data[3] = 1.0
        field_true.dat.data[0] = 2.0
        field_true.dat.data[1] = -1.0
        field_true.dat.data[2] = -3.0
        field_true.dat.data[3] = 1.0
        act_coords.dat.data[0, :] = [0.0, 0.0]
        act_coords.dat.data[1, :] = [1.0, 0.0]
        act_coords.dat.data[2, :] = [0.0, 1.0]
        act_coords.dat.data[3, :] = [1.0, 1.0]
        eff_coords.dat.data[0, :] = [0.5, 0.5]
        eff_coords.dat.data[1, :] = [1.0, 0.5]
        eff_coords.dat.data[2, :] = [0.5, 1.0]
        eff_coords.dat.data[3, :] = [1.0, 1.0]

    return field_init, field_true, act_coords, eff_coords


@pytest.mark.parametrize("geometry", ["1D", "2D"])
@pytest.mark.xfail
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
