"""
A test of the Average kernel used for the Averager.
"""

from firedrake import (Function, IntervalMesh, as_vector, Constant,
                       FunctionSpace, FiniteElement, dx)
from firedrake.parloops import par_loop, READ, INC

from gusto import kernels
import numpy as np
import pytest

def test_average():

    L = 3.0
    n = 3

    mesh = IntervalMesh(n, L)


    cell = mesh.ufl_cell().cellname()
    DG1_elt = FiniteElement("DG", cell, 1, variant="equispaced")
    DG1 = FunctionSpace(mesh, DG1_elt)
    CG1 = FunctionSpace(mesh, "CG", 1)

    # We will fill DG_field with values, and average them to CG_field
    DG_field = Function(DG1)
    CG_field = Function(CG1)
    weights = Function(CG1)

    # The numbering of DoFs for DG1 and CG1 near the origin in this mesh is
    #  |      DG1                |      |      |
    #  |1-----0|3----2|--        0------1------2

    # Let us focus on the point at (1,1)
    # For DG1 this is the DoFs numbered 0 and 3
    # For CG1 this is the DoF numbered 1
    # The test is if at CG_field[1] we get the average of the corresponding DG_field values

    DG_field.dat.data[0] = 6.0
    DG_field.dat.data[3] = -10.0

    true_value = 0.5 * (6.0 - 10.0)

    weights = Function(CG1)
    weights.dat.data[1] = 2.0
    # kernel = kernels.AverageWeightings(CG1)
    # par_loop(kernel, dx,
    #          {"w": (weights, INC)},
    #          is_loopy_kernel=True)

    kernel = kernels.Average(CG1)
    par_loop(kernel, dx,
             {"vo": (CG_field, INC),
              "v": (DG_field, READ),
              "w": (weights, READ)},
             is_loopy_kernel=True)
    print(CG_field.dat.data[:])

    tolerance = 1e-12
    assert abs(CG_field.dat.data[1] - true_value) < tolerance
