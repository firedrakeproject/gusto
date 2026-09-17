"""
Tests the kernels used by the MonotonicMeanLimiter:
  - MeanMixingRatioStencilBounds, which gathers the min/max of a field over
    each cell and its facet-neighbours.
  - MonotonicMeanMixingRatioWeights, which computes the blending weight
    (lamda) needed to keep a transported field within those bounds.
"""

import numpy as np
from firedrake import (
    PeriodicIntervalMesh, FunctionSpace, Function, FiniteElement
)
from gusto import kernels


def build_vertex_to_cells(CG1):
    """
    Returns a dict mapping each CG1 (global vertex) dof index to the list of
    cell indices that share that vertex.
    """
    cell_node_list = CG1.cell_node_list
    ncells = cell_node_list.shape[0]
    vertex_to_cells = {}
    for c in range(ncells):
        for v in cell_node_list[c]:
            vertex_to_cells.setdefault(v, []).append(c)
    return vertex_to_cells


def setup_mesh_and_spaces():
    ncells = 6
    mesh = PeriodicIntervalMesh(ncells, float(ncells))

    DG0 = FunctionSpace(mesh, "DG", 0)
    # Use the equispaced variant so that DOFs correspond to vertex values,
    # matching what MonotonicMeanLimiter uses internally
    cell = mesh.ufl_cell().cellname
    DG1_element = FiniteElement("DG", cell, 1, variant="equispaced")
    DG1 = FunctionSpace(mesh, DG1_element)
    CG1 = FunctionSpace(mesh, "CG", 1)

    return ncells, DG0, DG1, CG1


def test_mean_mixing_ratio_stencil_bounds():

    ncells, DG0, DG1, CG1 = setup_mesh_and_spaces()
    vertex_to_cells = build_vertex_to_cells(CG1)

    old_values = np.array([0.0, 0.0, 2.0, 4.0, 2.0, 0.0])
    field = Function(DG1)
    for c in range(ncells):
        field.dat.data[DG1.cell_node_list[c]] = old_values[c]

    # Expected min/max at each vertex: the min/max of "old_values" over all
    # cells sharing that vertex
    expected_vertex_min = {}
    expected_vertex_max = {}
    for v, cells in vertex_to_cells.items():
        expected_vertex_min[v] = min(old_values[c] for c in cells)
        expected_vertex_max[v] = max(old_values[c] for c in cells)

    stencil_min_cg = Function(CG1)
    stencil_max_cg = Function(CG1)
    stencil_min_cg.assign(1.0e10)
    stencil_max_cg.assign(-1.0e10)

    kernel = kernels.MeanMixingRatioStencilBounds(DG1)
    kernel.apply(stencil_min_cg, stencil_max_cg, field)

    stencil_min_dg1 = Function(DG1).interpolate(stencil_min_cg)
    stencil_max_dg1 = Function(DG1).interpolate(stencil_max_cg)

    # For a degree 1 Lagrange element, the local dof ordering convention on
    # the reference cell is the same for CG1 and (equispaced) DG1, so local
    # dof k of cell c in DG1 corresponds to the same vertex as local dof k
    # of cell c in CG1.
    for c in range(ncells):
        dg1_dofs = DG1.cell_node_list[c]
        cg1_dofs = CG1.cell_node_list[c]
        for k in range(2):
            v = cg1_dofs[k]
            dof = dg1_dofs[k]
            assert np.isclose(stencil_min_dg1.dat.data[dof], expected_vertex_min[v]), \
                f"Cell {c}: incorrect stencil minimum at a vertex"
            assert np.isclose(stencil_max_dg1.dat.data[dof], expected_vertex_max[v]), \
                f"Cell {c}: incorrect stencil maximum at a vertex"


def test_monotonic_mean_mixing_ratio_weights():

    ncells, DG0, DG1, CG1 = setup_mesh_and_spaces()

    # Directly prescribe the stencil bounds (old_min, old_max), the
    # post-transport field (new_field, constant per cell) and the mean
    # field, to test the kernel's lamda formula in isolation.
    old_min_vals = np.array([0.0, 0.0, 2.0, 2.0, 0.0, 0.0])
    old_max_vals = np.array([0.0, 2.0, 4.0, 4.0, 2.0, 2.0])
    new_vals = np.array([0.0, -0.5, 2.0, 4.8, 2.0, 0.0])
    mean_vals = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 0.0])

    stencil_min = Function(DG1)
    stencil_max = Function(DG1)
    new_field = Function(DG1)
    for c in range(ncells):
        dofs = DG1.cell_node_list[c]
        stencil_min.dat.data[dofs] = old_min_vals[c]
        stencil_max.dat.data[dofs] = old_max_vals[c]
        new_field.dat.data[dofs] = new_vals[c]

    mean_field = Function(DG0)
    for c in range(ncells):
        mean_field.dat.data[DG0.cell_node_list[c]] = mean_vals[c]

    lamda = Function(DG0)
    lamda.assign(0.0)

    kernel = kernels.MonotonicMeanMixingRatioWeights(DG1)
    kernel.apply(lamda, new_field, stencil_min, stencil_max, mean_field)

    # Expected lamda values, computed by hand from:
    #   lamda_min = (old_min - new_min) / (mean - new_min), if new_min < old_min
    #   lamda_max = (new_max - old_max) / (new_max - mean), if new_max > old_max
    #   lamda = clip(max(lamda_min, lamda_max), 0, 1)
    expected_lamda = np.zeros(ncells)
    for c in range(ncells):
        lamda_min = 0.0
        lamda_max = 0.0
        if new_vals[c] < old_min_vals[c]:
            lamda_min = (old_min_vals[c] - new_vals[c]) / (mean_vals[c] - new_vals[c])
        if new_vals[c] > old_max_vals[c]:
            lamda_max = (new_vals[c] - old_max_vals[c]) / (new_vals[c] - mean_vals[c])
        expected_lamda[c] = min(max(max(lamda_min, lamda_max), 0.0), 1.0)

    for c in range(ncells):
        computed = lamda.dat.data[DG0.cell_node_list[c]][0]
        assert np.isclose(computed, expected_lamda[c], atol=1e-8), \
            f"Cell {c}: incorrect lamda, got {computed}, expected {expected_lamda[c]}"

    # Cells 1 and 3 should have genuinely fractional weights (not just 0 or 1)
    assert 0.0 < lamda.dat.data[DG0.cell_node_list[1]][0] < 1.0
    assert 0.0 < lamda.dat.data[DG0.cell_node_list[3]][0] < 1.0

    # Cells within bounds should have zero weight
    for c in [0, 2, 4, 5]:
        assert np.isclose(lamda.dat.data[DG0.cell_node_list[c]][0], 0.0, atol=1e-12)
