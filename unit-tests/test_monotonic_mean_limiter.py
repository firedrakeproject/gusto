"""
Tests the MonotonicMeanLimiter, which blends a DG1 mixing ratio field with
its DG0 mean companion field so that the result respects monotonicity
(rather than just non-negativity), following the derivation in
monotone_limiter.tex.

The test checks two properties:
  1. Monotonicity: in every cell, the limited field lies within the min/max
     of the pre-transport field over that cell and its facet-neighbours.
  2. Mass conservation: since the "mean" field is constructed here to be
     exactly the cell-average of the "new" (post-transport) field, blending
     with any lamda in [0, 1] must preserve the mass (integral) in every
     cell, and hence the total mass.
"""

import numpy as np
from firedrake import (
    PeriodicIntervalMesh, FunctionSpace, Function, FiniteElement, assemble, dx
)
from gusto import MonotonicMeanLimiter


def build_adjacency(CG1):
    """
    Returns, for each cell, the set of other cell indices that share a
    vertex with it (i.e. the facet-neighbours in 1D).
    """
    cell_node_list = CG1.cell_node_list
    ncells = cell_node_list.shape[0]
    vertex_to_cells = {}
    for c in range(ncells):
        for v in cell_node_list[c]:
            vertex_to_cells.setdefault(v, []).append(c)

    adjacency = [set() for _ in range(ncells)]
    for cells_sharing_vertex in vertex_to_cells.values():
        for c in cells_sharing_vertex:
            for c2 in cells_sharing_vertex:
                if c2 != c:
                    adjacency[c].add(c2)
    return adjacency


def test_monotonic_mean_limiter():

    ncells = 6
    mesh = PeriodicIntervalMesh(ncells, float(ncells))

    DG0 = FunctionSpace(mesh, "DG", 0)
    # Use the equispaced variant so that DOFs correspond to vertex values,
    # matching what MonotonicMeanLimiter uses internally
    cell = mesh.ufl_cell().cellname
    DG1_element = FiniteElement("DG", cell, 1, variant="equispaced")
    DG1 = FunctionSpace(mesh, DG1_element)
    CG1 = FunctionSpace(mesh, "CG", 1)

    adjacency = build_adjacency(CG1)
    assert all(len(a) == 2 for a in adjacency), \
        "Expected each cell on this periodic 1D mesh to have 2 neighbours"

    # ---------------------------------------------------------------------- #
    # Old (pre-transport) field: a "spike" shape, constant in each cell.
    # This defines the monotonic bounds.
    # ---------------------------------------------------------------------- #
    old_values = np.array([0.0, 0.0, 2.0, 4.0, 2.0, 0.0])
    old_field = Function(DG1)
    for c in range(ncells):
        old_field.dat.data[DG1.cell_node_list[c]] = old_values[c]

    expected_min = np.array([
        min([old_values[c]] + [old_values[n] for n in adjacency[c]])
        for c in range(ncells)
    ])
    expected_max = np.array([
        max([old_values[c]] + [old_values[n] for n in adjacency[c]])
        for c in range(ncells)
    ])

    # ---------------------------------------------------------------------- #
    # New (post-transport, pre-limiting) field. This is allowed to vary
    # linearly within a cell. Cells 1 and 3 are deliberately set to
    # undershoot/overshoot the bounds implied by the old field, while the
    # others stay within bounds and should be left unaffected.
    # ---------------------------------------------------------------------- #
    new_left_right = {
        0: (0.0, 0.0),
        1: (0.5, -0.3),   # undershoots expected_min[1] == 0.0, average stays in bounds
        2: (2.0, 2.0),
        3: (3.0, 4.8),    # overshoots expected_max[3] == 4.0, average stays in bounds
        4: (2.0, 2.0),
        5: (0.0, 0.0),
    }
    new_field = Function(DG1)
    for c in range(ncells):
        dofs = DG1.cell_node_list[c]
        left_val, right_val = new_left_right[c]
        new_field.dat.data[dofs[0]] = left_val
        new_field.dat.data[dofs[1]] = right_val

    # The mean field is set to the exact cell-average of the new field, as
    # would be the case for a genuinely mass-consistent low-order companion
    # field.
    mean_values = np.array(
        [0.5*(new_left_right[c][0] + new_left_right[c][1]) for c in range(ncells)]
    )
    mean_field = Function(DG0)
    for c in range(ncells):
        mean_field.dat.data[DG0.cell_node_list[c]] = mean_values[c]

    total_mass_before = assemble(new_field*dx)

    # ---------------------------------------------------------------------- #
    # Apply the limiter
    # ---------------------------------------------------------------------- #
    limiter = MonotonicMeanLimiter([DG1])
    mX_fields = [new_field]
    mean_fields = [mean_field]
    old_fields = [old_field]
    limiter.apply(mX_fields, mean_fields, old_fields)

    limited_field = mX_fields[0]

    # ---------------------------------------------------------------------- #
    # Check monotonicity
    # ---------------------------------------------------------------------- #
    tol = 1e-10
    for c in range(ncells):
        cell_vals = limited_field.dat.data[DG1.cell_node_list[c]]
        assert np.min(cell_vals) >= expected_min[c] - tol, \
            f"Cell {c}: limited field undershoots its monotonic bound"
        assert np.max(cell_vals) <= expected_max[c] + tol, \
            f"Cell {c}: limited field overshoots its monotonic bound"

    # Cells that were already within bounds should be unaffected
    for c in [0, 2, 4, 5]:
        cell_vals = limited_field.dat.data[DG1.cell_node_list[c]]
        expected_vals = np.array(new_left_right[c])
        np.testing.assert_allclose(np.sort(cell_vals), np.sort(expected_vals), atol=1e-10)

    # The violating cells should actually have been changed
    for c in [1, 3]:
        cell_vals = limited_field.dat.data[DG1.cell_node_list[c]]
        expected_vals = np.array(new_left_right[c])
        assert not np.allclose(np.sort(cell_vals), np.sort(expected_vals)), \
            f"Cell {c}: limiter should have modified this violating cell"

    # ---------------------------------------------------------------------- #
    # Check mass conservation (global integral)
    # ---------------------------------------------------------------------- #
    total_mass_after = assemble(limited_field*dx)
    assert np.isclose(total_mass_before, total_mass_after, atol=1e-10), \
        "MonotonicMeanLimiter did not conserve total mass"

    # Since mean_field was constructed as the exact cell-average of
    # new_field, mass should also be conserved on a per-cell basis,
    # regardless of the blending weight used in each cell.
    for c in range(ncells):
        cell_length = 1.0  # PeriodicIntervalMesh(ncells, ncells) -> unit cells
        mass_before = mean_values[c] * cell_length
        cell_vals = limited_field.dat.data[DG1.cell_node_list[c]]
        mass_after = 0.5*(cell_vals[0] + cell_vals[1]) * cell_length
        assert np.isclose(mass_before, mass_after, atol=1e-10), \
            f"Cell {c}: mass not conserved locally"
