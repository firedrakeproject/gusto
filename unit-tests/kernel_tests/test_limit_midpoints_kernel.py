"""
A test of the LimitMidpoints kernel, which is used for the ThetaLimiter to limit
the transport of variables in the degree = 1 temperature space.

This makes a degree = 1 temperature space on a 1-layer extruded mesh. The
initial values at the midpoint DoFs exceed those at the top and bottom of the
cells.
"""

from firedrake import (IntervalMesh, Function, BrokenElement,
                       FunctionSpace, FiniteElement, ExtrudedMesh,
                       interval, TensorProductElement, SpatialCoordinate)
from gusto import kernels
import numpy as np
import pytest


@pytest.mark.parametrize("profile", ["overshoot", "undershoot", "linear"])
def test_limit_midpoints(profile):

    # ------------------------------------------------------------------------ #
    # Set up meshes and spaces
    # ------------------------------------------------------------------------ #

    m = IntervalMesh(3, 3)
    mesh = ExtrudedMesh(m, layers=1, layer_height=3.0)

    cell = m.ufl_cell().cellname()
    DG_hori_elt = FiniteElement("DG", cell, 1, variant='equispaced')
    DG_vert_elt = FiniteElement("DG", interval, 1, variant='equispaced')
    Vt_vert_elt = FiniteElement("CG", interval, 2)
    DG_elt = TensorProductElement(DG_hori_elt, DG_vert_elt)
    theta_elt = TensorProductElement(DG_hori_elt, Vt_vert_elt)
    Vt_brok = FunctionSpace(mesh, BrokenElement(theta_elt))
    DG1 = FunctionSpace(mesh, DG_elt)

    new_field = Function(Vt_brok)
    init_field = Function(Vt_brok)
    DG1_field = Function(DG1)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    _, z = SpatialCoordinate(mesh)

    if profile == 'undershoot':
        # A quadratic whose midpoint is lower than the top and bottom values
        init_expr = (80./9.)*z**2 - 20.*z + 300.
    elif profile == 'overshoot':
        # A quadratic whose midpoint is higher than the top and bottom values
        init_expr = (-80./9.)*z**2 + (100./3)*z + 300.
    elif profile == 'linear':
        # Linear profile which must be unchanged
        init_expr = (20./3.)*z + 300.
    else:
        raise NotImplementedError

    # Linear DG field has the same values at top and bottom as quadratic
    DG_expr = (20./3.)*z + 300.

    init_field.interpolate(init_expr)
    DG1_field.interpolate(DG_expr)

    # ------------------------------------------------------------------------ #
    # Apply kernel
    # ------------------------------------------------------------------------ #

    kernel = kernels.LimitMidpoints(Vt_brok)
    kernel.apply(new_field, DG1_field, init_field)

    # ------------------------------------------------------------------------ #
    # Check values
    # ------------------------------------------------------------------------ #

    tol = 1e-12
    assert np.max(new_field.dat.data) <= np.max(init_field.dat.data) + tol, \
        'LimitMidpoints kernel is giving an overshoot'
    assert np.min(new_field.dat.data) >= np.min(init_field.dat.data) - tol, \
        'LimitMidpoints kernel is giving an undershoot'

    if profile == 'linear':
        assert np.allclose(init_field.dat.data, new_field.dat.data), \
            'For a profile with no maxima or minima, the LimitMidpoints ' + \
            'kernel should leave the field unchanged'
