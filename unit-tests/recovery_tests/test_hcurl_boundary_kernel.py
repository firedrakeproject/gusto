"""
A test of the BoundaryRecoveryHCurl kernel, which is used for the
BoundaryRecoverer on extruded meshes with HCurl fields.
"""

from firedrake import (PeriodicIntervalMesh, PeriodicRectangleMesh,
                       ExtrudedMesh, FiniteElement, HCurl, interval, Function,
                       TensorProductElement, FunctionSpace, VectorFunctionSpace,
                       SpatialCoordinate, as_vector, conditional)
from gusto.recovery.recovery_kernels import BoundaryRecoveryHCurl
import numpy as np
import pytest


def set_up_mesh(element, height):

    length = 3.0
    ncolumns = 3
    nlayers = 3
    deltaz = height / nlayers

    if element == 'interval':
        m = PeriodicIntervalMesh(ncolumns, length)
    else:
        quad = (element == 'quadrilateral')
        m = PeriodicRectangleMesh(ncolumns, ncolumns, length, length,
                                  direction='both', quadrilateral=quad)

    extruded_mesh = ExtrudedMesh(m, layers=nlayers, layer_height=deltaz)

    return extruded_mesh


def set_up_function_space(element, mesh):

    if element == 'interval':
        family = 'DG'
        degree = 0
    elif element == 'quadrilateral':
        family = 'RTCE'
        degree = 1
    elif element == 'triangular':
        family = 'RTE'
        degree = 1

    cell = mesh._base_mesh.ufl_cell().cellname()

    u_hori = FiniteElement(family, cell, degree)
    w_hori = FiniteElement("CG", cell, 1)
    u_vert = FiniteElement("CG", interval, 1)
    w_vert = FiniteElement("DG", interval, 0)

    u_element = HCurl(TensorProductElement(u_hori, u_vert))
    w_element = HCurl(TensorProductElement(w_hori, w_vert))
    v_element = u_element + w_element

    V = FunctionSpace(mesh, v_element)

    return V


def set_up_fields(element):

    height = 3.0

    mesh = set_up_mesh(element, height)
    V = set_up_function_space(element, mesh)
    initial_field = Function(V)
    true_field = Function(V)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    ux0 = 3.0
    dux_dz = -2.0
    uz0 = -100.0

    # The true expression is linearly varying with height
    if element == 'interval':
        _, z = SpatialCoordinate(mesh)
        true_expr = as_vector([ux0 + dux_dz*z, uz0])
    else:
        _, _, z = SpatialCoordinate(mesh)
        uy0 = -6.0
        duy_dz = 0.5
        true_expr = as_vector([ux0 + dux_dz*z, uy0 + duy_dz*z, uz0])

    # The initial expression has the gradient halved in the top and bottom
    # layers so as to represent inaccurate recovery
    ux_expr = conditional(z < height/3, ux0 + dux_dz*(height/6 + 0.5*z),
                          conditional(z > 2/3*height, ux0 + dux_dz*(1/3*height + 0.5*z),
                          ux0 + dux_dz*z))

    if element == 'interval':
        initial_expr = as_vector([ux_expr, uz0])
    else:
        uy_expr = conditional(z < height/3, uy0 + duy_dz*(height/6 + 0.5*z),
                              conditional(z > 2/3*height, uy0 + duy_dz*(1/3*height + 0.5*z),
                              uy0 + duy_dz*z))

        initial_expr = as_vector([ux_expr, uy_expr, uz0])

    # ------------------------------------------------------------------------ #
    # Set up initial conditions
    # ------------------------------------------------------------------------ #
    # Rather than projecting a conditional, interpolate into Vector CG1 and then project
    Vec_CG1 = VectorFunctionSpace(mesh, "CG", 1)
    initial_field_vec_CG1 = Function(Vec_CG1).interpolate(initial_expr)
    initial_field.project(initial_field_vec_CG1)
    true_field.project(true_expr)

    return initial_field, true_field


@pytest.mark.parametrize("element", ["interval", "quadrilateral", "triangular"])
def test_hcurl_recovery_kernels(element):

    initial_field, true_field = set_up_fields(element)

    V = true_field.function_space()
    new_field = Function(V).assign(initial_field)

    kernel = BoundaryRecoveryHCurl(V)
    kernel.apply(new_field, initial_field)

    tolerance = 1e-8
    assert np.allclose(true_field.dat.data, new_field.dat.data, tolerance), \
        f'HCurl boundary recovery incorrect for {element} elements'
