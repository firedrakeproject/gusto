"""
Test whether the boundary recovery is working in on 3D Cartesian meshes.
To be working, a linearly varying field should be exactly recovered.

This is tested for:
- the lowest-order density space recovered to DG1
- the lowest-order velocity space recovered to vector DG1
- the lowest-order temperature space recovered to DG1
- the lowest-order density space recovered to lowest-order temperature space.
"""

from firedrake import (PeriodicRectangleMesh, RectangleMesh, ExtrudedMesh,
                       SpatialCoordinate, FiniteElement, HDiv, FunctionSpace,
                       TensorProductElement, Function, interval, norm, errornorm,
                       VectorFunctionSpace, as_vector, HCurl)
from gusto import *
import numpy as np
import pytest

np.random.seed(0)


@pytest.fixture
def mesh(geometry, element):

    Lx = 100.
    Ly = 100.
    H = 100.

    deltax = Lx / 5.
    deltay = Ly / 5.
    deltaz = H / 5.
    ncolumnsy = int(Ly/deltay)
    ncolumnsx = int(Lx/deltax)
    nlayers = int(H/deltaz)

    quadrilateral = True if element == "quadrilateral" else False

    if geometry == "periodic-in-both":
        m = PeriodicRectangleMesh(ncolumnsx, ncolumnsy, Lx, Ly,
                                  direction='both', quadrilateral=quadrilateral)
    elif geometry == "periodic-in-x":
        m = PeriodicRectangleMesh(ncolumnsx, ncolumnsy, Lx, Ly,
                                  direction='x', quadrilateral=quadrilateral)
    elif geometry == "periodic-in-y":
        m = PeriodicRectangleMesh(ncolumnsx, ncolumnsy, Lx, Ly,
                                  direction='y', quadrilateral=quadrilateral)
    elif geometry == "non-periodic":
        m = RectangleMesh(ncolumnsx, ncolumnsy, Lx, Ly, quadrilateral=quadrilateral)

    extruded_mesh = ExtrudedMesh(m, layers=nlayers, layer_height=deltaz)

    return extruded_mesh


@pytest.fixture
def expr(geometry, mesh):

    x, y, z = SpatialCoordinate(mesh)

    if geometry == "periodic-in-both":
        analytic_expr = np.random.randn() + np.random.randn() * z
    elif geometry == "periodic-in-x":
        analytic_expr = np.random.randn() + np.random.randn() * y + np.random.randn() * z
    elif geometry == "periodic-in-y":
        analytic_expr = np.random.randn() + np.random.randn() * x + np.random.randn() * z
    elif geometry == "non-periodic":
        analytic_expr = np.random.randn() + np.random.randn() * x + np.random.randn() * y + np.random.randn() * z

    return analytic_expr


@pytest.mark.parametrize("geometry", ["periodic-in-both", "periodic-in-x",
                                      "periodic-in-y", "non-periodic"])
@pytest.mark.parametrize("element", ["quadrilateral", "triangular"])
def test_3D_cartesian_recovery(geometry, element, mesh, expr):

    if element == "quadrilateral":
        family = "RTCF"
    elif geometry == "periodic-in-both":
        # TODO: NB: boundary recovery does not exactly work for RT with lateral boundaries
        family = "RT"
    else:
        family = "BDM"

    # horizontal base spaces
    cell = mesh._base_mesh.ufl_cell().cellname()
    u_div_hori = FiniteElement(family, cell, 1)
    w_div_hori = FiniteElement("DG", cell, 0)

    # vertical base spaces
    u_div_vert = FiniteElement("DG", interval, 0)
    w_div_vert = FiniteElement("CG", interval, 1)

    # build elements
    u_div_element = HDiv(TensorProductElement(u_div_hori, u_div_vert))
    w_div_element = HDiv(TensorProductElement(w_div_hori, w_div_vert))
    theta_element = TensorProductElement(w_div_hori, w_div_vert)
    v_div_element = u_div_element + w_div_element

    # spaces
    DG0 = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)
    Vt = FunctionSpace(mesh, theta_element)
    Vu_div = FunctionSpace(mesh, v_div_element)
    vec_CG1 = VectorFunctionSpace(mesh, "CG", 1)

    # our actual theta and rho and v
    rho_CG1_true = Function(CG1).interpolate(expr)
    theta_CG1_true = Function(CG1).interpolate(expr)
    v_CG1_true = Function(vec_CG1).interpolate(as_vector([expr, expr, expr]))
    rho_Vt_true = Function(Vt).interpolate(expr)

    # make the initial fields by projecting expressions into the lowest order spaces
    rho_DG0 = Function(DG0).interpolate(expr)
    rho_CG1 = Function(CG1)
    theta_Vt = Function(Vt).interpolate(expr)
    theta_CG1 = Function(CG1)
    v_Vu_div = Function(Vu_div).project(as_vector([expr, expr, expr]))
    v_CG1 = Function(vec_CG1)
    rho_Vt = Function(Vt)

    # make the recoverers and do the recovery
    rho_recoverer = Recoverer(rho_DG0, rho_CG1, boundary_method=BoundaryMethod.taylor)
    theta_recoverer = Recoverer(theta_Vt, theta_CG1, boundary_method=BoundaryMethod.taylor)
    v_recoverer = Recoverer(v_Vu_div, v_CG1, boundary_method=BoundaryMethod.taylor)
    rho_Vt_recoverer = Recoverer(rho_DG0, rho_Vt, boundary_method=BoundaryMethod.extruded)

    rho_recoverer.project()
    theta_recoverer.project()
    v_recoverer.project()
    rho_Vt_recoverer.project()

    rho_diff = errornorm(rho_CG1, rho_CG1_true) / norm(rho_CG1_true)
    theta_diff = errornorm(theta_CG1, theta_CG1_true) / norm(theta_CG1_true)
    v_div_diff = errornorm(v_CG1, v_CG1_true) / norm(v_CG1_true)
    rho_Vt_diff = errornorm(rho_Vt, rho_Vt_true) / norm(rho_Vt_true)

    tolerance = 1e-7
    error_message = ("""
                     Incorrect recovery for {variable} with {boundary} boundary method
                     on {geometry} 3D Cartesian domain with {element} elements
                     """)
    assert rho_diff < tolerance, error_message.format(variable='rho', boundary='taylor',
                                                      geometry=geometry, element=element)
    assert v_div_diff < tolerance, error_message.format(variable='v', boundary='taylor',
                                                        geometry=geometry, element=element)
    assert theta_diff < tolerance, error_message.format(variable='rho', boundary='taylor',
                                                        geometry=geometry, element=element)
    assert rho_Vt_diff < tolerance, error_message.format(variable='rho', boundary='physics',
                                                         geometry=geometry, element=element)

    # ------------------------------------------------------------------------ #
    # Special test for hcurl boundary recovery
    # ------------------------------------------------------------------------ #

    if geometry == 'periodic-in-both':
        family = "RTCE" if element == "quadrilateral" else "RTE"

        u_curl_hori = FiniteElement(family, cell, 1)
        w_curl_hori = FiniteElement("CG", cell, 1)
        u_curl_vert = FiniteElement("CG", interval, 1)
        w_curl_vert = FiniteElement("DG", interval, 0)

        u_curl_element = HCurl(TensorProductElement(u_curl_hori, u_curl_vert))
        w_curl_element = HCurl(TensorProductElement(w_curl_hori, w_curl_vert))
        v_curl_element = u_curl_element + w_curl_element

        Vu_curl = FunctionSpace(mesh, v_curl_element)

        # make expression and fields -- x and y components vary linearly in z
        xyz = SpatialCoordinate(mesh)
        x_expr = np.random.randn() + np.random.randn()*xyz[2]
        y_expr = np.random.randn() + np.random.randn()*xyz[2]
        v_Vu_div = Function(Vu_div).project(as_vector([x_expr, y_expr, 0.0]))
        v_curl_true = Function(Vu_curl).project(as_vector([x_expr, y_expr, 0.0]))
        v_curl = Function(Vu_curl)

        # make the recoverers and do the recovery
        v_curl_recoverer = Recoverer(v_Vu_div, v_curl, method='project', boundary_method=BoundaryMethod.hcurl)

        v_curl_recoverer.project()
        v_curl_diff = errornorm(v_curl, v_curl_true) / norm(v_curl_true)

        assert v_curl_diff < tolerance, error_message.format(variable='v', boundary='hcurl',
                                                             geometry=geometry, element=element)
