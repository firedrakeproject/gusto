"""
Test whether the boundary recovery is working in on 3D spherical meshes.
To be working, a linearly-varying field in the radial direction should be
exactly recovered.

This is tested for:
- the lowest-order density space recovered to DG1
- the lowest-order hdiv space recovered to a hcurl space
- the lowest-order temperature space recovered to DG1
- the lowest-order density space recovered to lowest-order temperature space.
"""

from firedrake import (CubedSphereMesh, IcosahedralSphereMesh, ExtrudedMesh,
                       SpatialCoordinate, FiniteElement, HDiv, FunctionSpace,
                       TensorProductElement, Function, interval, norm, errornorm,
                       HCurl, as_vector, sqrt)
from gusto import *
import numpy as np
import pytest

np.random.seed(0)


@pytest.fixture
def mesh(element):

    H = 100.

    deltaz = H / 5.
    nlayers = int(H/deltaz)

    ref_level = 1
    radius = 10.0
    if element == "quadrilateral":
        m = CubedSphereMesh(radius, refinement_level=ref_level, degree=2)
    elif element == "triangular":
        m = IcosahedralSphereMesh(radius, refinement_level=ref_level, degree=3)

    extruded_mesh = ExtrudedMesh(m, layers=nlayers, layer_height=deltaz,
                                 extrusion_type='radial')

    return extruded_mesh


@pytest.mark.parametrize("element", ["quadrilateral", "triangular"])
def test_3D_spherical_recovery(element, mesh):

    hdiv_family = "RTCF" if element == "quadrilateral" else "RT"
    hcurl_family = "RTCE" if element == "quadrilateral" else "RTE"

    # horizontal base spaces
    cell = mesh._base_mesh.ufl_cell().cellname()
    u_div_hori = FiniteElement(hdiv_family, cell, 1)
    w_div_hori = FiniteElement("DG", cell, 0)
    u_curl_hori = FiniteElement(hcurl_family, cell, 1)
    w_curl_hori = FiniteElement("CG", cell, 1)

    # vertical base spaces
    u_div_vert = FiniteElement("DG", interval, 0)
    w_div_vert = FiniteElement("CG", interval, 1)
    u_curl_vert = FiniteElement("CG", interval, 1)
    w_curl_vert = FiniteElement("DG", interval, 0)

    # build elements
    u_div_element = HDiv(TensorProductElement(u_div_hori, u_div_vert))
    w_div_element = HDiv(TensorProductElement(w_div_hori, w_div_vert))
    u_curl_element = HCurl(TensorProductElement(u_curl_hori, u_curl_vert))
    w_curl_element = HCurl(TensorProductElement(w_curl_hori, w_curl_vert))
    theta_element = TensorProductElement(w_div_hori, w_div_vert)
    hdiv_element = u_div_element + w_div_element
    hcurl_element = u_curl_element + w_curl_element

    # spaces
    DG0 = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)
    Vt = FunctionSpace(mesh, theta_element)
    VHDiv = FunctionSpace(mesh, hdiv_element)
    VHCurl = FunctionSpace(mesh, hcurl_element)

    # expressions
    xyz = SpatialCoordinate(mesh)
    r = sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2)
    scalar_expr = np.random.randn() + np.random.randn() * r
    u0 = np.random.randn()
    vector_expr = as_vector([-u0*xyz[1], u0*xyz[0], 0.0])

    # our actual theta and rho and v
    rho_CG1_true = Function(CG1).interpolate(scalar_expr)
    theta_CG1_true = Function(CG1).interpolate(scalar_expr)
    v_hcurl_true = Function(VHCurl).project(vector_expr)
    rho_Vt_true = Function(Vt).interpolate(scalar_expr)

    # make the initial fields by projecting expressions into the lowest order spaces
    rho_DG0 = Function(DG0).interpolate(scalar_expr)
    rho_CG1 = Function(CG1)
    theta_Vt = Function(Vt).interpolate(scalar_expr)
    theta_CG1 = Function(CG1)
    v_hdiv = Function(VHDiv).project(vector_expr)
    v_hcurl_bad = Function(VHCurl)
    v_hcurl_good = Function(VHCurl)
    rho_Vt = Function(Vt)

    # make the recoverers and do the recovery
    rho_recoverer = Recoverer(rho_DG0, rho_CG1, boundary_method=BoundaryMethod.extruded)
    theta_recoverer = Recoverer(theta_Vt, theta_CG1)
    v_recoverer_bad = Recoverer(v_hdiv, v_hcurl_bad, method='project')
    v_recoverer_good = Recoverer(v_hdiv, v_hcurl_good, method='project', boundary_method=BoundaryMethod.hcurl)
    rho_Vt_recoverer = Recoverer(rho_DG0, rho_Vt, boundary_method=BoundaryMethod.extruded)

    rho_recoverer.project()
    theta_recoverer.project()
    v_recoverer_bad.project()
    v_recoverer_good.project()
    rho_Vt_recoverer.project()

    rho_diff = errornorm(rho_CG1, rho_CG1_true) / norm(rho_CG1_true)
    theta_diff = errornorm(theta_CG1, theta_CG1_true) / norm(theta_CG1_true)
    v_diff_bad = errornorm(v_hcurl_bad, v_hcurl_true) / norm(v_hcurl_true)
    v_diff_good = errornorm(v_hcurl_good, v_hcurl_true) / norm(v_hcurl_true)
    rho_Vt_diff = errornorm(rho_Vt, rho_Vt_true) / norm(rho_Vt_true)

    tolerance = 1e-12
    error_message = ("""
                     Incorrect recovery for {variable} with {boundary} boundary method
                     on 3D spherical domain with {element} elements
                     """)
    assert rho_diff < tolerance, error_message.format(variable='rho', boundary='extruded',
                                                      element=element)
    assert v_diff_good < 0.75*v_diff_bad, error_message.format(variable='v', boundary='hcurl',
                                                               element=element)
    assert theta_diff < tolerance, error_message.format(variable='rho', boundary='no',
                                                        element=element)
    assert rho_Vt_diff < tolerance, error_message.format(variable='rho', boundary='physics',
                                                         element=element)
