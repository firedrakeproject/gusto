"""
This short program applies the boundary recoverer operation to check
the boundary values under some analytic forms.
"""
from gusto import *
from firedrake import (as_vector, RectangleMesh, SpatialCoordinate,
                       ExtrudedMesh, FunctionSpace, Function, errornorm,
                       VectorFunctionSpace, interval, TensorProductElement,
                       FiniteElement, HDiv, norm)


def setup_3d_recovery(dirname):

    L = 100.
    H = 10.
    W = 1.

    deltax = L / 5.
    deltay = W / 5.
    deltaz = H / 5.
    nlayers = int(H/deltaz)
    ncolumnsx = int(L/deltax)
    ncolumnsy = int(W/deltay)

    m = RectangleMesh(ncolumnsx, ncolumnsy, L, W, quadrilateral=True)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
    x, y, z = SpatialCoordinate(mesh)

    # horizontal base spaces
    cell = mesh._base_mesh.ufl_cell().cellname()
    u_hori = FiniteElement("RTCF", cell, 1)
    w_hori = FiniteElement("DG", cell, 0)

    # vertical base spaces
    u_vert = FiniteElement("DG", interval, 0)
    w_vert = FiniteElement("CG", interval, 1)

    # build elements
    u_element = HDiv(TensorProductElement(u_hori, u_vert))
    w_element = HDiv(TensorProductElement(w_hori, w_vert))
    theta_element = TensorProductElement(w_hori, w_vert)
    v_element = u_element + w_element

    # spaces
    VDG0 = FunctionSpace(mesh, "DG", 0)
    VCG1 = FunctionSpace(mesh, "CG", 1)
    VDG1 = FunctionSpace(mesh, "DG", 1)
    Vt = FunctionSpace(mesh, theta_element)
    Vu = FunctionSpace(mesh, v_element)
    VuCG1 = VectorFunctionSpace(mesh, "CG", 1)
    VuDG1 = VectorFunctionSpace(mesh, "DG", 1)

    # set up initial conditions
    expr = 10 + 5*x + y + z + x*y + x*z + 0.001*y*z + 3*x*y*z

    # our actual theta and rho and v
    rho_CG1_true = Function(VCG1).interpolate(expr)
    theta_CG1_true = Function(VCG1).interpolate(expr)
    v_CG1_true = Function(VuCG1).interpolate(as_vector([expr, expr, expr]))

    # make the initial fields by projecting expressions into the lowest order spaces
    rho_DG0 = Function(VDG0).interpolate(expr)
    rho_CG1 = Function(VCG1)
    theta_Vt = Function(Vt).interpolate(expr)
    theta_CG1 = Function(VCG1)
    v_Vu = Function(Vu).project(as_vector([expr, expr, expr]))
    v_CG1 = Function(VuCG1)

    # make the recoverers and do the recovery
    rho_recoverer = Recoverer(rho_DG0, rho_CG1, VDG=VDG1, boundary_method='scalar')
    theta_recoverer = Recoverer(theta_Vt, theta_CG1, VDG=VDG1, boundary_method='scalar')
    v_recoverer = Recoverer(v_Vu, v_CG1, VDG=VuDG1, boundary_method='vector')

    rho_recoverer.project()
    theta_recoverer.project()
    v_recoverer.project()

    rho_diff = errornorm(rho_CG1, rho_CG1_true) / norm(rho_CG1)
    theta_diff = errornorm(theta_CG1, theta_CG1_true) / norm(theta_CG1)
    v_diff = errornorm(v_CG1, v_CG1_true) / norm(v_CG1)

    return (rho_diff, theta_diff, v_diff)


def run_3d_recovery(dirname):

    (rho_diff, theta_diff, v_diff) = setup_3d_recovery(dirname)
    return (rho_diff, theta_diff, v_diff)


def test_3d_boundary_limiter(tmpdir):

    dirname = str(tmpdir)
    rho_diff, theta_diff, v_diff = run_3d_recovery(dirname)

    tolerance = 1e-7
    assert rho_diff < tolerance
    assert theta_diff < tolerance
    assert v_diff < tolerance
