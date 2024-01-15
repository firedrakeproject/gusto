
from firedrake import(UnitSquareMesh, ExtrudedMesh, Function, File,
                      FunctionSpace, SpatialCoordinate, CallableSobolevSpace, HCurlSobolevSpace,
                      cell, FiniteElement, TensorProductElement, interval, TestFunction, TrialFunction,
                      FacetNormal, inner, curl, jump, cross, LinearVariationalSolver, LinearVariationalProblem, dS_v, dx, as_vector)

from gusto.diagnostics import Vorticity

mesh = UnitSquareMesh(10, 10, quadrilateral=True)
emesh = ExtrudedMesh(mesh, layers=5, layer_height=2)

HCurl = CallableSobolevSpace(HCurlSobolevSpace.name, HCurlSobolevSpace.parents)

hori_hcurl = FiniteElement('RTF', cell, 2)
vert_cg = FiniteElement("CG", interval, 2)
vert_dg = FiniteElement("DG", interval, 1)
hori_cg = FiniteElement("S", cell, 2)


Vh_elt = HCurl(TensorProductElement(hori_hcurl, vert_cg))
Vv_elt = HCurl(TensorProductElement(hori_cg, vert_dg))
V_elt = Vh_elt + Vv_elt
VCurl = FunctionSpace(emesh, V_elt)
Vorticity = Function(Vcurl)
x, y, z = SpatialCoordinate(emesh)
u = as_vector([x, y, z])

omega = TrialFunction(VCurl)
n = FacetNormal(emesh)
w = TestFunction(VCurl)
a = inner(omega, w) * dx
L = inner(u, curl(w)) * dx - jump(cross(w, u), n) * dS_v

problem = LinearVariationalProblem(a, L, Vorticity)
evaluator = LinearVariationalSolver(problem, solver_parameters={'ksp_type': 'cg'})
evaluator.solve()

out = File('vorticity.pvd')
out.write(Vorticity)