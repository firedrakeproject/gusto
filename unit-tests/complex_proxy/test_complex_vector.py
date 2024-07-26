
import firedrake as fd
import gusto.complex_proxy.vector as cpx

import pytest


def assemble(form):
    return fd.assemble(form).riesz_representation(riesz_map='l2')


cell = fd.Cell('triangle')

scalar_elements = [
    pytest.param(fd.FiniteElement("CG", cell, 1), id="CG1"),
    pytest.param(fd.FiniteElement("BDM", cell, 2), id="BDM1")
]

vector_elements = [
    pytest.param(fd.VectorElement("DG", cell, 1), id="VectorDG1"),
    pytest.param(fd.VectorElement("DG", cell, 1, dim=3), id="VectorDG1_3D")
]

tensor_elements = [
    pytest.param(fd.TensorElement("Lagrange", cell, 1), id="TensorL1"),
    pytest.param(fd.TensorElement("Lagrange", cell, 1, shape=(2, 3, 4)), id="TensorL1_234D")
]

elements = scalar_elements + vector_elements + tensor_elements


complex_numbers = [2+0j, 0+3j, 3+2j]

constant_z = [
    pytest.param(False, id="complex_z"),
    pytest.param(True, id="Constant_z"),
]


@pytest.fixture
def nx():
    return 10


@pytest.fixture
def mesh(nx):
    return fd.UnitSquareMesh(nx, nx)


@pytest.fixture
def mixed_element():
    return fd.MixedElement([param.values[0] for param in elements])


@pytest.mark.parametrize("elem", scalar_elements)
def test_finite_element(elem):
    """
    Test that the complex proxy FiniteElement is constructed correctly from a real FiniteElement.
    """
    celem = cpx.FiniteElement(elem)

    assert celem.num_sub_elements == 2, "The cpx element should have two subcomponents"

    for ce in celem.sub_elements:
        assert ce == elem, "Each component element should be the same as the real element"


@pytest.mark.parametrize("elem", vector_elements)
def test_vector_element(elem):
    """
    Test that the complex proxy FiniteElement is constructed correctly from a real VectorElement.
    """
    celem = cpx.FiniteElement(elem)

    assert celem.num_sub_elements == 2*elem.num_sub_elements, "The cpx element of a vector element should have twice as many subcomponents as the real space"

    assert celem._shape == (2, elem.num_sub_elements), "The cpx element of a vector element should be a tensor element"

    for ce in celem.sub_elements:
        assert ce == elem.sub_elements[0], "The element for each component should be the same as the real element"


@pytest.mark.parametrize("elem", tensor_elements)
def test_tensor_element(elem):
    """
    Test that the complex proxy FiniteElement is constructed correctly from a real TensorElement.
    """
    celem = cpx.FiniteElement(elem)

    assert celem.num_sub_elements == 2*elem.num_sub_elements, "The cpx element of a tensor should have twice as many subcomponents as the real space"

    assert celem._shape == (2,) + elem._shape, "The cpx element of a tensor element should be a tensor element of one order higher"

    for ce in celem.sub_elements:
        assert ce == elem.sub_elements[0], "The element for each component should be the same as the real element"


def test_mixed_element(mixed_element):
    """
    Test that the complex proxy FiniteElement is constructed correctly from a real MixedElement.
    """

    celem = cpx.FiniteElement(mixed_element)

    assert celem.num_sub_elements == mixed_element.num_sub_elements, "The cpx element of a mixed element should have the same number of components as the real element"

    for csub, msub in zip(celem.sub_elements, mixed_element.sub_elements):
        assert csub == cpx.FiniteElement(msub), "The each component of the cpx element of a mixed real space should be the cpx element of the real component"


@pytest.mark.parametrize("elem", elements)
def test_function_space(mesh, elem):
    """
    Test that the proxy complex FunctionSpace is correctly constructed for a scalar real FunctionSpace
    """
    V = fd.FunctionSpace(mesh, elem)
    W = cpx.FunctionSpace(V)

    assert W.ufl_element() == cpx.FiniteElement(elem), "The complex function space should have the cpx Element corresponding to the cpx Element of the real space"


def test_mixed_function_space(mesh, mixed_element):
    """
    Test that the proxy complex FunctionSpace is correctly constructed for a mixed real FunctionSpace
    """

    V = fd.FunctionSpace(mesh, mixed_element)
    W = cpx.FunctionSpace(V)

    assert len(W.subfunctions) == len(V.subfunctions), "The complex space should have the same number of components as the real space"

    for wcpt, vcpt in zip(W.subfunctions, V.subfunctions):
        assert wcpt == cpx.FunctionSpace(vcpt), "Each component of the complex space should be the complex space of the component of the real space"


@pytest.mark.parametrize("split_tuple", [False, True])
@pytest.mark.parametrize("elem", scalar_elements)
def test_set_get_part(mesh, elem, split_tuple):
    """
    Test that the real and imaginary parts are set and get correctly from/to real FunctionSpace

    TODO: add tests for vector_elements and tensor_elements
    """
    eps = 1e-12

    x, y = fd.SpatialCoordinate(mesh)

    if elem.reference_value_shape != ():
        dim = elem.reference_value_shape[0]
        expr0 = fd.as_vector([x*i for i in range(dim)])
        expr1 = fd.as_vector([-y*i for i in range(dim)])
    else:
        expr0 = x
        expr1 = -2*y

    V = fd.FunctionSpace(mesh, elem)
    W = cpx.FunctionSpace(V)

    u0 = fd.Function(V).project(expr0)
    u1 = fd.Function(V).project(expr1)
    ur = fd.Function(V).assign(1)
    ui = fd.Function(V).assign(1)
    w = fd.Function(W).assign(0)

    # test with no tuples, both tuples, and mixed
    u0s = u0.subfunctions if split_tuple else u0
    u1s = u1
    urs = ur.subfunctions if split_tuple else ur
    uis = ui
    ws = w.subfunctions if split_tuple else w

    # check 0 initialisation
    cpx.get_real(ws, urs)
    cpx.get_imag(ws, uis)

    assert fd.norm(ur) < eps
    assert fd.norm(ui) < eps

    # check real value is set correctly and imag value is unchanged
    cpx.set_real(ws, u0s)

    cpx.get_real(ws, urs)
    cpx.get_imag(ws, uis)

    assert fd.errornorm(u0, ur) < eps, "real component should match after `set_real` called"
    assert fd.norm(ui) < eps, "imag component should be unchanged by `set_real`"

    # check imag value is set correctly and real value is unchanged
    cpx.set_imag(ws, u1s)

    cpx.get_real(ws, urs)
    cpx.get_imag(ws, uis)

    assert fd.errornorm(u0, ur) < eps, "real component should be unchanged by `set_real`"
    assert fd.errornorm(u1, ui) < eps, "imag component should match after `set_real` called"


def test_mixed_set_get_part(mesh):
    """
    Test that the real and imaginary parts are set and get correctly from/to real FunctionSpace

    TODO: add tests for tensor_elements
    """
    eps = 1e-12

    x, y = fd.SpatialCoordinate(mesh)

    # set up mixed real space and initialise some functions

    V0 = fd.FunctionSpace(mesh, "BDM", 1)
    V1 = fd.FunctionSpace(mesh, "DG", 0)
    V = V0*V1

    u0 = fd.Function(V)
    u1 = fd.Function(V)
    ur = fd.Function(V).assign(1)
    ui = fd.Function(V).assign(1)

    u0.sub(0).project(fd.as_vector([x, 2*x*x]))
    u1.sub(0).project(fd.as_vector([-3*y, -4*y*y]))

    u0.sub(1).project(5*x)
    u1.sub(1).project(-6*y)

    # set up complex mixed space

    W = cpx.FunctionSpace(V)

    w = fd.Function(W).assign(0)
    cpx.get_real(w, ur)
    cpx.get_imag(w, ui)

    assert fd.norm(ur) < eps
    assert fd.norm(ui) < eps

    # check real value is set correctly and imag value is unchanged
    cpx.set_real(w, u0)

    cpx.get_real(w, ur)
    cpx.get_imag(w, ui)

    assert fd.errornorm(u0, ur) < eps, "real component should match after `set_real` called"
    assert fd.norm(ui) < eps, "imag component should be unchanged by `set_real`"

    # check imag value is set correctly and real value is unchanged
    cpx.set_imag(w, u1)

    cpx.get_real(w, ur), "real component should be unchanged by `set_real`"
    cpx.get_imag(w, ui), "imag component should match after `set_real` called"

    assert fd.errornorm(u0, ur) < eps
    assert fd.errornorm(u1, ui) < eps


@pytest.mark.parametrize("elem", scalar_elements[:1])
@pytest.mark.parametrize("z", complex_numbers)
@pytest.mark.parametrize("z_is_constant", constant_z)
def test_linear_form(mesh, elem, z, z_is_constant):
    """
    Test that the linear Form is constructed correctly

    TODO: add tests for tensor_elements
    """
    eps = 1e-12

    if z_is_constant:
        z = cpx.ComplexConstant(z)

    V = fd.FunctionSpace(mesh, elem)
    W = cpx.FunctionSpace(V)

    x, y = fd.SpatialCoordinate(mesh)

    if elem.reference_value_shape != ():
        vec_expr = [x*x-y, y+x, -y-0.5*x]
        dim = elem.reference_value_shape[0]
        f = fd.as_vector(vec_expr[:dim])
    else:
        f = x*x-y

    def L(v):
        return fd.inner(f, v)*fd.dx

    v = fd.TestFunction(V)
    rhs = assemble(L(v))

    ur = fd.Function(V)
    ui = fd.Function(V)
    w = fd.Function(W)

    w = assemble(cpx.LinearForm(W, z, L))

    cpx.get_real(w, ur)
    cpx.get_imag(w, ui)

    zr = z.real
    zi = z.imag
    assert fd.errornorm(zr*rhs, ur) < eps, "z*LinearForm just does componentwise multiplication"
    assert fd.errornorm(zi*rhs, ui) < eps, "z*LinearForm just does componentwise multiplication"


@pytest.mark.parametrize("elem", scalar_elements[:1])
@pytest.mark.parametrize("z_is_constant", constant_z)
def test_bilinear_form(mesh, elem, z_is_constant):
    """
    Test that the bilinear form is constructed correctly

    TODO: add tests for tensor_elements
    """
    eps = 1e-12

    # set up the real problem
    V = fd.FunctionSpace(mesh, elem)

    x, y = fd.SpatialCoordinate(mesh)

    if elem.reference_value_shape != ():
        vec_expr = [x*x-y, y+x, -y-0.5*x]
        dim = elem.reference_value_shape[0]
        expr = fd.as_vector(vec_expr[:dim])
    else:
        expr = x*x-y

    f = fd.Function(V).interpolate(expr)

    def form_function(u, v):
        return fd.inner(u, v)*fd.dx

    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)

    a = form_function(u, v)

    # the real value
    b = assemble(fd.action(a, f))

    # set up the complex problem
    W = cpx.FunctionSpace(V)

    g = fd.Function(W)

    cpx.set_real(g, f)
    f.assign(2*f)
    cpx.set_imag(g, f)

    # real/imag parts of mat-vec product
    br = fd.Function(V)
    bi = fd.Function(V)

    # non-zero only on diagonal blocks: real and imag parts independent
    zr = 3+0j
    if z_is_constant:
        zr = cpx.ComplexConstant(zr)

    K = cpx.BilinearForm(W, zr, form_function)
    wr = assemble(fd.action(K, g))

    cpx.get_real(wr, br)
    cpx.get_imag(wr, bi)

    assert fd.errornorm(3*1*b, br) < eps, "If z is purely real, just does component-wise multiplication"
    assert fd.errornorm(3*2*b, bi) < eps, "If z is purely real, just does component-wise multiplication"

    # non-zero only on off-diagonal blocks: real and imag parts independent
    zi = 0+4j
    if z_is_constant:
        zi = cpx.ComplexConstant(zi)

    K = cpx.BilinearForm(W, zi, form_function)
    wi = assemble(fd.action(K, g))

    cpx.get_real(wi, br)
    cpx.get_imag(wi, bi)

    assert fd.errornorm(-4*2*b, br) < eps, "if z is purely imag, does off-diagonal multiplication"
    assert fd.errornorm(4*1*b, bi) < eps, "if z is purely imag, does off-diagonal multiplication"

    # non-zero in all blocks:
    if not z_is_constant:
        z = zr + zi
    else:
        z = cpx.ComplexConstant(zr.real, zi.imag)

    K = cpx.BilinearForm(W, z, form_function)
    wz = assemble(fd.action(K, g))

    cpx.get_real(wz, br)
    cpx.get_imag(wz, bi)

    # mat-vec multiplication should be linear
    br_check = fd.Function(V)
    bi_check = fd.Function(V)

    wz.assign(wr + wi)

    cpx.get_real(wz, br_check)
    cpx.get_imag(wz, bi_check)

    assert fd.errornorm(br_check, br) < eps, "MatVecMult z*A*x should be linear in z"
    assert fd.errornorm(bi_check, bi) < eps, "MatVecMult z*A*x should be linear in z"


@pytest.mark.parametrize("bc_type", ["nobc", "dirichletbc"])
def test_linear_solve(mesh, bc_type):
    """
    Test that the bilinear form is constructed correctly

    TODO: add tests for tensor_elements
    """
    from math import pi

    eps = 1e-12

    # set up the real problem
    V = fd.FunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)

    def form_function(u, v):
        return (fd.inner(u, v) + fd.dot(fd.grad(u), fd.grad(v)))*fd.dx

    f = (2*pi**2)*fd.sin(pi*x)*fd.sin(pi*y)

    def rhs(v):
        return fd.inner(f, v)*fd.dx

    # the real solution
    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)
    A = form_function(u, v)
    L = rhs(v)

    if bc_type == "nobc":
        bcs = []
    elif bc_type == "dirichletbc":
        bcs = [fd.DirichletBC(V, 0, 1)]
    else:
        raise ValueError(f"Unrecognised boundary condition type: {bc_type}")

    ureal = fd.Function(V)
    fd.solve(A == L, ureal, bcs=bcs)

    # set up the complex problem
    W = cpx.FunctionSpace(V)

    cpx_bcs = []
    for bc in bcs:
        cpx_bcs.extend([*cpx.DirichletBC(W, V, bc)])

    # A non-zero only on diagonal blocks: real and imag parts independent
    zr = 3+0j
    zl = 1+2j

    A = cpx.BilinearForm(W, zr, form_function)
    L = cpx.LinearForm(W, zl, rhs)

    wr = fd.Function(W)

    fd.solve(A == L, wr, bcs=cpx_bcs)

    wcheckr = fd.Function(V)
    wchecki = fd.Function(V)

    cpx.get_real(wr, wcheckr)
    cpx.get_imag(wr, wchecki)

    assert fd.errornorm(1*ureal/3, wcheckr) < eps, "If z is purely real, equivalent to component-wise solve"
    assert fd.errornorm(2*ureal/3, wchecki) < eps, "If z is purely real, equivalent to component-wise solve"

    # non-zero on all blocks but imag part of rhs is zero
    zi = 2+4j

    A = cpx.BilinearForm(W, zi, form_function)
    L = cpx.LinearForm(W, 1, rhs)

    wi = fd.Function(W)

    fd.solve(A == L, wi, bcs=cpx_bcs)

    cpx.get_real(wi, wcheckr)
    cpx.get_imag(wi, wchecki)

    # eliminate imaginary part to check real part
    assert fd.errornorm(2*ureal/(2*2+4*4), wcheckr) < eps, "real part should be calculated from eliminating the imaginary part"

    # back substitute real part to check imaginary part
    g = 0.25*(2*fd.action(form_function(u, v), wcheckr) - rhs(v))
    a = form_function(u, v)

    ui = fd.Function(V)
    fd.solve(a == g, ui, bcs=bcs)

    assert fd.errornorm(ui, wchecki) < eps, "real part should be calculated from eliminating the imaginary part"


@pytest.mark.parametrize("bc_type", ["nobc", "dirichletbc"])
def test_mixed_linear_solve(mesh, bc_type):
    """
    Test that the bilinear form is constructed correctly

    TODO: add tests for tensor_elements
    """
    eps = 1e-12

    # set up the real problem
    V0 = fd.FunctionSpace(mesh, "BDM", 1)
    V1 = fd.FunctionSpace(mesh, "DG", 0)
    V = V0*V1

    x, y = fd.SpatialCoordinate(mesh)

    def form_function(sigma, u, tau, v):
        return (fd.dot(sigma, tau) + fd.div(tau)*u + fd.div(sigma)*v)*fd.dx

    def rhs(tau, v):
        f = 10*fd.exp(-(pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.02)
        return -fd.inner(f, v)*fd.dx

    if bc_type == "nobc":
        bcs = []
    elif bc_type == "dirichletbc":
        bcs = [
            fd.DirichletBC(V.sub(1), 0, 3),
            fd.DirichletBC(V.sub(1), 0, 4)
        ]
    else:
        raise ValueError(f"Unrecognised boundary condition type: {bc_type}")

    # the real solution
    u = fd.TrialFunctions(V)
    v = fd.TestFunctions(V)
    A = form_function(*u, *v)
    L = rhs(*v)

    ureal = fd.Function(V)
    fd.solve(A == L, ureal, bcs=bcs)

    # set up the complex problem
    W = cpx.FunctionSpace(V)

    cpx_bcs = []
    for bc in bcs:
        cpx_bcs.extend([*cpx.DirichletBC(W, V, bc)])

    # A non-zero only on diagonal blocks: real and imag parts independent
    zr = 3+0j
    zl = 1+2j

    A = cpx.BilinearForm(W, zr, form_function)
    L = cpx.LinearForm(W, zl, rhs)

    wr = fd.Function(W)

    fd.solve(A == L, wr, bcs=cpx_bcs)

    wcheckr = fd.Function(V)
    wchecki = fd.Function(V)

    cpx.get_real(wr, wcheckr)
    cpx.get_imag(wr, wchecki)

    assert fd.errornorm(1*ureal/3, wcheckr) < eps, "If z is purely real, equivalent to component-wise solve"
    assert fd.errornorm(2*ureal/3, wchecki) < eps, "If z is purely real, equivalent to component-wise solve"

    # non-zero on all blocks but imag part of rhs is zero
    zi = 2+4j

    A = cpx.BilinearForm(W, zi, form_function)
    L = cpx.LinearForm(W, 1, rhs)

    wi = fd.Function(W)

    fd.solve(A == L, wi, bcs=cpx_bcs)

    cpx.get_real(wi, wcheckr)
    cpx.get_imag(wi, wchecki)

    # eliminate imaginary part to check real part
    assert fd.errornorm(2*ureal/(2*2+4*4), wcheckr) < eps

    # back substitute real part to check imaginary part
    g = 0.25*(2*fd.action(form_function(*u, *v), wcheckr) - rhs(*v))
    a = form_function(*u, *v)

    ui = fd.Function(V)
    fd.solve(a == g, ui, bcs=bcs)

    assert fd.errornorm(ui, wchecki) < eps, "imag part should be calculated from back-substituting the real part"


def test_derivative_solve(mesh):
    """
    Test that the bilinear form is constructed correctly

    TODO: add tests for tensor_elements
    """
    from math import pi

    eps = 1e-12

    # set up the real problem
    V = fd.FunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)

    mu = fd.Constant(0.05)

    def form_function(u, v):
        return (fd.inner(u, v) + (1 + mu*fd.inner(u, u))*fd.dot(fd.grad(u), fd.grad(v)))*fd.dx

    f = (1 + 8*pi*pi)*fd.cos(2*pi*x)*fd.cos(2*pi*y)

    def rhs(v):
        return fd.inner(f, v)*fd.dx

    init_expr = fd.cos(2*pi*x)*fd.cos(2*pi*y)

    # the real solution
    u = fd.Function(V).project(init_expr)
    v = fd.TestFunction(V)
    F = form_function(u, v)
    A = fd.derivative(F, u)
    L = rhs(v)

    ureal = fd.Function(V)
    fd.solve(A == L, ureal)

    # set up the complex problem
    W = cpx.FunctionSpace(V)

    # linearise around "u"
    u0 = fd.Function(W)
    cpx.set_real(u0, u)
    cpx.set_imag(u0, u)

    # A non-zero only on diagonal blocks: real and imag parts independent
    zr = 3+0j
    zl = 1+2j

    A = cpx.derivative(zr, form_function, u0)
    L = cpx.LinearForm(W, zl, rhs)

    wr = fd.Function(W)

    fd.solve(A == L, wr)

    wcheckr = fd.Function(V)
    wchecki = fd.Function(V)

    cpx.get_real(wr, wcheckr)
    cpx.get_imag(wr, wchecki)

    assert fd.errornorm(1*ureal/3, wcheckr) < eps, "If z is purely real, equivalent to component-wise solve"
    assert fd.errornorm(2*ureal/3, wchecki) < eps, "If z is purely real, equivalent to component-wise solve"

    # non-zero on all blocks but imag part of rhs is zero
    zi = 2+4j

    A = cpx.derivative(zi, form_function, u0)
    L = cpx.LinearForm(W, 1, rhs)

    wi = fd.Function(W)

    fd.solve(A == L, wi)

    cpx.get_real(wi, wcheckr)
    cpx.get_imag(wi, wchecki)

    # eliminate imaginary part to check real part
    assert fd.errornorm(2*ureal/(2*2+4*4), wcheckr) < eps, "real part should be calculated from eliminating the imaginary part"

    # back substitute real part to check imaginary part
    ut = fd.Function(V).project(init_expr)
    F = form_function(ut, v)
    A = fd.derivative(F, ut)
    g = 0.25*(2*fd.action(A, wcheckr) - rhs(v))

    ui = fd.Function(V)
    fd.solve(A == g, ui)

    assert fd.errornorm(ui, wchecki) < eps, "imag part should be calculated from back-substituting the real part"
