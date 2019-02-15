import pytest
from firedrake import (TestFunction, Function, FunctionSpace,
                       VectorFunctionSpace, MixedFunctionSpace,
                       UnitSquareMesh, as_vector,
                       dx, Constant, LinearVariationalProblem,
                       LinearVariationalSolver,
                       TrialFunctions, errornorm)
from gusto import *


@pytest.fixture
def mesh():
    return UnitSquareMesh(2, 2)


@pytest.fixture
def V(mesh):
    return FunctionSpace(mesh, "CG", 1)


@pytest.fixture
def W(mesh, V):
    V_ = VectorFunctionSpace(mesh, "CG", 2)
    return MixedFunctionSpace((V_, V))


@pytest.fixture
def function(V):
    return Function(V)


@pytest.fixture
def mixed_function(W):
    return Function(W)


@pytest.fixture
def label_a():
    return Label("a", validator=lambda arg: isinstance(arg, Function))


@pytest.fixture
def label_x():
    return Label("x", value="y", validator=lambda arg: isinstance(arg, str))


@pytest.fixture
def form(mesh, function, V):
    phi = TestFunction(V)
    return phi*function*dx


@pytest.fixture
def mixed_form(mesh, mixed_function, W, label_a):
    sigma, phi = TestFunctions(W)
    x, y = mixed_function.split()
    return label_a(index(inner(sigma, x)*dx, 0) + index(phi*y*dx, 1), mixed_function)


@pytest.fixture
def labelled_form(label_a, function, form):
    return label_a(form, function)


@pytest.fixture
def term(form):
    return Term(form)


def test_label_form(label_a, labelled_form, form):
    """
    tests that labelling a form returns a LabelledForm with the correct
    label and form
    """
    assert isinstance(labelled_form, LabelledForm)
    assert all([t.has_label(label_a) for t in labelled_form])
    assert labelled_form.form == form


def test_label_term(label_a, term, form):
    """
    tests that labelling a term returns another term with the correct label
    and form
    """
    labelled_term = label_a(term)
    assert isinstance(labelled_term, Term)
    assert labelled_term.has_label(label_a)
    assert labelled_term.form == form


def test_label_labelled_form(labelled_form, label_a, label_x):
    """
    test that labelling a labelled_form returns a labelled_form with the
    correct labels and that the label can be changed.
    """
    new = label_x(labelled_form)
    assert isinstance(new, LabelledForm)
    assert all([t.has_label(label_a, label_x) for t in new])
    assert all([t.get(label_x) == "y" for t in new])
    new = label_x(new, "z")
    assert all([t.get(label_x) == "z" for t in new])


def test_remove_label(form, labelled_form, label_a):
    """
    test that we can remove labels from both terms and labelled forms
    """
    t = Term(form, {label_a.label: label_a.value})
    assert t.has_label(label_a)
    t_new = label_a.remove(t)
    assert not t_new.has_label(label_a)
    new_labelled_form = label_a.remove(labelled_form)
    assert not any([t.has_label(label_a) for t in new_labelled_form])


def test_add_term(term, labelled_form):
    """
    test that adding to a term works as expected
    """
    a = term
    a += term
    assert isinstance(a, LabelledForm)
    assert len(a) == 2
    b = term
    b += None
    assert b == term
    b = None
    b += term
    assert b == term
    c = term
    c += labelled_form
    assert isinstance(c, LabelledForm)
    assert len(c) == 2
    c += term
    assert isinstance(c, LabelledForm)
    assert len(c) == 3


def test_mul_term(term):
    """
    test that we can multiply a term by a float and Constant
    """
    for coeff in [1., Constant(1.)]:
        new_term = coeff*term
        assert isinstance(new_term, Term)
        assert not new_term.form == term.form
        assert new_term.labels == term.labels


def test_add_labelled_form(form, term, labelled_form):
    """
    test that adding to a labelled form works as expected
    """
    for other in [form, term, labelled_form]:
        a = labelled_form
        a += other
        assert isinstance(a, LabelledForm)
        assert len(a) == 2

    b = labelled_form
    b += None
    assert b == b


def test_sub_labelled_form(term, labelled_form):
    """
    test that subtracting from a labelled form works as expected
    """
    for other in [term, labelled_form]:
        a = labelled_form
        a -= other
        assert isinstance(a, LabelledForm)
        assert len(a) == 2

    b = labelled_form
    b -= None
    assert b == b


def test_mul_labelled_form(labelled_form):
    """
    test that we can multiply a labelled_form by a float and Constant
    """
    for coeff in [1., Constant(1.)]:
        new = coeff*labelled_form
        assert isinstance(new, LabelledForm)
        assert len(new) == len(labelled_form)
        assert not new.form == labelled_form.form


def test_label_map(labelled_form, label_a, label_x, form):
    """
    test that label_map returns a labelled_form with the label_map correctly
    applied
    """
    a = labelled_form + label_x(form)
    new = a.label_map(lambda t: t.has_label(label_a), lambda t: label_x(t, "z"), lambda t: label_x(t, "q"))
    assert isinstance(new, LabelledForm)
    assert all([t.get(label_x) == "z" for t in new if "a" in t.labels.keys()])
    assert all([t.get(label_x) == "q" for t in new if "a" not in t.labels.keys()])


def test_identity(labelled_form, label_a):
    """
    test that the identity function leaves the term unchanged
    """
    new = labelled_form.label_map(lambda t: t.has_label(label_a), identity)
    t_old = labelled_form.terms[0]
    t_new = new.terms[0]
    assert t_new == t_old


def test_drop(labelled_form, label_x, form):
    """
    test that the drop function drops the terms satisfying the term_filter
    """
    a = labelled_form + label_x(form)
    new = a.label_map(lambda t: t.has_label(label_x), drop)
    assert len(new) == 1
    assert not any([t.has_label(label_x) for t in new])


def test_all_terms(labelled_form, label_x):
    """
    test that the all_terms function cycles through all the terms
    """
    new = labelled_form.label_map(all_terms, map_if_false=drop)
    assert len(new) == len(labelled_form)
    new = labelled_form.label_map(all_terms, lambda t: label_x(t))
    assert all([t.has_label(label_x) for t in new])


def test_replace_test(V, labelled_form):
    """
    test that the replace_test function replaces the TestFunction in the form
    """
    replacer = Function(V)
    new = labelled_form.label_map(all_terms, replace_test(replacer))
    assert len(new.terms[0].form.arguments()) == 0


def test_replace_labelled(V, labelled_form, label_a, label_x):
    """
    test the replace_labelled(label, replacer) function on forms that
    are defined on a FunctionSpace:
    * if the term does not have this label then return a new instance
      of the term with identical form and labels
    * if replacer is a TrialFunction then replace the labelled item
      with replacer
    * if replacer is a Function then replace the labelled item
      with replacer
    * if replacer is a MixedFunction then replace the labelled item
      with the right part of replacer
    * if replacer is a ufl expression then replace the labelled item
      with replacer
    """
    replacer_tri = TrialFunction(V)
    replacer_fn = Function(V).interpolate(Constant(2.))
    t_old = labelled_form.terms[0]

    new = labelled_form.label_map(all_terms, replace_labelled(replacer_fn, label_x))
    t_new = new.terms[0]
    # check that we have a new instance of term
    assert not t_new == t_old
    # check that form and labels are the same
    assert t_new.form == t_old.form
    assert t_new.labels == t_old.labels

    a = labelled_form.label_map(all_terms, replace_labelled(replacer_tri, label_a))
    # check that the form now has 2 arguments because it has both test
    # and trial functions
    assert len(a.form.arguments()) == 2

    # check that L contains replacer_fn by solving
    # <test, trial> = <test, replacer_fn>
    L = labelled_form.label_map(all_terms, replace_labelled(replacer_fn, label_a))
    b = Function(V)
    prob = LinearVariationalProblem(a.form, L.form, b)
    solver = LinearVariationalSolver(prob)
    solver.solve()
    assert errornorm(replacer_fn, b) < 1.e-15

    replacer_expr = 2*replacer_fn
    # check that L contains replacer_expr by solving
    # <test, trial> = <test, replacer_fn>
    L = labelled_form.label_map(all_terms, replace_labelled(replacer_expr, label_a))
    prob = LinearVariationalProblem(a.form, L.form, b)
    solver = LinearVariationalSolver(prob)
    solver.solve()
    assert errornorm(replacer_expr, b) < 1.e-15


def test_replace_labelled_mixed(W, mixed_form, label_a, label_x):
    """
    test the replace_labelled(label, replacer) function on forms that
    are defined on a MixedFunctionSpace:
    * if replacer is TrialFunctions then replace the labelled item
      with
    * if replacer is a Function then replace the labelled item
      with replacer
    * if replacer is a ufl expression then replace the labelled item
      with replacer
    """
    replacer_tri = TrialFunctions(W)
    a = mixed_form.label_map(all_terms, replace_labelled(replacer_tri, label_a))
    # check that the form now has 2 arguments because it has both test
    # and trial functions
    assert len(a.form.arguments()) == 2

    # check that L contains replacer_fn by solving
    # <test, trial> = <test, replacer_mixed_fn>
    replacer_mixed_fn = Function(W)
    f, g = replacer_mixed_fn.split()
    f.interpolate(as_vector([1., 0.]))
    g.interpolate(Constant(2.))
    L = mixed_form.label_map(all_terms, replace_labelled(replacer_mixed_fn,
                                                         label_a))
    b = Function(W)
    prob = LinearVariationalProblem(a.form, L.form, b)
    solver = LinearVariationalSolver(prob)
    solver.solve()
    err = Function(W).assign(replacer_mixed_fn-b)
    err1, err2 = err.split()
    assert errornorm(replacer_mixed_fn, b) < 1.e-14
