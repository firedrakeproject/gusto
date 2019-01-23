import pytest
from firedrake import (TestFunction, Function, FunctionSpace,
                       VectorFunctionSpace,
                       MixedFunctionSpace, UnitSquareMesh,
                       dx, Constant, LinearVariationalProblem,
                       LinearVariationalSolver)
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
def mixed_form(mesh, mixed_function, W):
    sigma, phi = TestFunctions(W)
    x, y = mixed_function.split()
    return index(inner(sigma, x)*dx, 0) + index(phi*y*dx, 1)


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
    assert all([t.get("x") == "y" for t in new])
    new = label_x(new, "z")
    assert all([t.get("x") == "z" for t in new])


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


def test_label_map(labelled_form, label_a, label_x, form):
    """
    test that label_map returns a labelled_form with the label_map correctly
    applied
    """
    a = labelled_form + label_x(form)
    new = a.label_map(lambda t: t.has_label(label_a), lambda t: label_x(t, "z"), lambda t: label_x(t, "q"))
    assert isinstance(new, LabelledForm)
    assert all([t.get("x") == "z" for t in new if "a" in t.labels.keys()])
    assert all([t.get("x") == "q" for t in new if "a" not in t.labels.keys()])


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


def test_extract(mixed_form, mixed_function, V):
    """
    test that the extract function interacts correctly with the index label
    """
    new = Function(V)
    a = subject(mixed_form, mixed_function)
    a = a.label_map(lambda t: t.get("index") == 1, extract(1), drop)
    a_old = a
    # this would fail due to the shape mismatch is there were any
    # terms still containing subject.split()[0]...
    a = a.label_map(
        all_terms, lambda t: Term(
            ufl.replace(t.form, {t.get("subject").split()[0]: new}),
            t.labels)
    )
    # ...instead nothing has happened
    assert a.form == a_old.form
    # this works...
    a = a.label_map(
        all_terms, lambda t: Term(
            ufl.replace(t.form, {t.get("subject").split()[1]: new}),
            t.labels)
    )
    # ...and the form has changed
    assert not a.form == a_old.form


def test_replace_test(V, labelled_form):
    """
    test that the replace_test function replaces the TestFunction in the form
    """
    replacer = Function(V)
    new = labelled_form.label_map(all_terms, replace_test(replacer))
    assert len(new.terms[0].form.arguments()) == 0


def test_replace_labelled(V, labelled_form, label_a, label_x, form):
    """
    test the replace_labelled(label, replacer) function:
    * if the term does not have this label then return a new instance
      of the term with identical form and labels
    * if replacer is a TrialFunction then replace the labelled item
      with replacer
    * if replacer is a Function then replace the labelled item
      with replacer
    * if replacer is a ufl expression then replace the labelled item
      with replacer
    """
    replacer_tri = TrialFunction(V)
    replacer_fn = Function(V).interpolate(Constant(2.))
    t_old = labelled_form.terms[0]

    new = labelled_form.label_map(all_terms, replace_labelled("x", replacer_fn))
    t_new = new.terms[0]
    # check that we have a new instance of term
    assert not t_new == t_old
    # check that form and labels are the same
    assert t_new.form == t_old.form
    assert t_new.labels == t_old.labels

    a = labelled_form.label_map(all_terms, replace_labelled("a", replacer_tri))
    # check that the form now has 2 arguments because it has both test
    # and trial functions
    assert len(a.form.arguments()) == 2

    # check that L contains replacer_fn by solving
    # <test, trial> = <test, replacer_fn>
    L = labelled_form.label_map(all_terms, replace_labelled("a", replacer_fn))
    b = Function(V)
    prob = LinearVariationalProblem(a.form, L.form, b)
    solver = LinearVariationalSolver(prob)
    solver.solve()
    err = Function(V).assign(replacer_fn-b)
    assert err.dat.data.max() < 1.e-15

    replacer_expr = 2*replacer_fn
    # check that L contains replacer_expr by solving
    # <test, trial> = <test, replacer_fn>
    L = labelled_form.label_map(all_terms, replace_labelled("a", replacer_expr))
    prob = LinearVariationalProblem(a.form, L.form, b)
    solver = LinearVariationalSolver(prob)
    solver.solve()
    err = Function(V).assign(replacer_expr-b)
    assert err.dat.data.max() < 1.e-14
