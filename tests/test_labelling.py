import pytest
from firedrake import TestFunction, Function, FunctionSpace, UnitSquareMesh, dx
from gusto import *


@pytest.fixture
def label_a():
    return Label("a")


@pytest.fixture
def label_x_is_y():
    return Label("x", "y")


@pytest.fixture
def label_x_is_z():
    return Label("x", "z")


@pytest.fixture
def form():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    phi = TestFunction(V)
    f = Function(V)
    return phi*f*dx


@pytest.fixture
def labelled_form(label_a, form):
    return label_a(form)


@pytest.fixture
def term(form):
    return Term(form)


def test_label_form(label_a, labelled_form, form):
    """
    tests that labelling a form returns a LabelledForm with the correct
    label and form
    """
    assert(isinstance(labelled_form, LabelledForm))
    assert(labelled_form[0].has_label(label_a))
    assert(labelled_form[0].form == form)


def test_label_term(label_a, term, form):
    """
    tests that labelling a term returns another term with the correct label
    and form
    """
    labelled_term = label_a(term)
    assert(isinstance(labelled_term, Term))
    assert(labelled_term.has_label(label_a))
    assert(labelled_term.form == form)


def test_label_labelled_form(labelled_form, label_a, label_x_is_y, label_x_is_z):
    """
    test that labelling a labelled_form returns a labelled_form with the
    correct labels and that the label can be changed.
    """
    new = label_x_is_y(labelled_form)
    assert(isinstance(new, LabelledForm))
    assert([t.has_label(label_a, label_x_is_y) for t in new])
    assert([t.labels["x"] == "y" for t in new])
    new = label_x_is_z(new)
    assert([t.labels["x"] == "z" for t in new])


def test_add_term(term, labelled_form):
    """
    test that adding to a term works as expected
    """
    a = term
    a += term
    assert(isinstance(a, LabelledForm))
    assert(len(a) == 2)
    b = term
    b += None
    assert(b == term)
    b = None
    b += term
    assert(b == term)
    c = term
    c += labelled_form
    assert(isinstance(c, LabelledForm))
    assert(len(c) == 2)


def test_add_labelled_form(term, labelled_form):
    """
    test that adding to a labelled form works as expected
    """
    a = labelled_form
    a += labelled_form
    assert(isinstance(a, LabelledForm))
    assert(len(a) == 2)
    b = labelled_form + term
    assert(isinstance(a, LabelledForm))
    assert(len(b) == 2)
    c = labelled_form
    c += None
    assert(c == c)


def test_label_map(labelled_form, label_x_is_y, label_x_is_z):
    """
    test that label_map returns a labelled_form with the label_map correctly
    applied
    """
    new = labelled_form.label_map(lambda t: t.has_label(label_x_is_y), label_x_is_z)
    assert(isinstance(new, LabelledForm))
    assert(t.labels["x"] is "z" for t in new if "x" in t.labels.keys())
