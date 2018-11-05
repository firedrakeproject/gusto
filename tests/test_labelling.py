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


@pytest.fixture
def eq(term, label_a, label_x_is_y):
    return label_a(term) + label_x_is_y(term)


def test_label_form(label_a, labelled_form, form):
    assert(isinstance(labelled_form, LabelledForm))
    assert(labelled_form[0].has_label(label_a))
    assert(labelled_form[0].form == form)


def test_label_term(label_a, term, form):
    labelled_term = label_a(term)
    assert(isinstance(labelled_term, Term))
    assert(labelled_term.has_label(label_a))
    assert(labelled_term.form == form)


def test_label_equation(labelled_form, label_a, label_x_is_y, label_x_is_z):
    eqn = label_x_is_y(labelled_form)
    assert(isinstance(eqn, LabelledForm))
    assert(eqn[0].has_label(label_a, label_x_is_y))
    assert(eqn[0].labels["x"] == "y")
    eqn = label_x_is_z(eqn)
    assert(eqn[0].labels["x"] == "z")


def test_add_term(eq):
    assert(isinstance(eq, LabelledForm))
    assert(len(eq) == 2)


def test_add_equation(eq):
    eq += eq
    assert(isinstance(eq, LabelledForm))
    assert(len(eq) == 4)


def test_add_term_and_equation(term, eq):
    eq += term
    assert(isinstance(eq, LabelledForm))
    assert(len(eq) == 3)
    a = term + eq
    assert(isinstance(a, LabelledForm))
    assert(len(a) == 4)


def test_label_map(eq, label_x_is_y, label_x_is_z):
    new_eq = eq.label_map(lambda t: t.has_label(label_x_is_y), label_x_is_z)
    assert(t.labels["x"] is "z" for t in new_eq if "x" in t.labels.keys())
