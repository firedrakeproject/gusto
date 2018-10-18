import pytest
from firedrake import TestFunction, Function, FunctionSpace, UnitSquareMesh, dx
from gusto import *


@pytest.fixture
def mass():
    mass = Label("mass")
    return mass


@pytest.fixture
def implicit():
    implicit = Label("time", "implicit")
    return implicit


@pytest.fixture
def mass_form():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    phi = TestFunction(V)
    f = Function(V)
    mass_form = phi*f*dx
    return mass_form


@pytest.fixture
def mass_term(mass, mass_form):
    mass_term = mass(mass_form)
    return mass_term


@pytest.fixture
def eq(mass_term):
    eq = mass_term + mass_term
    return eq

def test_label_form(mass_term, mass_form):
    assert(isinstance(mass_term, Equation))
    assert(mass_term[0].has_label(mass))
    assert(mass_term[0].form == mass_form)


def test_label_term(mass_term, implicit, mass):
    mass_term = implicit(mass_term)
    assert(isinstance(mass_term, Equation))
    assert(mass_term[0].has_label(mass))
    assert(mass_term[0].labels[time.label] == "implicit")
    explicit = Label("time", "explicit")
    mass_term = explicit(mass_term)
    assert(mass_term[0].labels[time.label] == "explicit")


def test_add_term(eq):
    assert(isinstance(eq, Equation))
    assert(len(eq) == 2)


def test_add_equation(eq):
    eq += eq
    assert(isinstance(eq, Equation))
    assert(len(eq) == 4)


def test_add_term_and_equation(mass_term, eq):
    eq += mass_term
    assert(isinstance(eq, Equation))
    assert(len(eq) == 3)
    a = mass_term + eq
    assert(isinstance(a, Equation))
    assert(len(a) == 4)
