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


def test_label(mass, implicit):
    assert(mass.mass)
    assert(implicit.time == "implicit")


def test_label_form(mass, mass_form):
    mass_term = mass(mass_form)
    assert(isinstance(mass_term, Term))
    assert(mass_term.labels["mass"])
    assert(mass_term.form == mass_form)
