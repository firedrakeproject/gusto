"""
Tests FML's Label objects.
"""

from firedrake import IntervalMesh, FunctionSpace, Function, TestFunction
from gusto.configuration import TransportEquationType
from gusto.fml import Label, LabelledForm, Term
from ufl import Form
import pytest

@pytest.fixture
def label(label_type):
    # Returns labels with different value validation

    if label_type == "boolean":
        # A label that is simply a string, whose value is Boolean
        return Label("foo")

    elif label_type == "integer":
        # A label whose value is an integer
        return Label("foo", validator=lambda value: (type(value) == int and value < 9))

    elif label_type == "other":
        # A label whose value is some other type
        return Label("foo", validator=type(value) == TransportEquationType)

    elif label_type == "function":
        # A label whose value is an Function
        return Label("foo", type(value) == Function)


@pytest.fixture
def object_to_label(object_type):
    # A series of different objects to be labelled

    if object_type == int:
        return 10

    else:
        # Create mesh and function space
        L = 3.0
        n = 3
        mesh = IntervalMesh(n, L)
        V = FunctionSpace(mesh, "DG", 0)
        f = Function(V)
        g = TestFunction(V)
        form = f*g*dx
        term = Term(form)

        if object_type == Form:
            return form

        elif object_type == Term:
            return term

        elif object_type == LabelledForm:
            return LabelledForm(term)

        else:
            raise ValueError(f'object_type {object_type} not implemented')


@pytest.mark.parametrize("label_type", ["boolean", "integer", "other", "function"],
                         "object_to_label", [LabelledForm, Term, Form, int])
def test_label(label_type, label, object_type, object_to_label):

    assert label.label == "foo"

    labelled_object = label(object_to_label)

    import pdb; pdb.set_trace()
