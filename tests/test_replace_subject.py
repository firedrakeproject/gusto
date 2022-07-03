"""
Tests the replace_subject routine from labels.py
"""

from firedrake import (UnitSquareMesh, FunctionSpace, Function, TestFunction,
                       VectorFunctionSpace, MixedFunctionSpace, dx, inner,
                       TrialFunctions)
from gusto.fml import Label, all_terms
from gusto import subject, replace_subject
import pytest


@pytest.mark.parametrize('subject_type', ['normal', 'mixed', 'vector'])
@pytest.mark.parametrize('replacement_type', ['normal', 'mixed', 'vector', 'tuple'])
def test_replace_subject(subject_type, replacement_type):

    # ------------------------------------------------------------------------ #
    # Only certain combinations of options are valid
    # ------------------------------------------------------------------------ #

    if subject_type == 'vector' and replacement_type != 'vector':
        return True
    elif replacement_type == 'vector' and subject_type != 'vector':
        return True

    # ------------------------------------------------------------------------ #
    # Set up
    # ------------------------------------------------------------------------ #

    # Some basic labels
    foo_label = Label("foo")
    bar_label = Label("bar")

    # Create mesh, function space and forms
    n = 3
    mesh = UnitSquareMesh(n, n)
    V0 = FunctionSpace(mesh, "DG", 0)
    V1 = FunctionSpace(mesh, "CG", 1)
    V2 = VectorFunctionSpace(mesh, "DG", 0)
    Vmixed = MixedFunctionSpace((V0, V1))

    idx = None

    # ------------------------------------------------------------------------ #
    # Choose subject
    # ------------------------------------------------------------------------ #

    if subject_type == 'normal':
        V = V0
    elif subject_type == 'mixed':
        V = Vmixed
        if replacement_type == 'normal':
            idx = 0
    elif subject_type == 'vector':
        V = V2
    else:
        raise ValueError

    the_subject = Function(V)
    not_subject = Function(V)
    test = TestFunction(V)

    form_1 = inner(the_subject, test)*dx
    form_2 = inner(not_subject, test)*dx

    term_1 = foo_label(subject(form_1, the_subject))
    term_2 = bar_label(form_2)
    labelled_form = term_1 + term_2

    # ------------------------------------------------------------------------ #
    # Choose replacement
    # ------------------------------------------------------------------------ #

    if replacement_type == 'normal':
        V = V1
    elif replacement_type == 'mixed':
        V = Vmixed
        if subject_type != 'mixed':
            idx = 0
    elif replacement_type == 'vector':
        V = V2
    elif replacement_type == 'tuple':
        V = Vmixed
    else:
        raise ValueError

    the_replacement = Function(V)

    if replacement_type == 'tuple':
        the_replacement = TrialFunctions(Vmixed)
        if subject_type == 'normal':
            idx = 0

    # ------------------------------------------------------------------------ #
    # Test replace_subject
    # ------------------------------------------------------------------------ #

    new_labelled_form = labelled_form.label_map(
        lambda t: t.has_label(subject),
        map_if_true=replace_subject(the_replacement, idx=idx)
    )