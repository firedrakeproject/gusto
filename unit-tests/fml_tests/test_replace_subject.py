"""
Tests the replace_subject routine from labels.py
"""

from firedrake import (UnitSquareMesh, FunctionSpace, Function, TestFunction,
                       VectorFunctionSpace, dx, inner, split, grad)
from gusto.fml import (Label, subject, replace_subject, drop)
import pytest

from collections import namedtuple

ReplaceArgs = namedtuple("ReplaceArgs", "new_subj idxs error")

# some dummy labels
foo_label = Label("foo")
bar_label = Label("bar")

nx = 2
mesh = UnitSquareMesh(nx, nx)
V0 = FunctionSpace(mesh, 'CG', 1)
V1 = FunctionSpace(mesh, 'DG', 1)
W = V0*V1
Vv = VectorFunctionSpace(mesh, 'CG', 1)
Wv = Vv*V1


@pytest.fixture
def primal_form():
    primal_subj = Function(V0)
    primal_test = TestFunction(V0)

    primal_term1 = foo_label(subject(primal_subj*primal_test*dx, primal_subj))
    primal_term2 = bar_label(inner(grad(primal_subj), grad(primal_test))*dx)

    return primal_term1 + primal_term2


def primal_argsets():
    argsets = [
        ReplaceArgs(Function(V0), {}, None),
        ReplaceArgs(Function(V0), {'new_idx': 0}, ValueError),
        ReplaceArgs(Function(V0), {'old_idx': 0}, ValueError),
        ReplaceArgs(Function(W), {'new_idx': 0}, None),
        ReplaceArgs(Function(W), {'new_idx': 1}, None),
        ReplaceArgs(split(Function(W)), {'new_idx': 1}, None),
        ReplaceArgs(Function(W), {'old_idx': 0}, ValueError),
        ReplaceArgs(Function(W), {'new_idx': 7}, IndexError),
    ]
    return argsets


@pytest.fixture
def mixed_form():
    mixed_subj = Function(W)
    mixed_test = TestFunction(W)

    mixed_subj0, mixed_subj1 = split(mixed_subj)
    mixed_test0, mixed_test1 = split(mixed_test)

    mixed_term1 = foo_label(subject(mixed_subj0*mixed_test0*dx, mixed_subj))
    mixed_term2 = bar_label(inner(grad(mixed_subj1), grad(mixed_test1))*dx)

    return mixed_term1 + mixed_term2


def mixed_argsets():
    argsets = [
        ReplaceArgs(Function(W), {}, None),
        ReplaceArgs(Function(W), {'new_idx': 0, 'old_idx': 0}, None),
        ReplaceArgs(Function(W), {'old_idx': 0}, ValueError),
        ReplaceArgs(Function(W), {'new_idx': 0}, ValueError),
        ReplaceArgs(Function(V0), {'old_idx': 0}, None),
        ReplaceArgs(Function(V0), {'new_idx': 0}, ValueError),
        ReplaceArgs(split(Function(W)), {'new_idx': 0, 'old_idx': 0}, None),
    ]
    return argsets


@pytest.fixture
def vector_form():
    vector_subj = Function(Vv)
    vector_test = TestFunction(Vv)

    vector_term1 = foo_label(subject(inner(vector_subj, vector_test)*dx, vector_subj))
    vector_term2 = bar_label(inner(grad(vector_subj), grad(vector_test))*dx)

    return vector_term1 + vector_term2


def vector_argsets():
    argsets = [
        ReplaceArgs(Function(Vv), {}, None),
        ReplaceArgs(Function(V0), {'new_idx': 0}, ValueError),
        ReplaceArgs(Function(V0), {'old_idx': 0}, ValueError),
        ReplaceArgs(Function(Wv), {'new_idx': 0}, None),
        ReplaceArgs(Function(Wv), {'new_idx': 1}, ValueError),
        ReplaceArgs(split(Function(Wv)), {'new_idx': 0}, None),
        ReplaceArgs(Function(W), {'old_idx': 0}, ValueError),
        ReplaceArgs(Function(W), {'new_idx': 7}, IndexError),
    ]
    return argsets


@pytest.mark.parametrize('argset', primal_argsets())
def test_replace_subject_primal(primal_form, argset):
    new_subj = argset.new_subj
    idxs = argset.idxs
    error = argset.error

    if error is None:
        old_subj = primal_form.form.coefficients()[0]

        new_form = primal_form.label_map(
            lambda t: t.has_label(foo_label),
            map_if_true=replace_subject(new_subj, **idxs),
            map_if_false=drop)

        # what if we only replace part of the subject?
        if 'new_idx' in idxs:
            split_new = new_subj if type(new_subj) is tuple else split(new_subj)
            new_subj = split_new[idxs['new_idx']].ufl_operands[0]

        assert new_subj in new_form.form.coefficients()
        assert old_subj not in new_form.form.coefficients()

    else:
        with pytest.raises(error):
            new_form = primal_form.label_map(
                lambda t: t.has_label(foo_label),
                map_if_true=replace_subject(new_subj, **idxs))


@pytest.mark.parametrize('argset', mixed_argsets())
def test_replace_subject_mixed(mixed_form, argset):
    new_subj = argset.new_subj
    idxs = argset.idxs
    error = argset.error

    if error is None:
        old_subj = mixed_form.form.coefficients()[0]

        new_form = mixed_form.label_map(
            lambda t: t.has_label(foo_label),
            map_if_true=replace_subject(new_subj, **idxs),
            map_if_false=drop)

        # what if we only replace part of the subject?
        if 'new_idx' in idxs:
            split_new = new_subj if type(new_subj) is tuple else split(new_subj)
            new_subj = split_new[idxs['new_idx']].ufl_operands[0]

        assert new_subj in new_form.form.coefficients()
        assert old_subj not in new_form.form.coefficients()

    else:
        with pytest.raises(error):
            new_form = mixed_form.label_map(
                lambda t: t.has_label(foo_label),
                map_if_true=replace_subject(new_subj, **idxs))


@pytest.mark.parametrize('argset', vector_argsets())
def test_replace_subject_vector(vector_form, argset):
    new_subj = argset.new_subj
    idxs = argset.idxs
    error = argset.error

    if error is None:
        old_subj = vector_form.form.coefficients()[0]

        new_form = vector_form.label_map(
            lambda t: t.has_label(foo_label),
            map_if_true=replace_subject(new_subj, **idxs),
            map_if_false=drop)

        # what if we only replace part of the subject?
        if 'new_idx' in idxs:
            split_new = new_subj if type(new_subj) is tuple else split(new_subj)
            new_subj = split_new[idxs['new_idx']].ufl_operands[0].ufl_operands[0]

        assert new_subj in new_form.form.coefficients()
        assert old_subj not in new_form.form.coefficients()

    else:
        with pytest.raises(error):
            new_form = vector_form.label_map(
                lambda t: t.has_label(foo_label),
                map_if_true=replace_subject(new_subj, **idxs))
