"""
Tests the replace_subject routine from labels.py
"""

from firedrake import (UnitSquareMesh, FunctionSpace, Function, TestFunction, TestFunctions,
                       VectorFunctionSpace, MixedFunctionSpace, dx, inner,
                       TrialFunctions, TrialFunction, split, grad)
from gusto.fml import (Label, subject, replace_subject, replace_test_function,
                       replace_trial_function, drop)
import pytest

from collections import namedtuple

ReplaceArgs = namedtuple("ReplaceArgs", "subject idxs error")

# some dummy labels
foo_label = Label("foo")
bar_label = Label("bar")

nx = 2
mesh = UnitSquareMesh(nx, nx)

V0 = FunctionSpace(mesh, 'CG', 1)
V1 = FunctionSpace(mesh, 'DG', 1)

W = V0*V1

subj = Function(V0)
v = TestFunction(V0)

term1 = foo_label(subject(subj*v*dx, subj))
term2 = bar_label(inner(grad(subj), grad(v))*dx)

labelled_form = term1 + term2

argsets = [
    ReplaceArgs(Function(V0), {}, None),
    ReplaceArgs(Function(V0), {'new_idx': 0}, ValueError),
    ReplaceArgs(Function(V0), {'old_idx': 0}, ValueError),
    ReplaceArgs(Function(W), {'new_idx': 0}, None),
    ReplaceArgs(Function(W), {'new_idx': 1}, None),
    ReplaceArgs(Function(W), {'old_idx': 0}, ValueError),
    ReplaceArgs(Function(W), {'new_idx': 7}, IndexError),
]


@pytest.mark.parametrize('argset', argsets)
def test_replace_subject_params(argset):
    arg = argset.subject
    idxs = argset.idxs
    error = argset.error

    if error is None:
        new_form = labelled_form.label_map(
            lambda t: t.has_label(foo_label),
            map_if_true=replace_subject(arg, **idxs),
            map_if_false=drop)
        assert arg == new_form.form.coefficients()[0]
        assert subj not in new_form.form.coefficients()

    else:
        with pytest.raises(error):
            new_form = labelled_form.label_map(
                lambda t: t.has_label(foo_label),
                map_if_true=replace_subject(arg, **idxs))


def test_replace_subject_primal():
    # setup some basic labels
    foo_label = Label("foo")
    bar_label = Label("bar")

    # setup the mesh and function space
    n = 2
    mesh = UnitSquareMesh(n, n)
    V = FunctionSpace(mesh, "CG", 1)

    # set up the form
    u = Function(V)
    v = TestFunction(V)

    form1 = inner(u, v)*dx
    form2 = inner(grad(u), grad(v))*dx

    term1 = foo_label(subject(form1, u))
    term2 = bar_label(form2)

    labelled_form = term1 + term2

    # replace with another function
    w = Function(V)

    # this should work
    new_form = labelled_form.label_map(
        lambda t: t.has_label(foo_label),
        map_if_true=replace_subject(w))

    # these should fail if given an index
    with pytest.raises(ValueError):
        new_form = labelled_form.label_map(
            lambda t: t.has_label(foo_label),
            map_if_true=replace_subject(w, new_idx=0))

    with pytest.raises(ValueError):
        new_form = labelled_form.label_map(
            lambda t: t.has_label(foo_label),
            map_if_true=replace_subject(w, old_idx=0))

    with pytest.raises(ValueError):
        new_form = labelled_form.label_map(
            lambda t: t.has_label(foo_label),
            map_if_true=replace_subject(w, old_idx=0, new_idx=0))

    # replace with mixed component
    wm = Function(V*V)
    wms = split(wm)
    wm0, wm1 = wms

    new_form = labelled_form.label_map(
        lambda t: t.has_label(foo_label),
        map_if_true=replace_subject(wm0))

    new_form = labelled_form.label_map(
        lambda t: t.has_label(foo_label),
        map_if_true=replace_subject(wms, new_idx=0))


def test_replace_subject_mixed():
    # setup some basic labels
    foo_label = Label("foo")
    bar_label = Label("bar")

    # setup the mesh and function space
    n = 2
    mesh = UnitSquareMesh(n, n)
    V0 = FunctionSpace(mesh, "CG", 1)
    V1 = FunctionSpace(mesh, "DG", 1)
    W = V0*V1

    # set up the form
    u = Function(W)
    u0, u1 = split(u)
    v0, v1 = TestFunctions(W)

    form1 = inner(u0, v0)*dx
    form2 = inner(grad(u1), grad(v1))*dx

    term1 = foo_label(subject(form1, u))
    term2 = bar_label(form2)

    labelled_form = term1 + term2

    # replace with another function
    w = Function(W)

    # replace all parts of the subject
    new_form = labelled_form.label_map(
        lambda t: t.has_label(foo_label),
        map_if_true=replace_subject(w))

    # replace either part of the subject
    new_form = labelled_form.label_map(
        lambda t: t.has_label(foo_label),
        map_if_true=replace_subject(w, old_idx=0, new_idx=0))

    new_form = labelled_form.label_map(
        lambda t: t.has_label(foo_label),
        map_if_true=replace_subject(w, old_idx=1, new_idx=1))

    # these should fail if given only one index
    with pytest.raises(ValueError):
        new_form = labelled_form.label_map(
            lambda t: t.has_label(foo_label),
            map_if_true=replace_subject(w, old_idx=1))

    with pytest.raises(ValueError):
        new_form = labelled_form.label_map(
            lambda t: t.has_label(foo_label),
            map_if_true=replace_subject(w, new_idx=1))

    # try indexing only one
    w0, w1 = split(w)

    # replace a specific part of the subject
    new_form = labelled_form.label_map(
        lambda t: t.has_label(foo_label),
        map_if_true=replace_subject(w1, old_idx=0))

    # replace with something from a primal space
    wp = Function(V0)
    new_form = labelled_form.label_map(
        lambda t: t.has_label(foo_label),
        map_if_true=replace_subject(wp, old_idx=1))


replace_funcs = [
    pytest.param((Function, replace_subject), id="replace_subj"),
    pytest.param((TestFunction, replace_test_function), id="replace_test"),
    pytest.param((TrialFunction, replace_trial_function), id="replace_trial")
]


@pytest.mark.parametrize('subject_type', ['normal', 'mixed', 'vector'])
@pytest.mark.parametrize('replacement_type', ['normal', 'mixed', 'vector', 'tuple'])
@pytest.mark.parametrize('function_or_indexed', ['function', 'indexed'])
@pytest.mark.parametrize('replace_func', replace_funcs)
def old_test_replace_subject(subject_type, replacement_type, function_or_indexed, replace_func):

    # ------------------------------------------------------------------------ #
    # Only certain combinations of options are valid
    # ------------------------------------------------------------------------ #

    # only makes sense to replace a vector with a vector
    if (subject_type == 'vector') ^ (replacement_type == 'vector'):
        pytest.skip("Invalid vector option combination")

    # ------------------------------------------------------------------------ #
    # Set up
    # ------------------------------------------------------------------------ #

    # Some basic labels
    foo_label = Label("foo")
    bar_label = Label("bar")

    # Create mesh, function space and forms
    n = 2
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
    not_subject = TrialFunction(V)
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

    FunctionType = replace_func[0]

    the_replacement = FunctionType(V)

    if function_or_indexed == 'indexed' and replacement_type != 'vector':
        the_replacement = split(the_replacement)

        if len(the_replacement) == 1:
            the_replacement = the_replacement[0]

    if replacement_type == 'tuple':
        the_replacement = TrialFunctions(Vmixed)
        if subject_type == 'normal':
            idx = 0

    # ------------------------------------------------------------------------ #
    # Test replace_subject
    # ------------------------------------------------------------------------ #

    replace_map = replace_func[1]

    if replace_map is replace_trial_function:
        match_label = bar_label
    else:
        match_label = subject

    labelled_form = labelled_form.label_map(
        lambda t: t.has_label(match_label),
        map_if_true=replace_map(the_replacement, idx=idx)
    )

    # also test indexed
    if subject_type == 'mixed' and function_or_indexed == 'indexed':
        idx = 0
        the_replacement = split(FunctionType(Vmixed))[idx]

        labelled_form = labelled_form.label_map(
            lambda t: t.has_label(match_label),
            map_if_true=replace_map(the_replacement, idx=idx)
        )
