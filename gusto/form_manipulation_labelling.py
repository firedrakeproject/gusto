import ufl
import functools
import operator
from firedrake import Function, TrialFunction, Constant
from firedrake.formmanipulation import split_form


identity = lambda t: t
drop = lambda t: None
all_terms = lambda t: True
extract = lambda idx: lambda t: Term(split_form(t.form)[idx].form, t.labels)


def has_labels(*labels):
    """:arg labels: the :class:`Label` object(s)

    Returns a function that takes in a :class:`Term` object and checks
    whether is has the label(s).
    """

    def t_has_labels(t):
        if len(labels) == 1:
            return t.has_label(labels[0])
        else:
            return any(t.has_label(*labels))

    return t_has_labels


def linearise(t):
    """
    :arg t: a :class:`.Term` object

    Returns a :class:`.LabelledForm` object.

    The linearisation of t must have been specified by calling the
    :class:`.linearisation` on t. Returns a labelled form with terms that
    have the linear form from l=t.get("linearisation") and inherit labels
    from both t and l, with the labels from l overwriting those from t if
    both present.
    """
    t_linear = t.get(linearisation)
    assert type(t_linear) == LabelledForm
    new_labels = t.labels.copy()
    new_labels["linearisation"] = True
    return LabelledForm(functools.reduce(
        operator.add,
        [Term(l.form, dict(new_labels, **l.labels))
         for l in t_linear]))


def replace_test(new_test):
    """
    :arg new_test: a :func:`TestFunction`

    Returns a function that takes in t, a :class:`Term`, and returns
    a new :class:`Term` with form containing the new_test and
    labels=t.labels
    """

    def rep(t):
        test = t.form.arguments()[0]
        new_form = ufl.replace(t.form, {test: new_test})
        return Term(new_form, t.labels)

    return rep


def replace_labelled(replacer_in, *labels):
    """
    :arg labels: :class:`Label` objects specifying the label(s) of the
    part of the form that is to be replaced
    :arg replacer: :class:`ufl.core.expr.Expr`, :func:`Function`,
    :func:`TrialFunction`, or tuple thereof, or a :class:`Label` object.

    Returns a function that takes in t, a :class:`Term`, and returns
    a new :class:`Term` where the part of the form labelled by label
    has been replaced by (possibly part of) the replacer. The labels
    remain the same, unless the replacer is another labelled part of
    the form, in which case the original label is removed.
    """

    def rep(t):

        # if replacer_in is aa label, we need to grab that part of the
        # form and flag that the original label needs to be removed
        if type(replacer_in) is Label:
            replacer = t.get(replacer_in)
            remove_labels = True
        else:
            replacer = replacer_in
            remove_labels = False

        # dictionary to pass to ufl.replace
        replace_dict = {}

        for label in labels:
            # get the thing we're going to replace
            labelled = t.get(label)

            # return a new term with form and labels identical to t if
            # no part of the form is labelled with label
            if labelled is None:
                return Term(t.form, t.labels)

            # if t.get(label) returns True then the label does not
            # apply to part of the form so we cannot replace it.
            if type(labelled) is bool:
                raise ValueError("This label does not label a part of this term's form")

            # flag if replacer is a ufl expression - if it is then we
            # trust it is of the right shape below
            repl_expr = (
                isinstance(replacer, ufl.core.expr.Expr)
                and not isinstance(replacer, Function)
            )

            # size returns either the length of a tuple or the length
            # of a function's function space
            size = lambda q: len(q) if type(q) is tuple else len(q.function_space())

            # if either replacer or labelled are Functions defined on
            # a mixed function space then split them to get a tuple of
            # functions
            if type(replacer) is Function and size(replacer) > 1:
                new = replacer.split()
            else:
                new = replacer
            if type(labelled) is Function and size(labelled) > 1:
                old = labelled.split()
            else:
                old = labelled

            # constract the replace_dict = {old: new}, comparing the
            # sizes of the part to be replaced and its replacement to
            # figure out if we need to extract just one component of
            # either of them
            if size(old) == 1:
                if not repl_expr and size(new) > 1:
                    new = new[old.function_space().index]
                replace_dict[old] = new
            else:
                if repl_expr or size(new) == 1:
                    replace_dict[old[new.function_space().index]] = new
                else:
                    for k, v in zip(old, new):
                        replace_dict[k] = v

        # construct new form and term
        new_form = ufl.replace(t.form, replace_dict)
        new_term = Term(new_form, t.labels)

        # if we have replaced a labelled part of the form with another
        # labelled part then we need to remove the original label
        if remove_labels:
            for label in labels:
                new_term = label.remove(new_term)

        return new_term

    return rep


class Term(object):
    """
    Term class, containing a form and its labels.

    :arg form: the form for this term
    :arg label_dict: dictionary of key-value pairs corresponding to current form labels.
    """
    __slots__ = ["form", "labels"]

    def __init__(self, form, label_dict=None):
        self.form = form
        self.labels = label_dict or {}

    def get(self, label, default=None):
        return self.labels.get(label.label)

    def has_label(self, *labels):
        if len(labels) == 1:
            return labels[0].label in self.labels
        else:
            return tuple(self.has_label(l) for l in labels)

    def __add__(self, other):
        if other is None:
            return self
        elif isinstance(other, Term):
            return LabelledForm(self, other)
        elif isinstance(other, LabelledForm):
            return LabelledForm(self, *other.terms)
        else:
            return NotImplemented

    __radd__ = __add__

    def __mul__(self, other):
        if type(other) is float:
            other = Constant(other)
        elif type(other) is not Constant:
            return NotImplemented
        return Term(other*self.form, self.labels)

    __rmul__ = __mul__

class LabelledForm(object):
    __slots__ = ["terms"]

    def __init__(self, *terms):
        if len(terms) == 1 and isinstance(terms[0], LabelledForm):
            self.terms = terms[0].terms
        else:
            self.terms = list(terms)

    def __add__(self, other):
        if isinstance(other, ufl.Form):
            return LabelledForm(*self, Term(other))
        elif type(other) is Term:
            return LabelledForm(*self, other)
        elif type(other) is LabelledForm:
            return LabelledForm(*self, *other)
        elif other is None:
            return self
        else:
            return NotImplemented

    def __sub__(self, other):
        if type(other) is Term:
            return LabelledForm(*self, Constant(-1.)*other)
        elif type(other) is LabelledForm:
            return LabelledForm(*self, *[Constant(-1.)*t for t in other])
        elif other is None:
            return self
        else:
            return NotImplemented

    def __mul__(self, other):
        if type(other) is float:
            other = Constant(other)
        elif type(other) is not Constant:
            return NotImplemented
        return self.label_map(all_terms, lambda t: Term(other*t.form, t.labels))

    __rmul__ = __mul__

    def __iter__(self):
        return iter(self.terms)

    def __len__(self):
        return len(self.terms)

    def label_map(self, term_filter, map_if_true=identity,
                  map_if_false=identity):
        """Return a new equation in which terms for which
        `term_filter` is `True` are transformed by
        `map_if_true`; terms for which `term_filter` is false are
        transformed by map_is_false."""

        return LabelledForm(
            functools.reduce(operator.add,
                             filter(lambda t: t is not None,
                                    (map_if_true(t) if term_filter(t) else
                                     map_if_false(t) for t in self.terms))))

    @property
    def form(self):
        return functools.reduce(operator.add, (t.form for t in self.terms))


class Label(object):
    """
    Class providing labelling functionality for Gusto terms and equations

    :arg label: str giving the name of the label.
    :arg value: str providing the value of the label. Defaults to True.
    :arg validator: (optional) function to check the validity of any
    value later passed to __call__
    """
    __slots__ = ["label", "default_value", "value", "validator"]

    def __init__(self, label, *, value=True, validator=None):
        self.label = label
        self.default_value = value
        self.validator = validator

    def __call__(self, target, value=None):
        # if value is provided, check that we have a validator function
        # and validate the value, otherwise use default value
        if value is not None:
            assert(self.validator)
            assert(self.validator(value))
            self.value = value
        else:
            self.value = self.default_value
        if isinstance(target, LabelledForm):
            return LabelledForm(*(self(t, value) for t in target.terms))
        elif isinstance(target, ufl.Form):
            return LabelledForm(Term(target, {self.label: self.value}))
        elif isinstance(target, Term):
            new_labels = target.labels.copy()
            new_labels.update({self.label: self.value})
            return Term(target.form, new_labels)
        else:
            raise ValueError("Unable to label %s" % target)

    def remove(self, target):
        """Remove any :form:`Label` with this `label` from
        `target`. If called on an :class:`LabelledForm`, act termwise."""
        if isinstance(target, LabelledForm):
            return LabelledForm(*(self.remove(t) for t in target.terms))
        elif isinstance(target, Term):
            try:
                d = target.labels.copy()
                d.pop(self.label)
                return Term(target.form, d)
            except KeyError:
                return target
        else:
            raise ValueError("Unable to unlabel %s" % target)


time_derivative = Label("time_derivative")
advection = Label("advection")
diffusion = Label("diffusion")
explicit = Label("explicit")
implicit = Label("implicit")
advecting_velocity = Label("uadv", validator=lambda value: type(value) == Function)
subject = Label("subject", validator=lambda value: type(value) in [TrialFunction, Function, ufl.tensors.ListTensor, ufl.indexed.Indexed])
linearisation = Label("linearisation", validator=lambda value: type(value) is LabelledForm)
index = Label("index", validator=lambda value: type(value) is int)
