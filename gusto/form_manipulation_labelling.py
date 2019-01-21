import ufl
import functools
import operator
from firedrake import Function, TrialFunction, Constant, split
from firedrake.formmanipulation import split_form


identity = lambda t: t
drop = lambda t: None
all_terms = lambda t: True
extract = lambda idx: lambda t: Term(split_form(t.form)[idx].form, t.labels)


def linearise(t):
    t_linear = t.get("linearisation")
    if type(t_linear) == LabelledForm:
        new_labels = t.labels.copy()
        new_labels["linearisation"] = True
        return LabelledForm(functools.reduce(
            operator.add,
            [Term(l.form, dict(new_labels, **l.labels))
             for l in t_linear]))
    elif t_linear is True:
        return Term(t.form, t.labels)


def replace_test(new_test):

    def rep(t):
        test = t.form.arguments()[0]
        new_form = ufl.replace(t.form, {test: new_test})
        return Term(new_form, t.labels)

    return rep


def replace_labelled(label, replacer):

    def rep(t):
        old = t.labels[label]
        # check if we need to extract part of the replacer
        repl_expr = isinstance(replacer, ufl.core.expr.Expr)
        size = lambda q: len(q) if type(q) is tuple else len(q.function_space())
        extract = lambda q, idx: q[idx] if type(q) is tuple else q.split()[idx]
        if size(old) == 1:
            if repl_expr:
                new = replacer
            elif size(replacer) == 1:
                new = replacer
            else:
                new = extract(replacer, old.function_space().index)
            replace_dict = {old: new}
        else:
            if repl_expr:
                new = replacer
                replace_dict = {extract(old, new.function_space().index): new}
            elif size(replacer) == 1:
                new = replacer
                replace_dict = {extract(old, new.function_space().index): new}
            else:
                if type(replacer) is Function:
                    new = split(replacer)
                else:
                    new = replacer
                replace_dict = {k: v for k, v in zip(old.split(), new)}
        new_form = ufl.replace(t.form, replace_dict)
        return Term(new_form, t.labels)

    return rep


def relabel_uadv(t):
    old = t.labels["uadv"]
    new = t.get("subject").split()[0]
    new_form = ufl.replace(t.form, {old: new})
    return advecting_velocity.remove(Term(new_form, t.labels))


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

    def get(self, key, default=None):
        return self.labels.get(key)

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

    def __rmul__(self, other):
        if type(other) is float:
            other = Constant(other)
        elif type(other) is not Constant:
            return NotImplemented
        return Term(other*self.form, self.labels)


class LabelledForm(object):
    __slots__ = ["terms"]

    def __init__(self, *terms):
        if len(terms) == 1 and isinstance(terms[0], LabelledForm):
            self.terms = terms[0].terms
        else:
            self.terms = list(terms)

    def __add__(self, other):
        if type(other) is Term:
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

    def __rmul__(self, other):
        if type(other) is float:
            other = Constant(other)
        elif type(other) is not Constant:
            return NotImplemented
        return self.label_map(all_terms, lambda t: Term(other*t.form, t.labels))

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
advecting_velocity = Label("uadv", validator=lambda value: type(value) == Function)
subject = Label("subject", validator=lambda value: type(value) in [TrialFunction, Function, ufl.tensors.ListTensor, ufl.indexed.Indexed])
linearisation = Label("linearisation", validator=lambda value: type(value) is LabelledForm)
index = Label("index", validator=lambda value: type(value) is int)
