import ufl
import functools
import operator
from firedrake import Constant


identity = lambda t: t
drop = lambda t: None
all_terms = lambda t: True


class Term(object):
    """
    Term class, contains a form and its labels.

    :arg form: the form for this term
    :arg label_dict: dictionary of key-value pairs corresponding to current form labels.
    """
    __slots__ = ["form", "labels"]

    def __init__(self, form, label_dict=None):
        self.form = form
        self.labels = label_dict or {}

    def get(self, label, default=None):
        return self.labels.get(label.label)

    def has_label(self, *labels, return_tuple=False):
        if len(labels) == 1 and not return_tuple:
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

    def __sub__(self, other):
        other = other * Constant(-1.0)
        return self + other

    def __mul__(self, other):
        if type(other) in (float, int):
            other = Constant(other)
        elif type(other) not in [Constant, ufl.algebra.Product]:
            return NotImplemented
        return Term(other*self.form, self.labels)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if type(other) in (float, int, Constant, ufl.algebra.Product):
            other = Constant(1.0 / other)
            return self * other
        else:
            return NotImplemented


NullTerm = Term(None)


class LabelledForm(object):
    """
    The `LabelledForm` object holds a list of terms, which pair :class:`Form`
    objects with :class:`Label`s. The `label_map` routine allows the terms to be
    manipulated or selected based on particular filters.

    :arg *terms: a list of `Term` objects or a single `LabelledForm`.
    """
    __slots__ = ["terms"]

    def __init__(self, *terms):
        if len(terms) == 1 and isinstance(terms[0], LabelledForm):
            self.terms = terms[0].terms
        else:
            if any([type(term) is not Term for term in list(terms)]):
                raise TypeError('Can only pass terms or a LabelledForm to LabelledForm')
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

    __radd__ = __add__

    def __sub__(self, other):
        if type(other) is Term:
            return LabelledForm(*self, Constant(-1.)*other)
        elif type(other) is LabelledForm:
            return LabelledForm(*self, *[Constant(-1.)*t for t in other])
        elif type(other) is ufl.algebra.Product:
            return LabelledForm(*self, Term(Constant(-1.)*other))
        elif other is None:
            return self
        else:
            return NotImplemented

    def __mul__(self, other):
        if type(other) in (float, int):
            other = Constant(other)
        # UFL can cancel constants to a Zero type which needs treating separately
        elif type(other) is ufl.constantvalue.Zero:
            other = Constant(0.0)
        elif type(other) not in [Constant, ufl.algebra.Product]:
            return NotImplemented
        return self.label_map(all_terms, lambda t: Term(other*t.form, t.labels))

    def __truediv__(self, other):
        if type(other) in (float, int, Constant, ufl.algebra.Product):
            other = Constant(1.0 / other)
            return self * other
        else:
            return NotImplemented

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

        new_labelled_form = LabelledForm(
            functools.reduce(operator.add,
                             filter(lambda t: t is not None,
                                    (map_if_true(t) if term_filter(t) else
                                     map_if_false(t) for t in self.terms)),
                             # TODO: Not clear what the initialiser should be!
                             # No initialiser means label_map can't work if everything is false
                             # None is a problem as cannot be added to Term
                             # NullTerm works but will need dropping ...
                             NullTerm))

        # Drop the NullTerm
        new_labelled_form.terms = list(filter(lambda t: t is not NullTerm,
                                              new_labelled_form.terms))

        return new_labelled_form

    @property
    def form(self):
        # Throw an error if there is no form
        if len(self.terms) == 0:
            raise TypeError('The labelled form cannot return a form as it has no terms')
        else:
            return functools.reduce(operator.add, (t.form for t in self.terms))


class Label(object):
    """
    Class providing labelling functionality for Gusto forms and equations

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
        """
        Application of the `Label` to the `target` adds the label to the terms
        in the `target`. If `value` is provided, the label takes this value.
        """
        # if value is provided, check that we have a validator function
        # and validate the value, otherwise use default value
        if value is not None:
            assert self.validator
            assert self.validator(value)
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

    def update_value(self, target, new):
        """Update any :form:`Label` with this `label` in `target`, giving it
        the new value of `new`."""

        if isinstance(target, LabelledForm):
            return LabelledForm(*(self.update_value(t, new) for t in target.terms))
        elif isinstance(target, Term):
            try:
                d = target.labels.copy()
                d[self.label] = new
                return Term(target.form, d)
            except KeyError:
                return target
        else:
            raise ValueError("Unable to relabel %s" % target)
