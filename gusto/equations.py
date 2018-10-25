import ufl

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

    def has_label(self, *labels):
        if len(labels) == 1:
            return labels[0] in self.labels
        else:
            return tuple(self.has_label(l) for l in labels)

    def __add__(self, other):
        if other is None:
            return self
        elif isinstance(other, Term):
            return Equation(self, other)
        elif isinstance(other, Equation):
            return Equation(self, *other.terms)
        else:
            return NotImplemented


class Equation(object):
    __slots__ = ["terms"]

    def __init__(self, *terms):
        if len(terms) == 1 and isinstance (terms[0], Equation):
            self.terms = terms[0].terms
        else:
            self.terms = list(terms)

    def __add__(self, other):
        if type(other) not in {Term, Equation}:
            return NotImplemented
        elif other is None:
            return self
        return Equation(*self, *other)

    def __iter__(self):
        return iter(self.terms)

    def __len__(self):
        return len(self.terms)

    def __getitem__(self, key):
        return self.terms[key]

    def label_map(self, term_filter, map_function):
        """Return a new equation in which terms for which
        `term_filter` is `True` are transformed by
        `map_function`. Terms for which `term_filter` is false are
        included unaltered."""

        return Equation(functools.reduce(lambda x, y: x + y,
                                         (map_function(t) if term_filter(t) else t
                                          for t in self.terms)))
    @property
    def form(self):
        return functools.reduce(lambda x, y: x + y, (t.form for t in self.terms))


class Label(object):
    """
    Class provinding labelling functionality for Gusto terms and equations

    :arg label: str giving the name of the label.
    :arg value: str providing the value of the label. Defaults to True.
    """
    __slots__ = ["label", "value"]

    def __init__(self, label, value=True):
        self.label = label
        self.value = value

    def __call__(self, target):
        if isinstance(target, Equation):
            return Equation(*(self(t) for t in target.terms))
        elif isinstance(target, ufl.Form):
            return Equation(Term(target, {self.label: self.value}))
        elif isinstance(target, Term):
            new_labels = target.labels.copy()
            new_labels.update({self.label: self.value})
            return Term(target.form, new_labels)
        else:
            raise ValueError("Unable to label %s" % target)

    def remove(self, target):
        """Remove any :form:`Label` with this `label` from
        `target`. If called on an :class:`Equation`, act termwise."""
        if isinstance(target, Equation):
            return Equation(*(self.remove(t) for t in target.terms))
        elif isinstance(target, Term):
            try:
                d = target.labels.copy()
                d.pop(self.label)
                return Term(target.form, d)
            except KeyError:
                return target
        else:
            raise ValueError("Unable to unlabel %s" % target)
