
class Term(object):
    """
    Term class, containing a form and its labels.

    :arg form: the form for this term
    """

    def __init__(self, form):
        self.__setattr__("form", form)


class Label(object):
    """
    Class provinding labelling functionality for Gusto terms and equations

    :arg label: str giving the name of the label.
    :arg value: str providing the value of the label. Defaults to True.
    """

    def __init__(self, label, value=True):
        self.label = label
        self.__setattr__(label, value)

    def __call__(self, q):
        if not isinstance(q, Term):
            q = Term(q)
        q.__setattr__(self.label, getattr(self, self.label))
        return q
