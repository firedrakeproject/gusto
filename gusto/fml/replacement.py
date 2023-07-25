"""
Generic routines for replacing functions using FML.
"""

import ufl
from .form_manipulation_language import Term, subject
from firedrake import split, MixedElement

__all__ = ["replace_test_function", "replace_trial_function",
           "replace_subject"]


# ---------------------------------------------------------------------------- #
# A general routine for building the replacement dictionary
# ---------------------------------------------------------------------------- #
def _replace_dict(old, new, idx, replace_type):
    """
    Build a dictionary to pass to the ufl.replace routine
    The dictionary matches variables in the old term with those in the new

    Does not check types unless indexing is required (leave type-checking to ufl.replace)
    """

    replace_dict = {}

    if type(old.ufl_element()) is MixedElement:

        mixed_new = hasattr(new, "ufl_element") and type(new.ufl_element()) is MixedElement
        indexable_new = type(new) is tuple or mixed_new

        if indexable_new:
            split_new = new if type(new) is tuple else split(new)

            if len(split_new) != len(old.function_space()):
                raise ValueError(f"new {replace_type} of type {new} must be same length"
                                 + f"as replaced mixed {replace_type} of type {old}")

            if idx is None:
                for k, v in zip(split(old), split_new):
                    replace_dict[k] = v
            else:
                replace_dict[split(old)[idx]] = split_new[idx]

        else:  # new is not indexable
            if idx is None:
                raise ValueError(f"idx must be specified to replace_{replace_type} when"
                                 + f" replaced {replace_type} of type {old} is mixed and"
                                 + f" new {replace_type} of type {new} is a single component")

            replace_dict[split(old)[idx]] = new

    else:  # old is not mixed

        mixed_new = hasattr(new, "ufl_element") and type(new.ufl_element()) is MixedElement
        indexable_new = type(new) is tuple or mixed_new

        if indexable_new:
            split_new = new if type(new) is tuple else split(new)

            if idx is None:
                raise ValueError(f"idx must be specified to replace_{replace_type} when"
                                 + f" replaced {replace_type} of type {old} is not mixed"
                                 + f" and new {replace_type} of type {new} is indexable")

            replace_dict[old] = split_new[idx]

        else:
            replace_dict[old] = new

    return replace_dict


# ---------------------------------------------------------------------------- #
# Replacement routines
# ---------------------------------------------------------------------------- #
def replace_test_function(new_test, idx=None):
    """
    A routine to replace the test function in a term with a new test function.

    Args:
        new_test (:class:`TestFunction`): the new test function.

    Returns:
        a function that takes in t, a :class:`Term`, and returns a new
        :class:`Term` with form containing the new_test and labels=t.labels
    """

    def repl(t):
        """
        Replaces the test function in a term with a new expression. This is
        built around the ufl replace routine.

        Args:
            t (:class:`Term`): the original term.

        Returns:
            :class:`Term`: the new term.
        """
        old_test = t.form.arguments()[0]
        replace_dict = _replace_dict(old_test, new_test, idx, 'test')

        try:
            new_form = ufl.replace(t.form, replace_dict)
        except Exception as err:
            error_message = f"{type(err)} raised by ufl.replace when trying to" \
                            + f" replace_test_function with {new_test}"
            raise type(err)(error_message) from err

        return Term(new_form, t.labels)

    return repl


def replace_trial_function(new_trial, idx=None):
    """
    A routine to replace the trial function in a term with a new expression.

    Args:
        new (:class:`TrialFunction` or :class:`Function`): the new function.

    Returns:
        a function that takes in t, a :class:`Term`, and returns a new
        :class:`Term` with form containing the new_test and labels=t.labels
    """

    def repl(t):
        """
        Replaces the trial function in a term with a new expression. This is
        built around the ufl replace routine.

        Args:
            t (:class:`Term`): the original term.

        Raises:
            TypeError: if the form is linear.

        Returns:
            :class:`Term`: the new term.
        """
        if len(t.form.arguments()) != 2:
            raise TypeError('Trying to replace trial function of a form that is not linear')
        old_trial = t.form.arguments()[1]
        replace_dict = _replace_dict(old_trial, new_trial, idx, 'trial')

        try:
            new_form = ufl.replace(t.form, replace_dict)
        except Exception as err:
            error_message = f"{type(err)} raised by ufl.replace when trying to" \
                            + f" replace_trial_function with {new_trial}"
            raise type(err)(error_message) from err

        return Term(new_form, t.labels)

    return repl


def replace_subject(new_subj, idx=None):
    """
    A routine to replace the subject in a term with a new variable.

    Args:
        new (:class:`ufl.Expr`): the new expression to replace the subject.
        idx (int, optional): index of the subject in the equation's
            :class:`MixedFunctionSpace`. Defaults to None.
    """
    def repl(t):
        """
        Replaces the subject in a term with a new expression. This is built
        around the ufl replace routine.

        Args:
            t (:class:`Term`): the original term.

        Raises:
            ValueError: when the new expression and subject are not of
                compatible sizes (e.g. a mixed function vs a non-mixed function)

        Returns:
            :class:`Term`: the new term.
        """

        old_subj = t.get(subject)
        replace_dict = _replace_dict(old_subj, new_subj, idx, 'subject')

        try:
            new_form = ufl.replace(t.form, replace_dict)
        except Exception as err:
            error_message = f"{type(err)} raised by ufl.replace when trying to" \
                            + f" replace_subject with {new_subj}"
            raise type(err)(error_message) from err

        # When a term has the perp label, this indicates that replace
        # cannot see that the perped object should also be
        # replaced. In this case we also pass the perped object to
        # replace.
        if t.has_label(perp):
            perp_op = t.get(perp)
            perp_old = perp_op(t.get(subject))
            perp_new = perp_op(new_subj)
            try:
                new_form = ufl.replace(t.form, {perp_old: perp_new})

            except Exception as err:
                error_message = f"{type(err)} raised by ufl.replace when trying to" \
                    + f" replace_subject with {new_subj}"
                raise type(err)(error_message) from err

        return Term(new_form, t.labels)

    return repl
