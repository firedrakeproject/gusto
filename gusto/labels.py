"""Common labels and routines for manipulating forms using labels."""

import ufl
from firedrake import Function, split, MixedElement
from gusto.configuration import IntegrateByParts, TransportEquationType
from gusto.fml.form_manipulation_labelling import Term, Label, LabelledForm
from types import MethodType


def _replace_dict(old, new, idx, replace_type):
    """
    Build a dictionary to pass to the ufl.replace routine
    The dictionary matches variables in the old term with those in the new

    Consider cases that old is normal Function or MixedFunction
    vs cases of new being Function vs MixedFunction vs tuple
    Ideally catch all cases or fail gracefully
    """

    replace_dict = {}

    acceptable_types = (type(old), ufl.algebra.Sum, ufl.indexed.Indexed)
    if replace_type == 'trial':
        acceptable_types = (*acceptable_types, Function)

    type_error_message = f'new must be a {tuple} or '+' or '.join((f"{t}" for t in acceptable_types))+f', not {type(new)}'

    if type(old.ufl_element()) is MixedElement:
        if type(new) == tuple:
            assert len(new) == len(old.function_space())
            for k, v in zip(split(old), new):
                replace_dict[k] = v

        # Otherwise fail if new is not a function
        elif not isinstance(new, acceptable_types):
            raise TypeError(type_error_message)

        elif type(new) == ufl.algebra.Sum:
            replace_dict[old] = new

        elif isinstance(new, ufl.indexed.Indexed):
            if idx is None:
                raise ValueError('idx must be specified to replace_{replace_type}'
                                 + ' when {replace_type} is Mixed and new is a single component')
            replace_dict[split(old)[idx]] = new

        # Now handle MixedElements separately as these need indexing
        elif type(new.ufl_element()) is MixedElement:
            assert len(new.function_space()) == len(old.function_space())
            # If idx specified, replace only that component

            if idx is not None:
                replace_dict[split(old)[idx]] = split(new)[idx]

            # Otherwise replace all components
            else:
                for k, v in zip(split(old), split(new)):
                    replace_dict[k] = v

        # Otherwise 'new' is a normal Function
        else:
            if idx is None:
                raise ValueError('idx must be specified to replace_{replace_type}'
                                 + ' when {replace_type} is Mixed and new is a single component')
            replace_dict[split(old)[idx]] = new

    # old is a normal Function
    else:
        if type(new) is tuple:
            if idx is None:
                raise ValueError('idx must be specified to replace_{replace_type}'
                                 + ' when new is a tuple and {replace_type} is not Mixed')
            replace_dict[old] = new[idx]

        elif not isinstance(new, acceptable_types):
            raise TypeError(type_error_message)

        elif type(new) == ufl.algebra.Sum:
            replace_dict[old] = new

        elif isinstance(new, ufl.indexed.Indexed):
            replace_dict[old] = new

        elif type(new.ufl_element()) == MixedElement:
            if idx is None:
                raise ValueError('idx must be specified to replace_{replace_type}'
                                 + ' when new is a tuple and {replace_type} is not Mixed')
            replace_dict[old] = split(new)[idx]

        else:
            replace_dict[old] = new

    return replace_dict


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
        new_form = ufl.replace(t.form, replace_dict)
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
        new_form = ufl.replace(t.form, replace_dict)
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
        new_form = ufl.replace(t.form, replace_dict)
        return Term(new_form, t.labels)

    return repl


# ---------------------------------------------------------------------------- #
# Common Labels
# ---------------------------------------------------------------------------- #

time_derivative = Label("time_derivative")
transport = Label("transport", validator=lambda value: type(value) == TransportEquationType)
diffusion = Label("diffusion")
physics = Label("physics", validator=lambda value: type(value) == MethodType)
transporting_velocity = Label("transporting_velocity", validator=lambda value: type(value) == Function)
subject = Label("subject", validator=lambda value: type(value) == Function)
prognostic = Label("prognostic", validator=lambda value: type(value) == str)
pressure_gradient = Label("pressure_gradient")
coriolis = Label("coriolis")
linearisation = Label("linearisation", validator=lambda value: type(value) in [LabelledForm, Term])
name = Label("name", validator=lambda value: type(value) == str)
ibp_label = Label("ibp", validator=lambda value: type(value) == IntegrateByParts)
hydrostatic = Label("hydrostatic", validator=lambda value: type(value) in [LabelledForm, Term])
