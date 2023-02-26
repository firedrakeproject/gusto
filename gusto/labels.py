"""Common labels and routines for manipulating forms using labels."""

import ufl
import firedrake
from firedrake import Function, split, MixedElement
from gusto.configuration import IntegrateByParts, TransportEquationType
from gusto.fml.form_manipulation_labelling import Term, Label, LabelledForm
from types import MethodType, LambdaType


def replace_test_function(new_test):
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
        test = t.form.arguments()[0]
        new_form = ufl.replace(t.form, {test: new_test})
        return Term(new_form, t.labels)

    return repl


def replace_trial_function(new):
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
        trial = t.form.arguments()[1]
        new_form = ufl.replace(t.form, {trial: new})
        return Term(new_form, t.labels)

    return repl


def replace_subject(new, idx=None):
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

        subj = t.get(subject)

        # Build a dictionary to pass to the ufl.replace routine
        # The dictionary matches variables in the old term with those in the new
        replace_dict = {}

        # Consider cases that subj is normal Function or MixedFunction
        # vs cases of new being Function vs MixedFunction vs tuple
        # Ideally catch all cases or fail gracefully
        if type(subj.ufl_element()) is MixedElement:
            if type(new) == tuple:
                assert len(new) == len(subj.function_space())
                for k, v in zip(split(subj), new):
                    replace_dict[k] = v

            elif type(new) == ufl.algebra.Sum:
                replace_dict[subj] = new

            elif isinstance(new, ufl.indexed.Indexed):
                if idx is None:
                    raise ValueError('idx must be specified to replace_subject'
                                     + ' when subject is Mixed and new is a single component')
                replace_dict[split(subj)[idx]] = new

            # Otherwise fail if new is not a function
            elif isinstance(new, Function) :
                raise ValueError(f'new must be a tuple or Function, not type {type(new)}')

            # Now handle MixedElements separately as these need indexing
            elif type(new.ufl_element()) is MixedElement:
                assert len(new.function_space()) == len(subj.function_space())
                # If idx specified, replace only that component
                if idx is not None:
                    replace_dict[split(subj)[idx]] = split(new)[idx]
                # Otherwise replace all components
                else:
                    for k, v in zip(split(subj), split(new)):
                        replace_dict[k] = v

            # Otherwise 'new' is a normal Function
            else:
                if idx is None:
                    raise ValueError('idx must be specified to replace_subject'
                                     + ' when subject is Mixed and new is a single component')
                replace_dict[split(subj)[idx]] = new

        # subj is a normal Function
        else:
            if type(new) is tuple:
                if idx is None:
                    raise ValueError('idx must be specified to replace_subject'
                                     + ' when new is a tuple')
                replace_dict[subj] = new[idx]
            elif isinstance(new, ufl.indexed.Indexed):
                replace_dict[subj] = new
            elif not isinstance(new, Function):
                raise ValueError(f'new must be a Function, not type {type(new)}')
            elif type(new.ufl_element()) == MixedElement:
                if idx is None:
                    raise ValueError('idx must be specified to replace_subject'
                                     + ' when new is a tuple')
                replace_dict[subj] = split(new)[idx]
            else:
                replace_dict[subj] = new

        new_form = ufl.replace(t.form, replace_dict)

        # this is necessary to defer applying the perp until after the
        # subject is replaced because otherwise replace cannot find
        # the subject
        if t.has_label(perp):
            perp_function = t.get(perp)
            new_form = ufl.replace(new_form, {split(new)[0]: perp_function(split(new)[0])})

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
perp = Label("perp", validator=lambda value: isinstance(value, LambdaType))
