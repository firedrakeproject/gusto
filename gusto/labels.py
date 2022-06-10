import ufl
from firedrake import Function, split, MixedElement, VectorElement
from gusto.fml.form_manipulation_labelling import Term, Label, LabelledForm


def replace_test_function(new_test):
    """
    :arg new_test: a :func:`TestFunction`

    Returns a function that takes in t, a :class:`Term`, and returns
    a new :class:`Term` with form containing the new_test and
    labels=t.labels
    """

    def repl(t):
        test = t.form.arguments()[0]
        new_form = ufl.replace(t.form, {test: new_test})
        return Term(new_form, t.labels)

    return repl


def replace_subject(new, idx=None):
    """
    Returns a function that takes a :class:`Term` and returns a new
    :class:`Term` with the subject of a term replaced by another variable.

    :arg new: the new variable to replace the subject
    :arg idx: (Optional) index of the subject in a mixed function space
    """
    def repl(t):
        """
        Function returned by replace_subject to return a new :class:`Term` with
        the subject replaced by the variable `new`. It is built around the ufl
        replace routine.

        Returns a new :class:`Term`.

        :arg t: the original :class:`Term`.
        """

        subj = t.get(subject)

        # Build a dictionary to pass to the ufl.replace routine
        # The dictionary matches variables in the old term with those in the new
        replace_dict = {}

        # Consider cases that subj is normal Function or MixedFunction
        # vs cases of new being Function vs MixedFunction vs tuple
        # Ideally catch all cases or fail gracefully
        if (isinstance(subj.ufl_element(), MixedElement)
            and not isinstance(subj.ufl_element(), VectorElement)):
            if type(new) == tuple:
                assert len(new) == len(subj.function_space())
                for k, v in zip(split(subj), new):
                    replace_dict[k] = v

            # Otherwise fail if new is not a function
            elif not isinstance(new, Function):
                raise ValueError(f'new must be a tuple or Function, not type {type(new)}')

            # Now handle MixedElements separately as these need indexing
            elif (isinstance(new.ufl_element(), MixedElement)
              and not isinstance(new.ufl_element(), VectorElement)):
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
                replace_dict[split(subj)[idx]] = new

        # subj is a normal Function
        else:
            if not isinstance(new, Function):
                raise ValueError(f'new must be a Function, not type {type(new)}')
            elif (isinstance(new.ufl_element(), MixedElement)
                  and not isinstance(new.ufl_element(), VectorElement)):
                replace_dict[subj] = split(new)[idx]
            else:
                replace_dict[subj] = new

        new_form = ufl.replace(t.form, replace_dict)

        return Term(new_form, t.labels)

    return repl


time_derivative = Label("time_derivative")
advection = Label("advection")
diffusion = Label("diffusion")
advecting_velocity = Label("advecting_velocity", validator=lambda value: type(value) == Function)
subject = Label("subject", validator=lambda value: type(value) == Function)
prognostic = Label("prognostic", validator=lambda value: type(value) == str)
linearisation = Label("linearisation", validator=lambda value: type(value) == LabelledForm)
name = Label("name", validator=lambda value: type(value) == str)
