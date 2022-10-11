import ufl
from firedrake import Function, split, MixedElement
from gusto.configuration import IntegrateByParts, TransportEquationType
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


def replace_trial_function(new):
    """
    :arg new: a :func:`Function` or `TrialFunction`

    Returns a function that takes in t, a :class:`Term`, and returns
    a new :class:`Term` containing a form with the trial function replaced
    labels=t.labels
    """

    def repl(t):
        if len(t.form.arguments()) != 2:
            raise TypeError('Trying to replace trial function of a form that is not linear')
        trial = t.form.arguments()[1]
        new_form = ufl.replace(t.form, {trial: new})
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
        if type(subj.ufl_element()) is MixedElement:
            if type(new) == tuple:
                assert len(new) == len(subj.function_space())
                for k, v in zip(split(subj), new):
                    replace_dict[k] = v

            elif type(new) == ufl.algebra.Sum:
                replace_dict[subj] = new

            # Otherwise fail if new is not a function
            elif not isinstance(new, Function):
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
                replace_dict[split(subj)[idx]] = new

        # subj is a normal Function
        else:
            if type(new) is tuple:
                if idx is None:
                    raise ValueError('idx must be specified to replace_subject'
                                     + ' when new is a tuple')
                replace_dict[subj] = new[idx]
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

        return Term(new_form, t.labels)

    return repl


time_derivative = Label("time_derivative")
transport = Label("transport", validator=lambda value: type(value) == TransportEquationType)
diffusion = Label("diffusion")
physics = Label("physics")
transporting_velocity = Label("transporting_velocity", validator=lambda value: type(value) == Function)
subject = Label("subject", validator=lambda value: type(value) == Function)
prognostic = Label("prognostic", validator=lambda value: type(value) == str)
pressure_gradient = Label("pressure_gradient")
coriolis = Label("coriolis")
linearisation = Label("linearisation", validator=lambda value: type(value) in [LabelledForm, Term])
name = Label("name", validator=lambda value: type(value) == str)
ibp_label = Label("ibp", validator=lambda value: type(value) == IntegrateByParts)
hydrostatic = Label("hydrostatic", validator=lambda value: type(value) in [LabelledForm, Term])
