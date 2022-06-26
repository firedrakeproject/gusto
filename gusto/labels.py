import ufl
from firedrake import Function, split, VectorElement
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


def replace_subject(new, idx=None):

    def repl(t):
        subj = t.get(subject)

        replace_dict = {}
        if len(subj.function_space()) > 1:
            if type(new) == tuple:
                assert len(new) == len(subj.function_space())
                for k, v in zip(split(subj), new):
                    replace_dict[k] = v
            else:
                if idx is None:
                    for k, v in zip(split(subj), split(new)):
                        replace_dict[k] = v
                # TODO: Could we do something better here?
                elif isinstance(new.ufl_element(), VectorElement):
                    replace_dict[split(subj)[idx]] = new
                else:
                    try:
                        # This needs to handle with MixedFunctionSpace and
                        # VectorFunctionSpace differently
                        replace_dict[split(subj)[idx]] = split(new)[idx]
                    except IndexError:
                        replace_dict[split(subj)[idx]] = new

        else:
            if len(new.function_space()) > 1:
                replace_dict[subj] = new[idx]
            else:
                replace_dict[subj] = new

        new_form = ufl.replace(t.form, replace_dict)
        return Term(new_form, t.labels)

    return repl


time_derivative = Label("time_derivative")
transport = Label("transport", validator=lambda value: type(value) == TransportEquationType)
diffusion = Label("diffusion")
transporting_velocity = Label("transporting_velocity", validator=lambda value: type(value) == Function)
subject = Label("subject", validator=lambda value: type(value) == Function)
prognostic = Label("prognostic", validator=lambda value: type(value) == str)
pressure_gradient = Label("pressure_gradient")
linearisation = Label("linearisation", validator=lambda value: type(value) in [LabelledForm, Term])
name = Label("name", validator=lambda value: type(value) == str)
ibp_label = Label("ibp", validator=lambda value: type(value) == IntegrateByParts)
