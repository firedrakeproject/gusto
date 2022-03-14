import ufl
from firedrake import Function
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
                for k, v in zip(subj.split(), new):
                    replace_dict[k] = v
            else:
                if idx is None:
                    for k, v in zip(subj.split(), new.split()):
                        replace_dict[k] = v
                else:
                    try:
                        replace_dict[subj.split()[idx]] = new.split()[idx]
                    except:
                        replace_dict[subj.split()[idx]] = new

        else:
            if subj.ufl_shape == new.ufl_shape:
                replace_dict[subj] = new
            else:
                replace_dict[subj] = new[idx]

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
