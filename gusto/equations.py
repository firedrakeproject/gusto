from abc import ABCMeta
from firedrake import TestFunction, Function, inner, dx
from gusto.form_manipulation_labelling import subject, time_derivative
from gusto.transport_equation import advection_form, continuity_form
from gusto.diffusion import interior_penalty_diffusion_form


class PrognosticEquation(object, metaclass=ABCMeta):
    """
    Base class for prognostic equations

    :arg state: :class:`.State` object
    :arg function space: :class:`.FunctionSpace` object, the function
         space that the equation is defined on
    :arg field_name: name of the prognostic field

    The class sets up the field in state and registers it with the
    diagnostics class.
    """
    def __init__(self, state, function_space, field_name):

        self.state = state
        self.function_space = function_space
        self.field_name = field_name

        # default is to dump the field unless user has specified
        # otherwise when setting up the output parameters
        dump = state.output.dumplist or True
        state.fields(field_name, space=function_space, dump=dump, pickup=True)

        state.diagnostics.register(field_name)


class AdvectionEquation(PrognosticEquation):
    """
    Class defining the advection equation.

    :arg state: :class:`.State` object
    :arg function_space: :class:`.FunctionSpace` object
    :arg field_name: name of the prognostic field
    :kwargs: any kwargs to be passed on to the advection_form
    """
    def __init__(self, state, function_space, field_name,
                 **kwargs):
        super().__init__(state, function_space, field_name)

        test = TestFunction(function_space)
        q = Function(function_space)
        mass_form = subject(time_derivative(inner(q, test)*dx), q)

        self.residual = (
            mass_form + advection_form(state, function_space, **kwargs)
        )


class ContinuityEquation(PrognosticEquation):
    """
    Class defining the continuity equation.

    :arg state: :class:`.State` object
    :arg function_space: :class:`.FunctionSpace` object
    :arg field_name: name of the prognostic field
    :kwargs: any kwargs to be passed on to the continuity_form
    """
    def __init__(self, state, function_space, field_name,
                 **kwargs):
        super().__init__(state, function_space, field_name)

        test = TestFunction(function_space)
        q = Function(function_space)
        mass_form = subject(time_derivative(inner(q, test)*dx), q)

        self.residual = (
            mass_form + continuity_form(state, function_space, **kwargs)
        )


class DiffusionEquation(PrognosticEquation):
    """
    Class defining the advection equation.

    :arg state: :class:`.State` object
    :arg function_space: :class:`.FunctionSpace` object
    :arg field_name: name of the prognostic field
    :kwargs: any kwargs to be passed on to the advection_form
    """
    def __init__(self, state, function_space, field_name,
                 **kwargs):
        super().__init__(state, function_space, field_name)

        test = TestFunction(function_space)
        q = Function(function_space)
        mass_form = subject(time_derivative(inner(q, test)*dx), q)

        self.residual = (
            mass_form
            + interior_penalty_diffusion_form(state, function_space, **kwargs)
        )


class AdvectionDiffusionEquation(PrognosticEquation):
    """
    Class defining the advection-diffusion equation.

    :arg state: :class:`.State` object
    :arg field_name: name of the prognostic field
    :arg function_space: :class:`.FunctionSpace` object, the function
    :kwargs: any kwargs to be passed on to the advection_form or diffusion_form
    """
    def __init__(self, state, function_space, field_name, **kwargs):
        super().__init__(state, function_space, field_name)
        dkwargs = {}
        for k in ["kappa", "mu"]:
            assert k in kwargs.keys(), "diffusion form requires %s kwarg " % k
            dkwargs[k] = kwargs.pop(k)
        akwargs = kwargs

        test = TestFunction(function_space)
        q = Function(function_space)
        mass_form = subject(time_derivative(inner(q, test)*dx), q)

        self.residual = (
            mass_form
            + advection_form(state, function_space, **akwargs)
            + interior_penalty_diffusion_form(state, function_space, **dkwargs)
        )
