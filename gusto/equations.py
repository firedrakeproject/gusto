from abc import ABCMeta
from firedrake import (TestFunction, Function, inner, dx, div,
                       FunctionSpace, MixedFunctionSpace, TestFunctions)
from gusto.form_manipulation_labelling import (subject, time_derivative,
                                               advection, prognostic)
from gusto.transport_equation import (advection_form, continuity_form,
                                      vector_invariant_form)
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

        if len(function_space) > 1:
            assert hasattr(self, "field_names")
            state.fields(field_name, function_space,
                         subfield_names=self.field_names)
            for name in self.field_names:
                state.diagnostics.register(name)
        else:
            state.fields(field_name, function_space)
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
                 ufamily=None, udegree=None, **kwargs):
        super().__init__(state, function_space, field_name)

        if not hasattr(state.fields, "u"):
            V = state.spaces("HDiv", ufamily, udegree)
            state.fields("u", V)
        test = TestFunction(function_space)
        q = Function(function_space)
        mass_form = time_derivative(inner(q, test)*dx)

        self.residual = subject(
            mass_form + advection_form(state, test, q, **kwargs), q
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
                 ufamily=None, udegree=None, **kwargs):
        super().__init__(state, function_space, field_name)

        if not hasattr(state.fields, "u"):
            V = state.spaces("HDiv", ufamily, udegree)
            state.fields("u", V)
        test = TestFunction(function_space)
        q = Function(function_space)
        mass_form = time_derivative(inner(q, test)*dx)

        self.residual = subject(
            mass_form + continuity_form(state, test, q, **kwargs), q
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
        mass_form = time_derivative(inner(q, test)*dx)

        self.residual = subject(
            mass_form
            + interior_penalty_diffusion_form(state, test, q, **kwargs), q
        )


class AdvectionDiffusionEquation(PrognosticEquation):
    """
    Class defining the advection-diffusion equation.

    :arg state: :class:`.State` object
    :arg field_name: name of the prognostic field
    :arg function_space: :class:`.FunctionSpace` object, the function
    :kwargs: any kwargs to be passed on to the advection_form or diffusion_form
    """
    def __init__(self, state, function_space, field_name,
                 ufamily=None, udegree=None, **kwargs):
        super().__init__(state, function_space, field_name)
        dkwargs = {}
        for k in ["kappa", "mu"]:
            assert k in kwargs.keys(), "diffusion form requires %s kwarg " % k
            dkwargs[k] = kwargs.pop(k)
        akwargs = kwargs

        if not hasattr(state.fields, "u"):
            V = state.spaces("HDiv", ufamily, udegree)
            state.fields("u", V)
        test = TestFunction(function_space)
        q = Function(function_space)
        mass_form = time_derivative(inner(q, test)*dx)

        self.residual = subject(
            mass_form
            + advection_form(state, test, q, **akwargs)
            + interior_penalty_diffusion_form(state, test, q, **dkwargs), q
        )


class ShallowWaterEquations(PrognosticEquation):

    field_names = ["u", "D"]

    def __init__(self, state, family, degree, fexpr=None):

        spaces = state.spaces.build_compatible_spaces(family, degree)
        W = MixedFunctionSpace(spaces)

        field_name = "_".join(self.field_names)
        super().__init__(state, W, field_name)

        V = FunctionSpace(state.mesh, "CG", 1)
        f = state.fields("coriolis", space=V)
        f.interpolate(fexpr)

        g = state.parameters.g

        w, phi = TestFunctions(W)
        X = Function(W)
        u, D = X.split()

        u_mass = prognostic(inner(u, w)*dx, "u")
        D_mass = prognostic(inner(D, phi)*dx, "D")
        mass_form = time_derivative(u_mass + D_mass)

        u_adv = prognostic(vector_invariant_form(state, w, u), "u")
        D_adv = prognostic(continuity_form(state, phi, D), "D")
        advection_form = advection(u_adv + D_adv)

        coriolis_form = prognostic(f*inner(state.perp(u), w)*dx, "u")

        pressure_gradient_form = prognostic(-g*div(w)*D*dx, "u")

        self.residual = subject(mass_form + advection_form
                                + coriolis_form + pressure_gradient_form, X)
