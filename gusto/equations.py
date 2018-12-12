from abc import ABCMeta, abstractproperty
from firedrake import (Function, TestFunction, inner, dx, div,
                       SpatialCoordinate, sqrt, FunctionSpace,
                       MixedFunctionSpace, TestFunctions)
from gusto.form_manipulation_labelling import (subject, time_derivative,
                                               prognostic_variable,
                                               linearisation)
from gusto.transport_equation import (vector_invariant_form,
                                      continuity_form, advection_form)
from gusto.state import build_spaces


class PrognosticEquation(object, metaclass=ABCMeta):

    def __init__(self, state, function_space, *field_names):
        self.state = state
        self.function_space = function_space

        if len(field_names) == 1:
            state.fields(field_names[0], function_space)
        else:
            state.fields(field_names, function_space)

        state.diagnostics.register(*field_names)

    def mass_term(self):
        test = TestFunction(self.function_space)
        q = Function(self.function_space)
        return subject(time_derivative(inner(q, test)*dx), q)

    @abstractproperty
    def form(self):
        pass

    def __call__(self):
        return self.mass_term() + self.form()


class AdvectionEquation(PrognosticEquation):

    def __init__(self, state, field_name, function_space, **kwargs):
        super().__init__(state, function_space, field_name)
        self.kwargs = kwargs

    def form(self):
        return advection_form(self.state, self.function_space, **self.kwargs)


class ContinuityEquation(PrognosticEquation):

    def __init__(self, state, field_name, function_space, **kwargs):
        super().__init__(state, field_name, function_space)
        self.kwargs = kwargs

    def form(self):
        return continuity_form(self.state, self.function_space, **self.kwargs)


class ShallowWaterEquations(PrognosticEquation):

    def __init__(self, state, family, degree):
        self.Vu, self.VD = build_spaces(state, family, degree)
        state.spaces.W = MixedFunctionSpace((self.Vu, self.VD))

        self.fieldlist = ['u', 'D']
        super().__init__(state, state.spaces.W, *self.fieldlist)
        Omega = state.parameters.Omega
        x = SpatialCoordinate(state.mesh)
        R = sqrt(inner(x, x))
        fexpr = 2*Omega*x[2]/R
        V = FunctionSpace(state.mesh, "CG", 1)
        f = Function(V).interpolate(fexpr)
        state.fields("coriolis", f)

    def mass_term(self):
        test_u = TestFunction(self.Vu)
        u = self.state.fields("u")
        test_D = TestFunction(self.VD)
        D = self.state.fields("D")
        umass = prognostic_variable(subject(time_derivative(inner(u, test_u)*dx), u), "u")
        Dmass = prognostic_variable(subject(time_derivative(inner(D, test_D)*dx), D), "D")
        return umass + Dmass

    def form(self):
        state = self.state
        g = state.parameters.g
        H = state.parameters.H
        f = state.fields("coriolis")
        u_adv = prognostic_variable(vector_invariant_form(state, self.Vu), "u")

        w = TestFunction(self.Vu)
        u = self.state.fields("u")
        D = self.state.fields("D")
        wm, phim = TestFunctions(state.spaces.W)

        coriolis_term = linearisation(prognostic_variable(subject(-f*inner(w, state.perp(u))*dx, u), "u"), subject(-f*inner(wm, state.perp(u))*dx, u))

        pressure_gradient_term = linearisation(prognostic_variable(subject(g*div(w)*D*dx, D), "u"), subject(-g*div(wm)*D*dx, D))

        u_form = u_adv + coriolis_term + pressure_gradient_term

        D_form = linearisation(prognostic_variable(continuity_form(state, self.VD), "D"), subject(H*phim*div(u)*dx, u))

        return u_form + D_form
