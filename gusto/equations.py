from abc import ABCMeta, abstractproperty
from firedrake import (Function, TestFunction, inner, dx, div,
                       SpatialCoordinate, sqrt, FunctionSpace,
                       MixedFunctionSpace)
from gusto.form_manipulation_labelling import subject, time_derivative
from gusto.transport_equation import (IntegrateByParts, vector_invariant_form,
                                      continuity_form, advection_form)
from gusto.state import build_spaces


class PrognosticEquation(object, metaclass=ABCMeta):

    def __init__(self, state, field_name, function_space):
        self.state = state
        self.function_space = function_space
        state.fields(field_name, function_space)
        state.diagnostics.register(field_name)

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
        super().__init__(state, field_name, function_space)
        self.kwargs = kwargs

    def form(self):
        return advection_form(self.state, self.function_space, **self.kwargs)


class ContinuityEquation(PrognosticEquation):

    def __init__(self, state, field_name, function_space, **kwargs):
        super().__init__(state, field_name, function_space)
        self.kwargs = kwargs

    def form(self):
        return continuity_form(self.state, self.function_space, **self.kwargs)


def shallow_water_equations(state, family, degree):

    g = state.parameters.g
    Omega = state.parameters.Omega
    x = SpatialCoordinate(state.mesh)
    R = sqrt(inner(x, x))
    fexpr = 2*Omega*x[2]/R
    V = FunctionSpace(state.mesh, "CG", 1)
    f = Function(V).interpolate(fexpr)
    state.fields("coriolis", f)

    Vu, VD = build_spaces(state, family, degree)
    state.spaces.W = MixedFunctionSpace((Vu, VD))
    u = state.fields("u", Vu)
    D = state.fields("D", VD)
    state.diagnostics.register("u")
    state.diagnostics.register("D")

    
    u_adv = prognostic_variable(vector_invariant_form(state, Vu), "u")

    w = TestFunction(Vu)
    coriolis_term = prognostic_variable(subject(-f*inner(w, state.perp(u))*dx, u), "u")
    pressure_gradient_term = prognostic_variable(subject(g*div(w)*D*dx, D), "u")

    u_eqn = u_adv + coriolis_term + pressure_gradient_term

    D_eqn = prognostic_variable(continuity_form(state, VD), "D")

    return u_eqn + D_eqn
