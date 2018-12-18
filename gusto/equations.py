from abc import ABCMeta, abstractproperty
import functools
import operator
from firedrake import (Function, TestFunction, inner, dx, div,
                       SpatialCoordinate, sqrt, FunctionSpace,
                       MixedFunctionSpace, TestFunctions)
from gusto.form_manipulation_labelling import (subject, time_derivative,
                                               linearisation, linearise,
                                               all_terms, drop)
from gusto.diffusion import interior_penalty_diffusion_form
from gusto.transport_equation import (vector_invariant_form,
                                      continuity_form, advection_form,
                                      linear_advection_form)
from gusto.state import build_spaces


class PrognosticEquation(object, metaclass=ABCMeta):

    def __init__(self, state, function_space, *field_names):
        self.state = state
        self.function_space = function_space

        if len(field_names) == 1:
            state.fields(field_names[0], function_space)
        else:
            assert len(field_names) == len(function_space)
            state.fields(field_names, function_space)

        state.diagnostics.register(*field_names)

    def mass_term(self):

        if len(self.function_space) == 1:
            test = TestFunction(self.function_space)
            q = Function(self.function_space)
            return subject(time_derivative(inner(q, test)*dx), q)
        else:
            tests = TestFunctions(self.function_space)
            qs = Function(self.function_space)
            return functools.reduce(
                operator.add, (subject(time_derivative(inner(q, test)*dx), q)
                               for q, test in zip(qs.split(), tests)))

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


class DiffusionEquation(PrognosticEquation):

    def __init__(self, state, field_name, function_space, **kwargs):
        super().__init__(state, function_space, field_name)
        self.kwargs = kwargs

    def form(self):
        return interior_penalty_diffusion_form(self.state, self.function_space, **self.kwargs)


class ShallowWaterEquations(PrognosticEquation):

    def __init__(self, state, family, degree, linear=False):

        self.linear = linear
        Vu, VD = build_spaces(state, family, degree)
        state.spaces.W = MixedFunctionSpace((Vu, VD))
        self.function_space = state.spaces.W

        self.fieldlist = ['u', 'D']
        super().__init__(state, state.spaces.W, *self.fieldlist)
        Omega = state.parameters.Omega
        x = SpatialCoordinate(state.mesh)
        R = sqrt(inner(x, x))
        fexpr = 2*Omega*x[2]/R
        V = FunctionSpace(state.mesh, "CG", 1)
        f = Function(V).interpolate(fexpr)
        state.fields("coriolis", f)

    def form(self):
        state = self.state
        g = state.parameters.g
        H = state.parameters.H
        f = state.fields("coriolis")

        u = self.state.fields("u")
        D = self.state.fields("D")
        W = state.spaces.W
        w, phi = TestFunctions(W)

        u_adv = vector_invariant_form(state, W, 0)

        coriolis_term = linearisation(subject(-f*inner(w, state.perp(u))*dx, u), subject(-f*inner(w, state.perp(u))*dx, u))

        pressure_gradient_term = linearisation(subject(g*div(w)*D*dx, D), subject(g*div(w)*D*dx, D))

        u_form = u_adv + coriolis_term + pressure_gradient_term

        D_form = linearisation(continuity_form(state, W, 1), linear_advection_form(state, W, 1, H))

        if self.linear:
            return u_form.label_map(lambda t: t.has_label(linearisation), linearise(), drop) + D_form.label_map(all_terms, linearise())
        else:
            return u_form + D_form
