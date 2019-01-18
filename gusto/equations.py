from abc import ABCMeta, abstractproperty
import functools
import operator
from firedrake import (Function, TestFunction, inner, dx, div,
                       SpatialCoordinate, sqrt, FunctionSpace,
                       MixedFunctionSpace, TestFunctions)
from gusto.form_manipulation_labelling import (all_terms,
                                               subject, time_derivative,
                                               linearisation, linearise,
                                               drop, index, advection,
                                               relabel_uadv)
from gusto.diffusion import interior_penalty_diffusion_form
from gusto.transport_equation import (vector_invariant_form,
                                      continuity_form, advection_form,
                                      linear_advection_form,
                                      advection_equation_circulation_form,
                                      advection_vector_manifold_form,
                                      kinetic_energy_form)
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
                operator.add,
                (index(subject(
                    time_derivative(inner(q, test)*dx), qs), tests.index(test))
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

    solver_parameters = {
        'ksp_type': 'preonly',
        'mat_type': 'matfree',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.HybridizationPC',
        'hybridization': {'ksp_type': 'cg',
                          'pc_type': 'gamg',
                          'ksp_rtol': 1e-8,
                          'mg_levels': {'ksp_type': 'chebyshev',
                                        'ksp_max_it': 2,
                                        'pc_type': 'bjacobi',
                                        'sub_pc_type': 'ilu'}}
    }

    def __init__(self, state, family, degree, **kwargs):

        self.u_advection_option = kwargs.pop("u_advection_option", "vector_invariant_form")
        if kwargs:
            raise ValueError("unexpected kwargs: %s" % list(kwargs.keys()))

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

        W = state.spaces.W
        w, phi = TestFunctions(W)
        X = Function(W)
        u, D = X.split()

        if self.u_advection_option == "circulation_form":
            ke_form = kinetic_energy_form(state, W, 0)
            ke_form = advection.remove(ke_form)
            ke_form = ke_form.label_map(all_terms, relabel_uadv)
            u_adv = advection_equation_circulation_form(state, W, 0) + ke_form
        elif self.u_advection_option == "vector_advection":
            u_adv = advection_vector_manifold_form(state, W, 0)
        elif self.u_advection_option == "vector_invariant_form":
            u_adv = vector_invariant_form(state, W, 0)
        else:
            raise ValueError("Invalid u_advection_option: %s" % self.u_advection_option)

        coriolis_term = linearisation(subject(f*inner(w, state.perp(u))*dx, X))

        pressure_gradient_term = linearisation(subject(-g*div(w)*D*dx, X))

        u_form = u_adv + coriolis_term + pressure_gradient_term

        D_form = linearisation(continuity_form(state, W, 1), linear_advection_form(state, W, 1, H))

        return index(u_form, 0) + index(D_form, 1)


class LinearShallowWaterEquations(ShallowWaterEquations):

    def form(self):

        return super().form().label_map(lambda t: t.has_label(linearisation), linearise, drop)
