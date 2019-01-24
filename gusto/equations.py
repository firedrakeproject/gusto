from abc import ABCMeta, abstractproperty
import functools
import operator
from firedrake import (Function, TestFunction, inner, dx, div, action,
                       SpatialCoordinate, sqrt, FunctionSpace,
                       MixedFunctionSpace, TestFunctions, TrialFunctions)
from gusto.configuration import logger
from gusto.form_manipulation_labelling import (all_terms,
                                               subject, time_derivative,
                                               linearisation,
                                               drop, index, advection,
                                               relabel_uadv, replace_labelled, Term)
from gusto.diffusion import interior_penalty_diffusion_form
from gusto.transport_equation import (vector_invariant_form,
                                      continuity_form, advection_form,
                                      linear_continuity_form,
                                      advection_equation_circulation_form,
                                      advection_vector_manifold_form,
                                      kinetic_energy_form)
from gusto.state import build_spaces


class PrognosticEquation(object, metaclass=ABCMeta):
    """
    Base class for prognostic equations

    :arg state: :class:`.State` object
    :arg function space: :class:`.FunctionSpace` object, the function
         space that the equation is defined on
    :arg field_names: name(s) of the prognostic field(s)

    The class sets up the fields in state and registers them with the
    diagnostics class. It defines a mass term, labelled with the
    time_derivative label. All remaining forms must be defined in the
    child class form method. Calling this class returns the form
    mass_term + form
    """
    def __init__(self, state, function_space, *field_names):
        self.state = state
        self.function_space = function_space

        state.fields(*field_names, space=function_space)

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
    """
    Class defining the advection equation.

    :arg state: :class:`.State` object
    :arg field_name: name of the prognostic field
    :arg function_space: :class:`.FunctionSpace` object, the function
    :arg advecting_velocity: (optional) a :class:`Function` specifying the
    prescribed advecting velocity
    :kwargs: any kwargs to be passed on to the advection_form
    """
    def __init__(self, state, field_name, function_space,
                 advecting_velocity=None,
                 **kwargs):
        super().__init__(state, function_space, field_name)
        self.uadv = advecting_velocity
        self.kwargs = kwargs

    def form(self):
        return advection_form(self.state, self.function_space, uadv=self.uadv,
                              **self.kwargs)


class ContinuityEquation(PrognosticEquation):
    """
    Class defining the continuity equation.

    :arg state: :class:`.State` object
    :arg field_name: name of the prognostic field
    :arg function_space: :class:`.FunctionSpace` object, the function
    :arg advecting_velocity: (optional) a :class:`Function` specifying the
    prescribed advecting velocity
    :kwargs: any kwargs to be passed on to the continuity_form
    """

    def __init__(self, state, field_name, function_space,
                 advecting_velocity=None,
                 **kwargs):
        super().__init__(state, field_name, function_space)
        self.uadv = advecting_velocity
        self.kwargs = kwargs

    def form(self):
        return continuity_form(self.state, self.function_space, uadv=self.uadv,
                               **self.kwargs)


class DiffusionEquation(PrognosticEquation):
    """
    Class defining the diffusion equation.

    :arg state: :class:`.State` object
    :arg field_name: name of the prognostic field
    :arg function_space: :class:`.FunctionSpace` object, the function
    :arg advecting_velocity: (optional) a :class:`Function` specifying the
    prescribed advecting velocity
    :kwargs: any kwargs to be passed on to the diffuson_form
    """

    def __init__(self, state, field_name, function_space, **kwargs):
        super().__init__(state, function_space, field_name)
        self.kwargs = kwargs

    def form(self):
        return interior_penalty_diffusion_form(self.state, self.function_space, **self.kwargs)


class ShallowWaterEquations(PrognosticEquation):
    """
    Class defining the shallow water equations.

    :arg state: :class:`.State` object
    :arg family: str, specifies the velocity space family to use
    :arg degree: int, specifies the degree for the depth space
    :kwargs: expressions for additional fields and discretisation options
    to be passed to the form
    """
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

        fexpr = kwargs.pop("fexpr", "default")
        bexpr = kwargs.pop("bexpr", None)
        self.u_advection_option = kwargs.pop("u_advection_option", "vector_invariant_form")
        if kwargs:
            raise ValueError("unexpected kwargs: %s" % list(kwargs.keys()))

        Vu, VD = build_spaces(state, family, degree)
        self.function_space = MixedFunctionSpace((Vu, VD))

        self.fieldlist = ['u', 'D']

        super().__init__(state, self.function_space, *self.fieldlist)

        if fexpr:
            self.setup_coriolis(state, fexpr)

        if bexpr:
            self.setup_topography(state, bexpr)

    def setup_coriolis(self, state, fexpr):

        if fexpr == "default":
            Omega = state.parameters.Omega
            x = SpatialCoordinate(state.mesh)
            R = sqrt(inner(x, x))
            fexpr = 2*Omega*x[2]/R

        V = FunctionSpace(state.mesh, "CG", 1)
        f = state.fields("coriolis", space=V)
        f.interpolate(fexpr)

    def setup_topography(self, state, bexpr):

        b = state.fields("topography", space=state.fields("D").function_space())
        b.interpolate(bexpr)

    def form(self):
        state = self.state
        g = state.parameters.g
        H = state.parameters.H

        W = self.function_space
        w, phi = TestFunctions(W)
        trials = TrialFunctions(W)
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

        pressure_gradient_term = subject(-g*div(w)*D*dx, X)
        linear_pg_term = pressure_gradient_term.label_map(
            all_terms, replace_labelled(trials, "subject"))

        u_form = u_adv + linearisation(pressure_gradient_term, linear_pg_term)

        for field_name in ["coriolis", "topography"]:
            try:
                field = state.fields(field_name)
                add_term = True
            except AttributeError:
                logger.info("field %s not present" % field_name)
                add_term = False

            if add_term:
                if field_name == "coriolis":
                    coriolis_term = subject(field*inner(w, state.perp(u))*dx, X)
                    linear_coriolis_term = coriolis_term.label_map(
                        all_terms, replace_labelled(trials, "subject"))
                    u_form += linearisation(coriolis_term, linear_coriolis_term)

                elif field_name == "topography":
                    u_form += -g*div(w)*field*dx

        Dadv = continuity_form(state, W, 1)
        Dadv_linear = linear_continuity_form(state, W, 1, qbar=H).label_map(
            all_terms, replace_labelled(trials, "subject", "uadv"))
        D_form = linearisation(Dadv, Dadv_linear)

        return index(u_form, 0) + index(D_form, 1)


class LinearShallowWaterEquations(ShallowWaterEquations):

    def form(self):
        sw_form = super().form()

        linear_form = sw_form.label_map(
            lambda t: t.has_label(linearisation),
            lambda t: Term(action(t.get("linearisation").form,
                                  t.get("subject")),
                           t.labels),
            drop)

        return linear_form
