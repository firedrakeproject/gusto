from abc import ABCMeta
from firedrake import (TestFunction, Function, inner, dx, div,
                       FunctionSpace, MixedFunctionSpace, TestFunctions,
                       TrialFunctions, FacetNormal, jump, avg, dS_v,
                       DirichletBC)
from gusto.form_manipulation_labelling import (subject, time_derivative,
                                               advection, prognostic,
                                               advecting_velocity, Term,
                                               all_terms, replace_subject,
                                               linearisation, name)
from gusto.thermodynamics import pi as Pi
from gusto.transport_equation import (advection_form, continuity_form,
                                      vector_invariant_form,
                                      vector_manifold_advection_form,
                                      kinetic_energy_form,
                                      advection_equation_circulation_form,
                                      linear_continuity_form,
                                      linear_advection_form, IntegrateByParts)
from gusto.diffusion import interior_penalty_diffusion_form
import ufl


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
            for fname in self.field_names:
                state.diagnostics.register(fname)
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

    def __init__(self, state, family, degree, fexpr=None,
                 u_advection_option="vector_invariant_form",
                 no_normal_flow_bc_ids=None):

        spaces = state.spaces.build_compatible_spaces(family, degree)
        W = MixedFunctionSpace(spaces)

        field_name = "_".join(self.field_names)
        super().__init__(state, W, field_name)

        Vu = self.function_space[0]
        self.bcs = []
        if no_normal_flow_bc_ids is None:
            no_normal_flow_bc_ids = []

        for id in no_normal_flow_bc_ids:
            self.bcs.append(DirichletBC(Vu, 0.0, id))

        V = FunctionSpace(state.mesh, "CG", 1)
        f = state.fields("coriolis", space=V)
        f.interpolate(fexpr)

        g = state.parameters.g
        H = state.parameters.H

        w, phi = TestFunctions(W)
        trials = TrialFunctions(W)
        X = Function(W)
        u, D = X.split()

        u_mass = subject(prognostic(inner(u, w)*dx, "u"), X)
        linear_u_mass = u_mass.label_map(all_terms,
                                         replace_subject(trials))
        u_mass = linearisation(u_mass, linear_u_mass)
        D_mass = subject(prognostic(inner(D, phi)*dx, "D"), X)
        linear_D_mass = D_mass.label_map(all_terms,
                                         replace_subject(trials))
        D_mass = linearisation(D_mass, linear_D_mass)
        mass_form = time_derivative(u_mass + D_mass)

        # define velocity advection term
        if u_advection_option == "vector_invariant_form":
            u_adv = prognostic(vector_invariant_form(state, w, u), "u")
        elif u_advection_option == "vector_advection_form":
            u_adv = prognostic(advection_form(state, w, u), "u")
        elif u_advection_option == "vector_manifold_advection_form":
            u_adv = prognostic(vector_manifold_advection_form(state, w, u), "u")
        elif u_advection_option == "circulation_form":
            ke_form = kinetic_energy_form(state, w, u)
            ke_form = advection.remove(ke_form)
            ke_form = ke_form.label_map(
                lambda t: t.has_label(advecting_velocity),
                lambda t: Term(ufl.replace(
                    t.form, {t.get(advecting_velocity): u}), t.labels))
            ke_form = advecting_velocity.remove(ke_form)
            u_adv = advection_equation_circulation_form(state, w, u) + ke_form
        else:
            raise ValueError("Invalid u_advection_option: %s" % u_advection_option)
        D_adv = prognostic(continuity_form(state, phi, D), "D")
        linear_D_adv = linear_continuity_form(state, phi, H).label_map(
            lambda t: t.has_label(advecting_velocity),
            lambda t: Term(ufl.replace(
                t.form, {t.get(advecting_velocity): trials[0]}), t.labels))
        D_adv = linearisation(D_adv, linear_D_adv)

        adv_form = subject(u_adv + D_adv, X)

        coriolis_form = subject(prognostic(f*inner(state.perp(u), w)*dx, "u"), X)

        pressure_gradient_form = subject(prognostic(-g*div(w)*D*dx, "u"), X)
        linear_pressure_gradient_form = pressure_gradient_form.label_map(
            all_terms, replace_subject(trials))
        pressure_gradient_form = linearisation(pressure_gradient_form,
                                               linear_pressure_gradient_form)

        self.residual = (mass_form + adv_form
                         + coriolis_form + pressure_gradient_form)


class CompressibleEulerEquations(PrognosticEquation):

    field_names = ['u', 'rho', 'theta']

    def __init__(self, state, family, degree, u_advection_option="vector_invariant_form", no_normal_flow_bc_ids=None):

        spaces = self._build_spaces(state, family, degree)
        W = MixedFunctionSpace(spaces)

        field_name = "_".join(self.field_names)
        super().__init__(state, W, field_name)

        Vu = W[0]
        self.bcs = []
        if no_normal_flow_bc_ids is None:
            no_normal_flow_bc_ids = []

        if Vu.extruded:
            self.bcs.append(DirichletBC(Vu, 0.0, "bottom"))
            self.bcs.append(DirichletBC(Vu, 0.0, "top"))
        for id in no_normal_flow_bc_ids:
            self.bcs.append(DirichletBC(Vu, 0.0, id))

        g = state.parameters.g
        cp = state.parameters.cp

        tests = TestFunctions(W)
        w, phi, gamma = tests[0:3]
        trials = TrialFunctions(W)
        X = Function(W)
        self.X = X
        u, rho, theta = X.split()[0:3]
        rhobar = state.fields("rhobar", space=rho.function_space(), dump=False)
        thetabar = state.fields("thetabar", space=theta.function_space(), dump=False)
        pi = Pi(state.parameters, rho, theta)
        n = FacetNormal(state.mesh)

        u_mass = subject(prognostic(inner(u, w)*dx, "u"), X)
        linear_u_mass = u_mass.label_map(all_terms,
                                         replace_subject(trials))
        u_mass = linearisation(u_mass, linear_u_mass)

        rho_mass = subject(prognostic(inner(rho, phi)*dx, "rho"), X)
        linear_rho_mass = rho_mass.label_map(all_terms,
                                             replace_subject(trials))
        rho_mass = linearisation(rho_mass, linear_rho_mass)

        theta_mass = subject(prognostic(inner(theta, gamma)*dx, "theta"), X)
        linear_theta_mass = theta_mass.label_map(all_terms,
                                                 replace_subject(trials))
        theta_mass = linearisation(theta_mass, linear_theta_mass)

        mass_form = time_derivative(u_mass + rho_mass + theta_mass)

        # define velocity advection term
        if u_advection_option == "vector_invariant_form":
            u_adv = prognostic(vector_invariant_form(state, w, u), "u")
        elif u_advection_option == "vector_advection_form":
            u_adv = prognostic(advection_form(state, w, u), "u")
        elif u_advection_option == "vector_manifold_advection_form":
            u_adv = prognostic(vector_manifold_advection_form(state, w, u), "u")
        elif u_advection_option == "circulation_form":
            ke_form = kinetic_energy_form(state, w, u)
            ke_form = advection.remove(ke_form)
            ke_form = ke_form.label_map(
                lambda t: t.has_label(advecting_velocity),
                lambda t: Term(ufl.replace(
                    t.form, {t.get(advecting_velocity): u}), t.labels))
            ke_form = advecting_velocity.remove(ke_form)
            u_adv = advection_equation_circulation_form(state, w, u) + ke_form
        else:
            raise ValueError("Invalid u_advection_option: %s" % u_advection_option)
        rho_adv = prognostic(continuity_form(state, phi, rho), "rho")
        linear_rho_adv = linear_continuity_form(state, phi, rhobar).label_map(
            lambda t: t.has_label(advecting_velocity),
            lambda t: Term(ufl.replace(
                t.form, {t.get(advecting_velocity): trials[0]}), t.labels))
        rho_adv = linearisation(rho_adv, linear_rho_adv)

        theta_adv = prognostic(advection_form(state, gamma, theta, ibp=IntegrateByParts.TWICE), "theta")
        linear_theta_adv = linear_advection_form(state, gamma, thetabar).label_map(
            lambda t: t.has_label(advecting_velocity),
            lambda t: Term(ufl.replace(
                t.form, {t.get(advecting_velocity): trials[0]}), t.labels))
        theta_adv = linearisation(theta_adv, linear_theta_adv)

        adv_form = subject(u_adv + rho_adv + theta_adv, X)

        # define pressure gradient term and its linearisation
        pressure_gradient_form = name(subject(prognostic(
            cp*(-div(theta*w)*pi*dx
                + jump(theta*w, n)*avg(pi)*dS_v), "u"), X), "pressure_gradient")

        # define gravity term and its linearisation
        gravity_form = subject(prognostic(Term(g*inner(state.k, w)*dx), "u"), X)

        self.residual = (mass_form + adv_form
                         + pressure_gradient_form + gravity_form)

    def _build_spaces(self, state, family, degree):
        return state.spaces.build_compatible_spaces(family, degree)


class MoistCompressibleEulerEquations(CompressibleEulerEquations):

    field_names = ['u', 'rho', 'theta', 'water_v', 'water_c']

    def __init__(self, state, family, degree,
                 u_advection_option="vector_invariant_form",
                 no_normal_flow_bc_ids=None):

        super().__init__(state, family, degree, u_advection_option,
                         no_normal_flow_bc_ids)

        Vth = state.spaces("theta")
        state.fields("water_vbar", space=Vth, dump=False)

        W = self.function_space
        _, _, gamma, p, q = TestFunctions(W)
        X = self.X
        u, _, theta, water_v, water_c = X.split()

        self.residual += time_derivative(subject(prognostic(inner(water_v, p)*dx, "water_v") + prognostic(inner(water_c, q)*dx, "water_c"), X))
        self.residual += subject(prognostic(advection_form(state, p, water_v), "water_v") + prognostic(advection_form(state, q, water_c), "water_c"), X)
        water_t = water_v + water_c

        self.residual = self.residual.label_map(
            lambda t: t.get(name) == "pressure_gradient",
            replace_subject(theta/(1+water_t), 2))

        cv = state.parameters.cv
        cp = state.parameters.cp
        c_vv = state.parameters.c_vv
        c_pv = state.parameters.c_pv
        c_pl = state.parameters.c_pl
        R_d = state.parameters.R_d
        R_v = state.parameters.R_v
        c_vml = cv + water_v * c_vv + water_c * c_pl
        c_pml = cp + water_v * c_pv + water_c * c_pl
        R_m = R_d + water_v * R_v

        self.residual -= subject(prognostic(gamma * theta * (R_m / c_vml - (R_d * c_pml) / (cp * c_vml)) * div(u)*dx, "theta"), X)

    def _build_spaces(self, state, family, degree):
        Vu, Vrho, Vth = state.spaces.build_compatible_spaces(family, degree)
        return Vu, Vrho, Vth, Vth, Vth
