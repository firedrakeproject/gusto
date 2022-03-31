from abc import ABCMeta
from firedrake import (TestFunction, Function, sin, inner, dx, div, cross,
                       FunctionSpace, MixedFunctionSpace, TestFunctions,
                       TrialFunctions, FacetNormal, jump, avg, dS_v,
                       DirichletBC, conditional, SpatialCoordinate,
                       as_vector, split)
from gusto.fml.form_manipulation_labelling import drop, Term, all_terms
from gusto.labels import (subject, time_derivative, advection, prognostic,
                          advecting_velocity, replace_subject, linearisation,
                          name)
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
        self.bcs = {}

        if len(function_space) > 1:
            assert hasattr(self, "field_names")
            state.fields(field_name, function_space,
                         subfield_names=self.field_names)
            for fname in self.field_names:
                state.diagnostics.register(fname)
                self.bcs[fname] = []
        else:
            state.fields(field_name, function_space)
            state.diagnostics.register(field_name)
            self.bcs[field_name] = []


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
                 diffusion_parameters):
        super().__init__(state, function_space, field_name)

        test = TestFunction(function_space)
        q = Function(function_space)
        mass_form = time_derivative(inner(q, test)*dx)

        self.residual = subject(
            mass_form
            + interior_penalty_diffusion_form(
                state, test, q, diffusion_parameters), q
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
                 ufamily=None, udegree=None, diffusion_parameters=None,
                 **kwargs):
        super().__init__(state, function_space, field_name)

        if not hasattr(state.fields, "u"):
            V = state.spaces("HDiv", ufamily, udegree)
            state.fields("u", V)
        test = TestFunction(function_space)
        q = Function(function_space)
        mass_form = time_derivative(inner(q, test)*dx)

        self.residual = subject(
            mass_form
            + advection_form(state, test, q, **kwargs)
            + interior_penalty_diffusion_form(
                state, test, q, diffusion_parameters), q
        )


class ShallowWaterEquations(PrognosticEquation):

    field_names = ["u", "D"]

    def __init__(self, state, family, degree, fexpr=None, bexpr=None,
                 u_advection_option="vector_invariant_form",
                 no_normal_flow_bc_ids=None):

        spaces = state.spaces.build_compatible_spaces(family, degree)
        W = MixedFunctionSpace(spaces)

        field_name = "_".join(self.field_names)
        super().__init__(state, W, field_name)

        Vu = state.spaces("HDiv")
        if no_normal_flow_bc_ids is None:
            no_normal_flow_bc_ids = []

        for id in no_normal_flow_bc_ids:
            self.bcs['u'].append(DirichletBC(Vu, 0.0, id))

        g = state.parameters.g
        H = state.parameters.H

        w, phi = TestFunctions(W)
        trials = TrialFunctions(W)
        X = Function(W)
        u, D = split(X)
        self.ubar = u

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

        pressure_gradient_form = subject(prognostic(-g*div(w)*D*dx, "u"), X)
        linear_pressure_gradient_form = pressure_gradient_form.label_map(
            all_terms, replace_subject(trials))
        pressure_gradient_form = linearisation(pressure_gradient_form,
                                               linear_pressure_gradient_form)

        self.residual = (mass_form + adv_form + pressure_gradient_form)

        # add on optional coriolis and topography forms
        if fexpr is not None:
            V = FunctionSpace(state.mesh, "CG", 1)
            f = state.fields("coriolis", space=V)
            f.interpolate(fexpr)
            coriolis_form = subject(prognostic(f*inner(state.perp(u), w)*dx, "u"), X)
            self.residual += coriolis_form

        if bexpr is not None:
            b = state.fields("topography", state.spaces("DG"))
            b.interpolate(bexpr)
            topography_form = subject(prognostic(-g*div(w)*b*dx, "u"), X)
            self.residual += topography_form

        self.residual = self.residual.label_map(
            lambda t: t.has_label(advecting_velocity),
            lambda t: Term(ufl.replace(
                t.form, {t.get(advecting_velocity): u}), t.labels))


class CompressibleEulerEquations(PrognosticEquation):

    field_names = ['u', 'rho', 'theta']

    def __init__(self, state, family, degree, Omega=None, sponge=None,
                 extra_terms=None,
                 u_advection_option="vector_invariant_form",
                 diffusion_options=None,
                 no_normal_flow_bc_ids=None):

        spaces = self._build_spaces(state, family, degree)
        W = MixedFunctionSpace(spaces)

        field_name = "_".join(self.field_names)
        super().__init__(state, W, field_name)

        Vu = W.sub(0)
        if no_normal_flow_bc_ids is None:
            no_normal_flow_bc_ids = []

        if Vu.extruded:
            self.bcs['u'].append(DirichletBC(Vu, 0.0, "bottom"))
            self.bcs['u'].append(DirichletBC(Vu, 0.0, "top"))
        for id in no_normal_flow_bc_ids:
            self.bcs['u'].append(DirichletBC(Vu, 0.0, id))

        g = state.parameters.g
        cp = state.parameters.cp

        tests = TestFunctions(W)
        w, phi, gamma = tests[0:3]
        trials = TrialFunctions(W)
        X = Function(W)
        self.X = X
        u, rho, theta = split(X)[0:3]
        self.ubar = u
        rhobar = state.fields("rhobar", space=state.spaces("DG"), dump=False)
        thetabar = state.fields("thetabar", space=state.spaces("theta"), dump=False)
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

        # define velocity advection form
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

        # define pressure gradient form and its linearisation
        pressure_gradient_form = name(subject(prognostic(
            cp*(-div(theta*w)*pi*dx
                + jump(theta*w, n)*avg(pi)*dS_v), "u"), X), "pressure_gradient")

        # define gravity form and its linearisation
        gravity_form = subject(prognostic(Term(g*inner(state.k, w)*dx), "u"), X)

        self.residual = (mass_form + adv_form
                         + pressure_gradient_form + gravity_form)

        if Omega is not None:
            self.residual += subject(prognostic(
                inner(w, cross(2*Omega, u))*dx, "u"), X)

        if sponge is not None:
            W_DG = FunctionSpace(state.mesh, "DG", 2)
            x = SpatialCoordinate(state.mesh)
            z = x[len(x)-1]
            H = sponge.H
            zc = sponge.z_level
            mubar = sponge.mubar
            muexpr = conditional(z <= zc,
                                 0.0,
                                 mubar*sin((pi/2.)*(z-zc)/(H-zc))**2)
            self.mu = Function(W_DG).interpolate(muexpr)
            self.residual += name(subject(prognostic(
                self.mu*inner(w, state.k)*inner(u, state.k)*dx, "u"), X), "sponge")

        if diffusion_options is not None:
            for field, diffusion in diffusion_options:
                idx = self.field_names.index(field)
                test = tests[idx]
                fn = split(X)[idx]
                self.residual += subject(
                    prognostic(interior_penalty_diffusion_form(
                        state, test, fn, diffusion), field), X)

        if extra_terms is not None:
            for field, term in extra_terms:
                idx = self.field_names.index(field)
                test = tests[idx]
                self.residual += subject(prognostic(
                    inner(test, term)*dx, field), X)

    def _build_spaces(self, state, family, degree):
        return state.spaces.build_compatible_spaces(family, degree)


class MoistCompressibleEulerEquations(CompressibleEulerEquations):

    field_names = ['u', 'rho', 'theta', 'water_v', 'water_c']

    def __init__(self, state, family, degree, Omega=None, sponge=None,
                 u_advection_option="vector_invariant_form",
                 diffusion_options=None,
                 no_normal_flow_bc_ids=None):

        super().__init__(state, family, degree,
                         Omega=Omega, sponge=sponge,
                         u_advection_option=u_advection_option,
                         diffusion_options=diffusion_options,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids)

        Vth = state.spaces("theta")
        state.fields("water_vbar", space=Vth, dump=False)

        W = self.function_space
        w, _, gamma, p, q = TestFunctions(W)
        X = self.X
        u, rho, theta, water_v, water_c = split(X)

        self.residual += time_derivative(subject(prognostic(inner(water_v, p)*dx, "water_v") + prognostic(inner(water_c, q)*dx, "water_c"), X))
        self.residual += subject(prognostic(advection_form(state, p, water_v), "water_v") + prognostic(advection_form(state, q, water_c), "water_c"), X)

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

        water_t = water_v + water_c
        pi = Pi(state.parameters, rho, theta)
        n = FacetNormal(state.mesh)

        self.residual = self.residual.label_map(
            lambda t: t.get(name) == "pressure_gradient",
            drop)
        theta_v = theta/(1+water_t)
        self.residual += subject(prognostic(
            cp*(-div(theta_v*w)*pi*dx
                + jump(theta_v*w, n)*avg(pi)*dS_v), "u"), X)

        self.residual -= subject(prognostic(gamma * theta * (R_m / c_vml - (R_d * c_pml) / (cp * c_vml)) * div(u)*dx, "theta"), X)

    def _build_spaces(self, state, family, degree):
        Vu, Vrho, Vth = state.spaces.build_compatible_spaces(family, degree)
        return Vu, Vrho, Vth, Vth, Vth


class CompressibleEadyEquations(CompressibleEulerEquations):

    def __init__(self, state, family, degree, Omega=None, sponge=None,
                 u_advection_option="vector_invariant_form",
                 diffusion_options=None,
                 no_normal_flow_bc_ids=None):

        super().__init__(state, family, degree,
                         Omega=Omega, sponge=sponge,
                         u_advection_option=u_advection_option,
                         diffusion_options=diffusion_options,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids)

        dthetady = state.parameters.dthetady
        Pi0 = state.parameters.Pi0
        cp = state.parameters.cp
        y_vec = as_vector([0., 1., 0.])

        W = self.function_space
        w, _, gamma = TestFunctions(W)
        X = self.X
        u, rho, theta = split(X)

        pi = Pi(state.parameters, rho, theta)

        self.residual -= subject(prognostic(
            cp*dthetady*(pi-Pi0)*inner(w, y_vec)*dx, "u"), X)

        self.residual += subject(prognostic(
            gamma*(dthetady*inner(u, y_vec))*dx, "theta"), X)


class IncompressibleBoussinesqEquations(PrognosticEquation):

    field_names = ['u', 'p', 'b']

    def __init__(self, state, family, degree, Omega=None,
                 u_advection_option="vector_invariant_form",
                 no_normal_flow_bc_ids=None):

        spaces = state.spaces.build_compatible_spaces(family, degree)
        W = MixedFunctionSpace(spaces)

        field_name = "_".join(self.field_names)
        super().__init__(state, W, field_name)

        Vu = state.spaces("HDiv")
        if no_normal_flow_bc_ids is None:
            no_normal_flow_bc_ids = []

        if Vu.extruded:
            self.bcs['u'].append(DirichletBC(Vu, 0.0, "bottom"))
            self.bcs['u'].append(DirichletBC(Vu, 0.0, "top"))
        for id in no_normal_flow_bc_ids:
            self.bcs['u'].append(DirichletBC(Vu, 0.0, id))

        tests = TestFunctions(W)
        w, phi, gamma = tests[0:3]
        trials = TrialFunctions(W)
        X = Function(W)
        self.X = X
        u, p, b = split(X)
        bbar = state.fields("bbar", space=state.spaces("theta"), dump=False)
        bbar = state.fields("pbar", space=state.spaces("DG"), dump=False)

        u_mass = subject(prognostic(inner(u, w)*dx, "u"), X)
        linear_u_mass = u_mass.label_map(all_terms,
                                         replace_subject(trials))
        u_mass = linearisation(u_mass, linear_u_mass)

        p_mass = subject(prognostic(inner(p, phi)*dx, "p"), X)
        linear_p_mass = p_mass.label_map(all_terms,
                                         replace_subject(trials))
        p_mass = linearisation(p_mass, linear_p_mass)

        b_mass = subject(prognostic(inner(b, gamma)*dx, "b"), X)
        linear_b_mass = b_mass.label_map(all_terms,
                                         replace_subject(trials))
        b_mass = linearisation(b_mass, linear_b_mass)

        mass_form = time_derivative(u_mass + p_mass + b_mass)

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

        b_adv = prognostic(advection_form(state, gamma, b, ibp=IntegrateByParts.TWICE), "b")
        linear_b_adv = linear_advection_form(state, gamma, bbar).label_map(
            lambda t: t.has_label(advecting_velocity),
            lambda t: Term(ufl.replace(
                t.form, {t.get(advecting_velocity): trials[0]}), t.labels))
        b_adv = linearisation(b_adv, linear_b_adv)

        adv_form = subject(u_adv + b_adv, X)

        pressure_gradient_form = subject(prognostic(div(w)*p*dx, "u"), X)

        gravity_form = subject(prognostic(b*inner(w, state.k)*dx, "u"), X)

        divergence_form = name(subject(prognostic(phi*div(u)*dx, "p"), X),
                               "divergence_form")

        self.residual = (mass_form + adv_form + divergence_form
                         + pressure_gradient_form + gravity_form)

        if Omega is not None:
            self.residual += subject(prognostic(
                inner(w, cross(2*Omega, u))*dx, "u"), X)


class IncompressibleEadyEquations(IncompressibleBoussinesqEquations):
    def __init__(self, state, family, degree, Omega=None,
                 u_advection_option="vector_invariant_form",
                 no_normal_flow_bc_ids=None):

        super().__init__(state, family, degree,
                         Omega=Omega,
                         u_advection_option=u_advection_option,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids)

        dbdy = state.parameters.dbdy
        H = state.parameters.H
        _, _, z = SpatialCoordinate(state.mesh)
        eady_exp = Function(state.spaces("DG")).interpolate(z-H/2.)
        y_vec = as_vector([0., 1., 0.])

        W = self.function_space
        w, _, gamma = TestFunctions(W)
        X = self.X
        u, _, b = split(X)

        self.residual += subject(prognostic(
            dbdy*eady_exp*inner(w, y_vec)*dx, "u"), X)

        self.residual += subject(prognostic(
            gamma*dbdy*inner(u, y_vec)*dx, "b"), X)
