from abc import ABCMeta
from firedrake import (TestFunction, Function, sin, inner, dx, div, cross,
                       FunctionSpace, MixedFunctionSpace, TestFunctions,
                       TrialFunctions, FacetNormal, jump, avg, dS_v,
                       DirichletBC, conditional, SpatialCoordinate,
                       as_vector, split, Constant)
from gusto.fml.form_manipulation_labelling import Term, all_terms
from gusto.labels import (subject, time_derivative, transport, prognostic,
                          transporting_velocity, replace_subject, linearisation,
                          name)
from gusto.thermodynamics import pi as Pi
from gusto.transport_forms import (advection_form, continuity_form,
                                   vector_invariant_form,
                                   vector_manifold_advection_form,
                                   kinetic_energy_form,
                                   advection_equation_circulation_form,
                                   linear_continuity_form,
                                   linear_advection_form)
from gusto.diffusion import interior_penalty_diffusion_form
from gusto.active_tracers import ActiveTracer, Phases, TracerVariableType
from gusto.configuration import TransportEquationType
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
    Class defining the diffusion equation.

    :arg state: :class:`.State` object
    :arg function_space: :class:`.FunctionSpace` object
    :arg field_name: name of the prognostic field
    :kwargs: any kwargs to be passed on to the diffusion form
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

    def __init__(self, state, family, degree, fexpr=None, bexpr=None,
                 u_transport_option="vector_invariant_form",
                 no_normal_flow_bc_ids=None, active_tracers=None):

        self.field_names = ["u", "D"]

        spaces = state.spaces.build_compatible_spaces(family, degree)

        if active_tracers is not None:
            raise NotImplementedError('Tracers not implemented for shallow water equations')

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

        w, phi = TestFunctions(W)[0:2]
        trials = TrialFunctions(W)
        X = Function(W)
        u, D = split(X)

        u_mass = subject(prognostic(inner(u, w)*dx, "u"), X)
        linear_u_mass = u_mass.label_map(all_terms,
                                         replace_subject(trials))
        u_mass = linearisation(u_mass, linear_u_mass)
        D_mass = subject(prognostic(inner(D, phi)*dx, "D"), X)
        linear_D_mass = D_mass.label_map(all_terms,
                                         replace_subject(trials))
        D_mass = linearisation(D_mass, linear_D_mass)
        mass_form = time_derivative(u_mass + D_mass)

        # define velocity transport term
        if u_transport_option == "vector_invariant_form":
            u_adv = prognostic(vector_invariant_form(state, w, u), "u")
        elif u_transport_option == "vector_advection_form":
            u_adv = prognostic(advection_form(state, w, u), "u")
        elif u_transport_option == "vector_manifold_advection_form":
            u_adv = prognostic(vector_manifold_advection_form(state, w, u), "u")
        elif u_transport_option == "circulation_form":
            ke_form = kinetic_energy_form(state, w, u)
            ke_form = transport.remove(ke_form)
            ke_form = ke_form.label_map(
                lambda t: t.has_label(transporting_velocity),
                lambda t: Term(ufl.replace(
                    t.form, {t.get(transporting_velocity): u}), t.labels))
            ke_form = transporting_velocity.remove(ke_form)
            u_adv = advection_equation_circulation_form(state, w, u) + ke_form
        else:
            raise ValueError("Invalid u_transport_option: %s" % u_transport_option)
        D_adv = prognostic(continuity_form(state, phi, D), "D")
        linear_D_adv = linear_continuity_form(state, phi, H).label_map(
            lambda t: t.has_label(transporting_velocity),
            lambda t: Term(ufl.replace(
                t.form, {t.get(transporting_velocity): trials[0]}), t.labels))
        D_adv = linearisation(D_adv, linear_D_adv)

        adv_form = subject(u_adv + D_adv, X)

        # define pressure gradient form and its linearisation
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

        if diffusion_options is not None:
            tests = TestFunctions(W)
            for field, diffusion in diffusion_options:
                idx = self.field_names.index(field)
                test = tests[idx]
                fn = X.split()[idx]
                self.residual += subject(
                    prognostic(interior_penalty_diffusion_form(
                        state, test, fn, diffusion), field), X)

        if sponge is not None:
            #sponge_form = sponge*inner(w, state.k)*inner(u, state.k)*dx
            sponge_form_u = sponge * w[0] * u[0] * dx
            sponge_form_v = sponge * w[1] * u[1] * dx
            self.residual += subject(prognostic(sponge_form_u, "u"), X)
            self.residual += subject(prognostic(sponge_form_v, "u"), X)

    def _build_spaces(self, state, family, degree):
        Vu, VD = state.spaces.build_compatible_spaces(family, degree)
        return Vu, VD


class MoistShallowWaterEquations(ShallowWaterEquations):

    field_names = ["u", "D", "Q"]

    def __init__(self, state, family, degree, fexpr=None, bexpr=None,
                 sponge=None,
                 u_advection_option="vector_invariant_form",
                 diffusion_options=None,
                 no_normal_flow_bc_ids=None,
                 static_heating = None):

        if diffusion_options is not None:
            dry_diffusion_options = []
            for field, diffusion in diffusion_options:
                if field in super().field_names:
                    dry_diffusion_options.append((field, diffusion))
        else:
            dry_diffusion_options = None

        super().__init__(state, family, degree, fexpr=fexpr, bexpr=bexpr,
                         u_advection_option=u_advection_option,
                         diffusion_options=dry_diffusion_options,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids)

        W = self.function_space
        w, phi, gamma = TestFunctions(W)
        X = self.X
        u, D, Q = X.split()

        # add moisture evolution equation
        self.residual += (
            subject(
                prognostic(
                    time_derivative(inner(gamma, Q)*dx)
                    + advection(advection_form(state, gamma, Q)),
                    "Q"),
                X)
        )

        # add weak form of condensation to moisture evolution equation
        self.residual += (
            subject(prognostic(
                    condensation(gamma, Q, D, state.parameters),
                    "Q"),
                X)
        )

        # add weak form of evaporation to moisture evolution equation (add -E)
        self.residual -= (
            subject(
                prognostic(
                    evaporation(gamma, Q, state.parameters),
                    "Q"),
                X)
        )

        # add weak form of condensation forcing to depth equation
        self.residual += (
            subject(
                prognostic(
                    state.parameters.gamma * condensation(
                        phi, Q, D, state.parameters),
                    "D"),
                X)
        )

        # add weak form of the radiative relaxation term to the D equation
        self.residual += (
            subject(
                prognostic(
                    state.parameters.lamda_r * phi * (D-state.parameters.H) * dx,
                    "D"),
                X)
        )

        if diffusion_options is not None:
            tests = TestFunctions(W)
            for field, diffusion in diffusion_options:
                if field in set(self.field_names)-set(super().field_names):
                    idx = self.field_names.index(field)
                    test = tests[idx]
                    fn = X.split()[idx]
                    self.residual += subject(
                        prognostic(interior_penalty_diffusion_form(
                            state, test, fn, diffusion), field), X)

        if sponge is not None:
#            sponge_form = sponge*inner(w, state.k)*inner(u, state.k)*dx
#            self.residual += subject(prognostic(sponge_form, "u"), X)
            sponge_form_u = sponge * w[0] * u[0] * dx
            sponge_form_v = sponge * w[1] * u[1] * dx
            self.residual += subject(prognostic(sponge_form_u, "u"), X)

            self.residual += subject(prognostic(sponge_form_v, "u"), X)

        if static_heating is not None:
            heating_form = static_heating * phi * dx
            self.residual -= subject(prognostic(heating_form, "D"), X)

    def _build_spaces(self, state, family, degree):
        Vu, VD = state.spaces.build_compatible_spaces(family, degree)
        return Vu, VD, VD


class CompressibleEulerEquations(PrognosticEquation):

    def __init__(self, state, family, degree, Omega=None, sponge=None,
                 extra_terms=None,
                 u_transport_option="vector_invariant_form",
                 diffusion_options=None,
                 no_normal_flow_bc_ids=None,
                 active_tracers=None):

        self.field_names = ['u', 'rho', 'theta']

        # Construct spaces for the core set of prognostic variables
        spaces = [space for space in self._build_spaces(state, family, degree)]

        if active_tracers is None:
            active_tracers = []
        add_tracers_to_prognostics(state, self.field_names, spaces, active_tracers)

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

        g = state.parameters.g
        cp = state.parameters.cp

        tests = TestFunctions(W)
        w, phi, gamma = tests[0:3]
        trials = TrialFunctions(W)
        X = Function(W)
        self.X = X
        u, rho, theta = split(X)[0:3]
        rhobar = state.fields("rhobar", space=state.spaces("DG"), dump=False)
        thetabar = state.fields("thetabar", space=state.spaces("theta"), dump=False)
        zero_expr = Constant(0.0)*theta
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
        if len(active_tracers) > 0:
            mass_form += tracer_mass_forms(X, tests, self.field_names, active_tracers)

        # define velocity transport form
        if u_transport_option == "vector_invariant_form":
            u_adv = prognostic(vector_invariant_form(state, w, u), "u")
        elif u_transport_option == "vector_advection_form":
            u_adv = prognostic(advection_form(state, w, u), "u")
        elif u_transport_option == "vector_manifold_advection_form":
            u_adv = prognostic(vector_manifold_advection_form(state, w, u), "u")
        elif u_transport_option == "circulation_form":
            ke_form = kinetic_energy_form(state, w, u)
            ke_form = transport.remove(ke_form)
            ke_form = ke_form.label_map(
                lambda t: t.has_label(transporting_velocity),
                lambda t: Term(ufl.replace(
                    t.form, {t.get(transporting_velocity): u}), t.labels))
            ke_form = transporting_velocity.remove(ke_form)
            u_adv = advection_equation_circulation_form(state, w, u) + ke_form
        else:
            raise ValueError("Invalid u_transport_option: %s" % u_transport_option)
        rho_adv = prognostic(continuity_form(state, phi, rho), "rho")
        linear_rho_adv = linear_continuity_form(state, phi, rhobar).label_map(
            lambda t: t.has_label(transporting_velocity),
            lambda t: Term(ufl.replace(
                t.form, {t.get(transporting_velocity): trials[0]}), t.labels))
        rho_adv = linearisation(rho_adv, linear_rho_adv)

        theta_adv = prognostic(advection_form(state, gamma, theta), "theta")
        linear_theta_adv = linear_advection_form(state, gamma, thetabar).label_map(
            lambda t: t.has_label(transporting_velocity),
            lambda t: Term(ufl.replace(
                t.form, {t.get(transporting_velocity): trials[0]}), t.labels))
        theta_adv = linearisation(theta_adv, linear_theta_adv)

        adv_form = subject(u_adv + rho_adv + theta_adv, X)
        if len(active_tracers) > 0:
            adv_form += tracer_transport_forms(state, X, tests, self.field_names, active_tracers)

        # define pressure gradient form and its linearisation
        tracer_mr_total = zero_expr
        for tracer in active_tracers:
            if tracer.variable_type == TracerVariableType.mixing_ratio:
                idx = self.field_names.index(tracer.name)
                tracer_mr_total += split(X)[idx]
            else:
                raise NotImplementedError('Only mixing ratio tracers are implemented')
        theta_v = theta / (Constant(1.0) + tracer_mr_total)

        pressure_gradient_form = name(subject(prognostic(
            cp*(-div(theta_v*w)*pi*dx
                + jump(theta_v*w, n)*avg(pi)*dS_v), "u"), X), "pressure_gradient")

        # define gravity form and its linearisation
        gravity_form = subject(prognostic(Term(g*inner(state.k, w)*dx), "u"), X)

        self.residual = (mass_form + adv_form
                         + pressure_gradient_form + gravity_form)

        # define forcing term for theta
        if len(active_tracers) > 0:
            cv = state.parameters.cv
            c_vv = state.parameters.c_vv
            c_pv = state.parameters.c_pv
            c_pl = state.parameters.c_pl
            R_d = state.parameters.R_d
            R_v = state.parameters.R_v

            # Get gas and liquid moisture mixing ratios
            mr_l = zero_expr
            mr_v = zero_expr

            for tracer in active_tracers:
                if tracer.is_moisture:
                    if tracer.variable_type == TracerVariableType.mixing_ratio:
                        idx = self.field_names.index(tracer.name)
                        if tracer.phase == Phases.gas:
                            mr_v += split(X)[idx]
                        elif tracer.phase == Phases.liquid:
                            mr_l += split(X)[idx]
                    else:
                        raise NotImplementedError('Only mixing ratio tracers are implemented')

            c_vml = cv + mr_v * c_vv + mr_l * c_pl
            c_pml = cp + mr_v * c_pv + mr_l * c_pl
            R_m = R_d + mr_v * R_v

            self.residual -= subject(prognostic(gamma * theta * (R_m / c_vml - (R_d * c_pml) / (cp * c_vml)) * div(u)*dx, "theta"), X)

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


class CompressibleEadyEquations(CompressibleEulerEquations):

    def __init__(self, state, family, degree, Omega=None, sponge=None,
                 u_transport_option="vector_invariant_form",
                 diffusion_options=None,
                 no_normal_flow_bc_ids=None,
                 active_tracers=None):

        super().__init__(state, family, degree,
                         Omega=Omega, sponge=sponge,
                         u_transport_option=u_transport_option,
                         diffusion_options=diffusion_options,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

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

    def __init__(self, state, family, degree, Omega=None,
                 u_transport_option="vector_invariant_form",
                 no_normal_flow_bc_ids=None,
                 active_tracers=None):

        self.field_names = ['u', 'p', 'b']

        spaces = state.spaces.build_compatible_spaces(family, degree)

        if active_tracers is not None:
            raise NotImplementedError('Tracers not implemented for Boussinesq equations')

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

        # define velocity transport term
        if u_transport_option == "vector_invariant_form":
            u_adv = prognostic(vector_invariant_form(state, w, u), "u")
        elif u_transport_option == "vector_advection_form":
            u_adv = prognostic(advection_form(state, w, u), "u")
        elif u_transport_option == "vector_manifold_advection_form":
            u_adv = prognostic(vector_manifold_advection_form(state, w, u), "u")
        elif u_transport_option == "circulation_form":
            ke_form = kinetic_energy_form(state, w, u)
            ke_form = transport.remove(ke_form)
            ke_form = ke_form.label_map(
                lambda t: t.has_label(transporting_velocity),
                lambda t: Term(ufl.replace(
                    t.form, {t.get(transporting_velocity): u}), t.labels))
            ke_form = transporting_velocity.remove(ke_form)
            u_adv = advection_equation_circulation_form(state, w, u) + ke_form
        else:
            raise ValueError("Invalid u_transport_option: %s" % u_transport_option)

        b_adv = prognostic(advection_form(state, gamma, b), "b")
        linear_b_adv = linear_advection_form(state, gamma, bbar).label_map(
            lambda t: t.has_label(transporting_velocity),
            lambda t: Term(ufl.replace(
                t.form, {t.get(transporting_velocity): trials[0]}), t.labels))
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
                 u_transport_option="vector_invariant_form",
                 no_normal_flow_bc_ids=None,
                 active_tracers=None):

        super().__init__(state, family, degree,
                         Omega=Omega,
                         u_transport_option=u_transport_option,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

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


# ============================================================================ #
# Active Tracer Routines
# ============================================================================ #

# These should eventually be moved to be routines belonging to the
# PrognosticEquation class
# For now they are here so that equation sets above are kept tidy

def add_tracers_to_prognostics(state, field_names, spaces, active_tracers):
    """
    This routine adds the active tracers to the equation sets.

    :arg state:          The `State` object.
    :arg field_names:    The list of field names.
    :arg spaces:         The list of `FunctionSpace` objects to form the
                         `MixedFunctionSpace`.
    :arg active_tracers: A list of `ActiveTracer` objects that encode the
                         metadata for the active tracers.
    """

    # Loop through tracer fields and add field names and spaces
    for tracer in active_tracers:
        if isinstance(tracer, ActiveTracer):
            if tracer.name not in field_names:
                field_names.append(tracer.name)
            else:
                raise ValueError(f'There is already a field named {tracer.name}')
            spaces.append(state.spaces(tracer.space))
        else:
            raise ValueError(f'Tracers must be ActiveTracer objects, not {type(tracer)}')


def tracer_mass_forms(X, tests, field_names, active_tracers):
    """
    Adds the mass forms for the active tracers to the equation.

    :arg X:              The prognostic variables on the `MixedFunctionSpace`.
    :arg tests:          `TestFunctions` for the prognostic variables.
    :arg field_names:    The list of field names.
    :arg active_tracers: A list of `ActiveTracer` objects that encode the
                         metadata for the active tracers.
    """

    for i, tracer in enumerate(active_tracers):
        idx = field_names.index(tracer.name)
        tracer_prog = split(X)[idx]
        tracer_test = tests[idx]
        tracer_mass = subject(prognostic(inner(tracer_prog, tracer_test)*dx, tracer.name), X)
        if i == 0:
            mass_form = time_derivative(tracer_mass)
        else:
            mass_form += time_derivative(tracer_mass)

    return mass_form


def tracer_transport_forms(state, X, tests, field_names, active_tracers):
    """
    Adds the transport forms for the active tracers to the equation.

    :arg state:          The `State` object.
    :arg X:              The prognostic variables on the `MixedFunctionSpace`.
    :arg tests:          `TestFunctions` for the prognostic variables.
    :arg field_names:    The list of field names.
    :arg active_tracers: A list of `ActiveTracer` objects that encode the
                         metadata for the active tracers.
    """

    for i, tracer in enumerate(active_tracers):
        if tracer.transport_flag:
            idx = field_names.index(tracer.name)
            tracer_prog = split(X)[idx]
            tracer_test = tests[idx]
            if tracer.transport_eqn == TransportEquationType.advective:
                tracer_adv = prognostic(advection_form(state, tracer_test, tracer_prog), tracer.name)
            elif tracer.transport_eqn == TransportEquationType.conservative:
                tracer_adv = prognostic(continuity_form(state, tracer_test, tracer_prog), tracer.name)
            else:
                raise ValueError(f'Transport eqn {tracer.transport_eqn} not recognised')

            if i == 0:
                adv_form = subject(tracer_adv, X)
            else:
                adv_form += subject(tracer_adv, X)

    return adv_form
