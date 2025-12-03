"""Defines variants of the compressible Euler equations."""

from firedrake import (
    sin, pi, inner, dx, div, cross, FunctionSpace, FacetNormal, jump, avg,
    conditional, SpatialCoordinate, split, Constant, as_vector, dot,
    DirichletBC, MixedFunctionSpace, TrialFunctions, TestFunctions,
    dS_v, dS_h, ds_v, ds_t, ds_b, ds_tb, grad

)
from firedrake.fml import subject, replace_subject, all_terms
from gusto.core.labels import (
    time_derivative, prognostic, hydrostatic, linearisation,
    pressure_gradient, coriolis, gravity, sponge
)
from gusto.equations.thermodynamics import exner_pressure, dexner_drho, dexner_dtheta
from gusto.equations.common_forms import (
    advection_form, continuity_form, vector_invariant_form,
    kinetic_energy_form, advection_equation_circulation_form,
    diffusion_form, linear_continuity_form, linear_advection_form,
    linear_vector_invariant_form, linear_circulation_form
)
from gusto.equations.active_tracers import Phases, TracerVariableType
from gusto.equations.prognostic_equations import PrognosticEquationSet
__all__ = ["CompressibleEulerEquations", "HydrostaticCompressibleEulerEquations"]


class CompressibleEulerEquations(PrognosticEquationSet):
    """
    Class for the compressible Euler equations, which evolve the velocity 'u',
    the dry density 'rho' and the (virtual dry) potential temperature 'theta',
    solving:                                                                  \n
    ∂u/∂t + (u.∇)u + 2Ω×u + c_p*θ*∇Π + g = 0,                                 \n
    ∂ρ/∂t + ∇.(ρ*u) = 0,                                                      \n
    ∂θ/∂t + (u.∇)θ = 0,                                                       \n
    where Π is the Exner pressure, g is the gravitational vector, Ω is the
    planet's rotation vector and c_p is the heat capacity of dry air at constant
    pressure.
    """

    name = "CompressibleEulerEquations"

    def __init__(self, domain, parameters, sponge_options=None,
                 extra_terms=None, space_names=None,
                 linearisation_map=all_terms,
                 u_transport_option="vector_invariant_form",
                 diffusion_options=None,
                 no_normal_flow_bc_ids=None,
                 active_tracers=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            x (:class:`Configuration`, optional): an object containing
                the model's physical parameters.
            sponge_options (:class:`SpongeLayerParameters`, optional): any
                parameters for applying a sponge layer to the upper boundary.
                Defaults to None.
            extra_terms (:class:`ufl.Expr`, optional): any extra terms to be
                included in the equation set. Defaults to None.
            space_names (dict, optional): a dictionary of strings for names of
                the function spaces to use for the spatial discretisation. The
                keys are the names of the prognostic variables. Defaults to None
                in which case the spaces are taken from the de Rham complex.
            linearisation_map (func, optional): a function specifying which
                terms in the equation set to linearise. If None is specified
                then no terms are linearised. Defaults to the FML `all_terms`
                function.
            u_transport_option (str, optional): specifies the transport term
                used for the velocity equation. Supported options are:
                'vector_invariant_form', 'vector_advection_form' and
                'circulation_form'.
                Defaults to 'vector_invariant_form'.
            diffusion_options (:class:`DiffusionParameters`, optional): any
                options to specify for applying diffusion terms to variables.
                Defaults to None.
            no_normal_flow_bc_ids (list, optional): a list of IDs of domain
                boundaries at which no normal flow will be enforced. Defaults to
                None.
            active_tracers (list, optional): a list of `ActiveTracer` objects
                that encode the metadata for any active tracers to be included
                in the equations.. Defaults to None.

        Raises:
            NotImplementedError: only mixing ratio tracers are implemented.
        """

        field_names = ['u', 'rho', 'theta']

        if space_names is None:
            space_names = {'u': 'HDiv', 'rho': 'L2', 'theta': 'theta'}

        if active_tracers is None:
            active_tracers = []

        super().__init__(field_names, domain, space_names,
                         linearisation_map=linearisation_map,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        self.parameters = parameters
        g = parameters.g
        cp = parameters.cp

        w, phi, gamma = self.tests[0:3]
        u, rho, theta = split(self.X)[0:3]
        u_trial, rho_trial, theta_trial = split(self.trials)[0:3]
        u_bar, rho_bar, theta_bar = split(self.X_ref)[0:3]
        zero_expr = Constant(0.0)*theta
        exner = exner_pressure(parameters, rho, theta)
        n = FacetNormal(domain.mesh)

        # Specify quadrature degree to use for pressure gradient term
        dx_qp = dx(degree=(domain.max_quad_degree))
        dS_v_qp = dS_v(degree=(domain.max_quad_degree))

        # -------------------------------------------------------------------- #
        # Time Derivative Terms
        # -------------------------------------------------------------------- #
        mass_form = self.generate_mass_terms()

        # -------------------------------------------------------------------- #
        # Transport Terms
        # -------------------------------------------------------------------- #
        # Velocity transport term -- depends on formulation
        if u_transport_option == "vector_invariant_form":
            u_adv = prognostic(vector_invariant_form(self.domain, w, u, u), 'u')
            # Manually add linearisation, as linearisation cannot handle the
            # perp function on the plane / vertical slice
            if self.linearisation_map(u_adv.terms[0]):
                linear_u_adv = linear_vector_invariant_form(self.domain, w, u_trial, u_bar)
                u_adv = linearisation(u_adv, linear_u_adv)

        elif u_transport_option == "circulation_form":
            # This is different to vector invariant form as the K.E. form
            # doesn't have a variable marked as "transporting velocity"
            ke_form = prognostic(kinetic_energy_form(w, u, u), 'u')
            circ_form = prognostic(advection_equation_circulation_form(self.domain, w, u, u), 'u')
            # Manually add linearisation, as linearisation cannot handle the
            # perp function on the plane / vertical slice
            if self.linearisation_map(circ_form.terms[0]):
                linear_circ_form = linear_circulation_form(self.domain, w, u_trial, u_bar)
                circ_form = linearisation(circ_form, linear_circ_form)
            u_adv = circ_form + ke_form

        elif u_transport_option == "vector_advection_form":
            u_adv = prognostic(advection_form(w, u, u), 'u')

        else:
            raise ValueError("Invalid u_transport_option: %s" % self.u_transport_option)

        # Density transport (conservative form)
        rho_adv = prognostic(continuity_form(phi, rho, u), 'rho')

        # Transport term needs special linearisation
        # TODO #651: we should remove this hand-coded linearisation
        # currently REXI can't handle generated transport linearisations
        if self.linearisation_map(rho_adv.terms[0]):
            linear_rho_adv = linear_continuity_form(phi, rho_trial, u_trial, rho_bar, u_bar)
            rho_adv = linearisation(rho_adv, linear_rho_adv)

        # Potential temperature transport (advective form)
        theta_adv = prognostic(advection_form(gamma, theta, u), 'theta')

        # Transport term needs special linearisation
        # TODO #651: we should remove this hand-coded linearisation
        # currently REXI can't handle generated transport linearisations
        if self.linearisation_map(theta_adv.terms[0]):
            linear_theta_adv = linear_advection_form(gamma, theta_trial, u_trial, theta_bar, u_bar)
            theta_adv = linearisation(theta_adv, linear_theta_adv)

        adv_form = subject(u_adv + rho_adv + theta_adv, self.X)

        # Add transport of tracers
        if len(active_tracers) > 0:
            adv_form += self.generate_tracer_transport_terms(active_tracers)

        # -------------------------------------------------------------------- #
        # Pressure Gradient Term
        # -------------------------------------------------------------------- #
        # First get total mass of tracers
        tracer_mr_total = zero_expr
        for tracer in active_tracers:
            if tracer.variable_type == TracerVariableType.mixing_ratio:
                idx = self.field_names.index(tracer.name)
                tracer_mr_total += split(self.X)[idx]
            else:
                raise NotImplementedError('Only mixing ratio tracers are implemented')
        theta_v = theta / (Constant(1.0) + tracer_mr_total)

        pressure_gradient_form = pressure_gradient(subject(prognostic(
            cp*(-div(theta_v*w)*exner*dx_qp
                + jump(theta_v*w, n)*avg(exner)*dS_v_qp), 'u'), self.X))

        # -------------------------------------------------------------------- #
        # Gravitational Term
        # -------------------------------------------------------------------- #
        gravity_form = gravity(subject(prognostic(g*inner(domain.k, w)*dx,
                                                  'u'), self.X))

        residual = (mass_form + adv_form + pressure_gradient_form
                    + gravity_form)

        # -------------------------------------------------------------------- #
        # Moist Thermodynamic Divergence Term
        # -------------------------------------------------------------------- #
        if len(active_tracers) > 0:
            cv = parameters.cv
            c_vv = parameters.c_vv
            c_pv = parameters.c_pv
            c_pl = parameters.c_pl
            R_d = parameters.R_d
            R_v = parameters.R_v

            # Get gas and liquid moisture mixing ratios
            mr_l = zero_expr
            mr_v = zero_expr

            for tracer in active_tracers:
                if tracer.chemical == 'H2O':
                    if tracer.variable_type == TracerVariableType.mixing_ratio:
                        idx = self.field_names.index(tracer.name)
                        if tracer.phase == Phases.gas:
                            mr_v += split(self.X)[idx]
                        elif tracer.phase == Phases.liquid:
                            mr_l += split(self.X)[idx]
                    else:
                        raise NotImplementedError('Only mixing ratio tracers are implemented')

            c_vml = cv + mr_v * c_vv + mr_l * c_pl
            c_pml = cp + mr_v * c_pv + mr_l * c_pl
            R_m = R_d + mr_v * R_v

            residual += subject(prognostic(
                gamma * theta * div(u)
                * (R_m / c_vml - (R_d * c_pml) / (cp * c_vml))*dx_qp, 'theta'), self.X)

        # -------------------------------------------------------------------- #
        # Extra Terms (Coriolis, Sponge, Diffusion and others)
        # -------------------------------------------------------------------- #
        if parameters.Omega is not None:
            Omega = as_vector((0, 0, parameters.Omega))
            coriolis_form = coriolis(subject(prognostic(
                inner(w, cross(2*Omega, u))*dx, 'u'), self.X))
            residual += coriolis_form

        if sponge_options is not None:
            W_DG = FunctionSpace(domain.mesh, "DG", 2)
            x = SpatialCoordinate(domain.mesh)
            z = x[len(x)-1]
            H = sponge_options.H
            zc = sponge_options.z_level
            assert float(zc) < float(H), \
                "The sponge level is set above the height the your domain"
            mubar = sponge_options.mubar
            muexpr = conditional(z <= zc,
                                 0.0,
                                 mubar*sin((pi/2.)*(z-zc)/(H-zc))**2)
            self.mu = self.prescribed_fields("sponge", W_DG).interpolate(muexpr)

            residual += sponge(subject(prognostic(
                self.mu*inner(w, domain.k)*inner(u, domain.k)*dx_qp, 'u'), self.X))

        if diffusion_options is not None:
            for field, diffusion in diffusion_options:
                idx = self.field_names.index(field)
                test = self.tests[idx]
                fn = split(self.X)[idx]
                residual += subject(
                    prognostic(diffusion_form(test, fn, diffusion.kappa), field),
                    self.X)

        if extra_terms is not None:
            for field, term in extra_terms:
                idx = self.field_names.index(field)
                test = self.tests[idx]
                residual += subject(prognostic(
                    inner(test, term)*dx, field), self.X)

        # -------------------------------------------------------------------- #
        # Linearise equations
        # -------------------------------------------------------------------- #
        # Add linearisations to equations
        self.residual = self.generate_linear_terms(residual, self.linearisation_map)

    def schur_complement_form(self, alpha=0.5, tau_values=None):
        domain = self.domain
        dt = domain.dt
        equations = self
        n = FacetNormal(self.domain.mesh)
        cp = self.parameters.cp
        tau_values = tau_values or {}
        # Set relaxation parameters. If an alternative has not been given, set
        # to semi-implicit off-centering factor
        beta_u = dt*tau_values.get("u", alpha)
        beta_t = dt*tau_values.get("theta", alpha)
        beta_r = dt*tau_values.get("rho", alpha)
        Vu = domain.spaces("HDiv")
        Vtheta = domain.spaces("theta")
        Vrho = domain.spaces("DG")

        # Build the reduced function space for u,p
        M = MixedFunctionSpace((Vu, Vrho))
        w, phi = TestFunctions(M)
        u, rho = TrialFunctions(M)

          # Get background fields
        _, rhobar, thetabar = split(self.X_ref)[0:3]
        exnerbar = exner_pressure(self.parameters, rhobar, thetabar)
        exnerbar_rho = dexner_drho(self.parameters, rhobar, thetabar)
        exnerbar_theta = dexner_dtheta(self.parameters, rhobar, thetabar)

        # Analytical (approximate) elimination of theta
        k = self.domain.k             # Upward pointing unit vector
        theta = -dot(k, u)*dot(k, grad(thetabar))*beta_t

        #q22 = - beta_t*beta_u*cp*(dot(k, w)*dot(k, grad(exner))*dot(k, u)*dot(k, grad(thetabar)))*dx

        # The exner prime term (here, bars are for mean and no bars are
        # for linear perturbations)
        exner = exnerbar_theta*theta + exnerbar_rho*rho

        # vertical projection
        def V(u):
            return k*inner(u, k)

        # hydrostatic projection
        h_project = lambda u: u - k*inner(u, k)

        # Specify degree for some terms as estimated degree is too large
        dx_qp = dx(degree=(equations.domain.max_quad_degree))
        dS_v_qp = dS_v(degree=(equations.domain.max_quad_degree))
        dS_h_qp = dS_h(degree=(equations.domain.max_quad_degree))
        ds_v_qp = ds_v(degree=(equations.domain.max_quad_degree))
        ds_tb_qp = (ds_t(degree=(equations.domain.max_quad_degree))
                    + ds_b(degree=(equations.domain.max_quad_degree)))

        # Add effect of density of water upon theta, using moisture reference profiles
        # TODO: Explore if this is the right thing to do for the linear problem
        if self.active_tracers is not None:
            mr_t = Constant(0.0)*thetabar
            for tracer in self.active_tracers:
                if tracer.chemical == 'H2O':
                    if tracer.variable_type == TracerVariableType.mixing_ratio:
                        idx = self.field_names.index(tracer.name)
                        mr_bar = split(self.X_ref)[idx]
                        mr_t += mr_bar
                    else:
                        raise NotImplementedError('Only mixing ratio tracers are implemented')

            theta_w = theta / (1 + mr_t)
            thetabar_w = thetabar / (1 + mr_t)
        else:
            theta_w = theta
            thetabar_w = thetabar

        # NOTE: no ds_v integrals since equations are defined on
        # a periodic (or sphere) base mesh.
        if any([t.has_label(hydrostatic) for t in self.residual]):
            u_mass = inner(w, (h_project(u)))*dx
        else:
            u_mass = inner(w, (u))*dx
        

        seqn = (
            # momentum equation
            u_mass
            - beta_u*cp*div(theta_w*V(w))*exnerbar*dx_qp
            # # following does nothing but is preserved in the comments
            # # to remind us why (because V(w) is purely vertical).
            # # + beta*cp*jump(theta_w*V(w), n=n)*exnerbar_avg('+')*dS_v_qp
            + beta_u*cp*jump(theta_w*V(w), n=n)*avg(exnerbar)*dS_h_qp
            + beta_u*cp*dot(theta_w*V(w), n)*exnerbar*ds_tb_qp
            - beta_u*cp*div(thetabar_w*w)*exner*dx_qp
            # Terms appearing after integrating momentum equation
            + beta_u*cp*jump(thetabar_w*w, n=n)*avg(exner)*(dS_v_qp + dS_h_qp)
            + beta_u*cp*dot(thetabar_w*w, n)*exner*(ds_tb_qp + ds_v_qp)
            # mass continuity equation
            + (phi*rho - beta_r*inner(grad(phi), u)*rhobar)*dx
            + beta_r*jump(phi*u, n=n)*avg(rhobar)*(dS_v + dS_h)
            # term added because u.n=0 is enforced weakly via the traces
            + beta_r*phi*dot(u, n)*rhobar*(ds_tb + ds_v)
            #(phi*rho + beta_r*phi*div(rhobar*u))*dx
        )

        if hasattr(self, "mu"):
            seqn += beta_u*self.mu*inner(w, k)*inner(u, k)*dx

        if self.parameters.Omega is not None:
            Omega = as_vector((0, 0, self.parameter.Omega))
            seqn += beta_u*inner(w, cross(2*Omega, u))*dx

        # Boundary conditions (assumes extruded mesh)
        # BCs are declared for the plain velocity space. As we need them in
        # a mixed problem, we replicate the BCs but for subspace of M
        sbcs = [DirichletBC(M.sub(0), bc.function_arg, bc.sub_domain) for bc in self.bcs['u']]

        return seqn, sbcs


class HydrostaticCompressibleEulerEquations(CompressibleEulerEquations):
    """
    The hydrostatic form of the compressible Euler equations. In this case the
    vertical velocity derivative is zero in the equations, so only 'u_h', the
    horizontal component of the velocity is allowed to vary in time. The
    equations, for velocity 'u', dry density 'rho' and (dry) potential
    temperature 'theta' are:                                                  \n
    ∂u_h/∂t + (u.∇)u_h + 2Ω×u + c_p*θ*∇Π + g = 0,                             \n
    ∂ρ/∂t + ∇.(ρ*u) = 0,                                                      \n
    ∂θ/∂t + (u.∇)θ = 0,                                                       \n
    where Π is the Exner pressure, g is the gravitational vector, Ω is the
    planet's rotation vector and c_p is the heat capacity of dry air at constant
    pressure.

    This is implemented through a hydrostatic switch to the compressible Euler
    equations.
    """

    def __init__(self, domain, parameters, sponge_options=None,
                 extra_terms=None, space_names=None,
                 linearisation_map=all_terms,
                 u_transport_option="vector_invariant_form",
                 diffusion_options=None,
                 no_normal_flow_bc_ids=None,
                 active_tracers=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            parameters (:class:`Configuration`, optional): an object containing
                the model's physical parameters.
            sponge_options (:class:`SpongeLayerParameters`, optional): any
                parameters for applying a sponge layer to the upper boundary.
                Defaults to None.
            extra_terms (:class:`ufl.Expr`, optional): any extra terms to be
                included in the equation set. Defaults to None.
            space_names (dict, optional): a dictionary of strings for names of
                the function spaces to use for the spatial discretisation. The
                keys are the names of the prognostic variables. Defaults to None
                in which case the spaces are taken from the de Rham complex.
            linearisation_map (func, optional): a function specifying which
                terms in the equation set to linearise. If None is specified
                then no terms are linearised. Defaults to the FML `all_terms`
                function.
            u_transport_option (str, optional): specifies the transport term
                used for the velocity equation. Supported options are:
                'vector_invariant_form', 'vector_advection_form' and
                'circulation_form'.
                Defaults to 'vector_invariant_form'.
            diffusion_options (:class:`DiffusionOptions`, optional): any options
                to specify for applying diffusion terms to variables. Defaults
                to None.
            no_normal_flow_bc_ids (list, optional): a list of IDs of domain
                boundaries at which no normal flow will be enforced. Defaults to
                None.
            active_tracers (list, optional): a list of `ActiveTracer` objects
                that encode the metadata for any active tracers to be included
                in the equations.. Defaults to None.

        Raises:
            NotImplementedError: only mixing ratio tracers are implemented.
        """

        super().__init__(domain, parameters, sponge_options=sponge_options,
                         extra_terms=extra_terms, space_names=space_names,
                         linearisation_map=linearisation_map,
                         u_transport_option=u_transport_option,
                         diffusion_options=diffusion_options,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        # Replace
        self.residual = self.residual.label_map(
            lambda t: t.has_label(time_derivative),
            map_if_true=lambda t: self.hydrostatic_projection(t, 'u')
        )

        # Add an extra hydrostatic term
        u_idx = self.field_names.index('u')
        u = split(self.X)[u_idx]
        k = self.domain.k
        self.residual += hydrostatic(
            subject(
                prognostic(
                    -inner(k, self.tests[u_idx]) * inner(k, u) * dx, "u"),
                self.X
            )
        )

    def hydrostatic_projection(self, term, field_name):
        """
        Performs the 'hydrostatic' projection.

        Takes a term involving a vector prognostic variable and replaces the
        prognostic with only its horizontal components. It also adds the
        'hydrostatic' label to that term.

        Args:
            term (:class:`Term`): the term to perform the projection upon.
            field_name (str): the prognostic field to make hydrostatic.

        Returns:
            :class:`LabelledForm`: the labelled form containing the new term.
        """

        f_idx = self.field_names.index(field_name)
        k = self.domain.k
        X = term.get(subject)
        field = split(X)[f_idx]

        new_subj = field - inner(field, k) * k
        # In one step:
        # - set up the replace_subject routine (which returns a function)
        # - call that function on the supplied `term` argument,
        #   to replace the subject with the new hydrostatic subject
        # - add the hydrostatic label
        return replace_subject(new_subj, old_idx=f_idx)(term)
