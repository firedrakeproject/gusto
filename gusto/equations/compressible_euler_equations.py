"""Defines variants of the compressible Euler equations."""

from firedrake import (
    sin, pi, inner, dx, div, cross, FunctionSpace, FacetNormal, jump, avg, dS_v,
    conditional, SpatialCoordinate, split, Constant, as_vector, ln, as_vector,
    grad
)
from firedrake.fml import subject, replace_subject, all_terms
from gusto.core.labels import (
    time_derivative, prognostic, hydrostatic, linearisation,
    pressure_gradient, coriolis, gravity, sponge, transport,
    transporting_velocity
)
from gusto.equations.thermodynamics import exner_pressure
from gusto.equations.common_forms import (
    advection_form, continuity_form, vector_invariant_form,
    kinetic_energy_form, advection_equation_circulation_form,
    diffusion_form, linear_continuity_form, linear_advection_form,
    linear_vector_invariant_form, linear_circulation_form,
    advection_form_1d
)
from gusto.equations.active_tracers import Phases, TracerVariableType
from gusto.equations.prognostic_equations import PrognosticEquationSet
from gusto.core.configuration import TransportEquationType
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

    def __init__(self, domain, parameters, sponge_options=None,
                 PML_options=None,
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
            PML_options (:class:`PMLParameters`, optional): any
                parameters for applying a PML layer to the upper boundary.
                This is an alternative to a sponge layer.
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

        if PML_options is not None:
            field_names.extend(['q_u', 'q_rho', 'q_theta'])
            space_names.update({'q_u': 'HDiv', 'q_rho': 'L2', 'q_theta': 'theta'})

        super().__init__(field_names, domain, space_names,
                         linearisation_map=linearisation_map,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        self.parameters = parameters
        g = parameters.g
        cp = parameters.cp

        if PML_options is not None:
            w, phi, gamma, q_u_test, q_rho_test, q_theta_test = self.tests[0:6]
            u, rho, theta, q_u, q_rho, q_theta = split(self.X)[0:6]
            u_trial, rho_trial, theta_trial, q_u_trial, q_rho_trial, q_theta_trial = split(self.trials)[0:6]
            u_bar, rho_bar, theta_bar, q_u_bar, q_rho_bar, q_theta_bar = split(self.X_ref)[0:6]
        else:
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
        # PML options, if using
        # -------------------------------------------------------------------- #
        if PML_options is not None:
            # Extract the key PML parameters
            c_max = PML_options.c_max
            delta_frac = PML_options.delta_frac
            tol = PML_options.tol
            gamma0 = PML_options.gamma0
            H = PML_options.H

            delta = delta_frac*H
            Lz = H - delta
            sigma0 = (4*c_max/(2*delta))*ln(1/tol)
            alpha = 0.05*sigma0

            x = SpatialCoordinate(domain.mesh)
            z = x[len(x)-1]

            print(len(x))

            sigma_expr = conditional(z <= Lz,
                                     0.0,
                                     sigma0*((z-Lz)/delta)**3)

            W_DG = FunctionSpace(domain.mesh, "DG", 2)
            self.sigma = self.prescribed_fields("PML", W_DG).interpolate(sigma_expr)

            self.gamma_z = self.prescribed_fields("gamma_z", W_DG).interpolate(Constant(1.0) + gamma0*self.sigma)

        # -------------------------------------------------------------------- #
        # Time Derivative Terms
        # -------------------------------------------------------------------- #
        mass_form = self.generate_mass_terms()

        # -------------------------------------------------------------------- #
        # Transport Terms
        # -------------------------------------------------------------------- #

        # If using a PML, scale the vertical advecting velocity:
        if PML_options is not None:
            scale_vect = as_vector([Constant(1.0), self.gamma_z])
            #u_bar = u*scale_vect
            u_advect = as_vector([u[0], u[1]/self.gamma_z])
            u_w = as_vector([Constant(0.0), u[1]/self.gamma_z])
            print(u)
            print(u_advect)
            print(u_w)
        else:
            u_advect = u

        # Velocity transport term -- depends on formulation
        if u_transport_option == "vector_invariant_form":
            u_adv = prognostic(vector_invariant_form(self.domain, w, u, u_advect), 'u')
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
            u_adv = prognostic(advection_form(w, u, u_advect), 'u')

        else:
            raise ValueError("Invalid u_transport_option: %s" % self.u_transport_option)

        # Density transport (conservative form)
        if PML_options is not None:
            # Directly construct the form, for now.
            L = inner(phi, (u[0]*rho).dx(0) + (1/self.gamma_z)*(u[1]*rho).dx(1))*dx
            form = transporting_velocity(L, u)
            rho_adv = prognostic(transport(form, TransportEquationType.conservative), 'rho')
        else:
            rho_adv = prognostic(continuity_form(phi, rho, u), 'rho')

        # Transport term needs special linearisation
        # TODO #651: we should remove this hand-coded linearisation
        # currently REXI can't handle generated transport linearisations
        if self.linearisation_map(rho_adv.terms[0]):
            print('linearising ...')
            # Hmmmm, this won't be right with the PML vertical scaling.
            linear_rho_adv = linear_continuity_form(phi, rho_trial, u_trial, rho_bar, u_bar)
            rho_adv = linearisation(rho_adv, linear_rho_adv)

        # Potential temperature transport (advective form)
        theta_adv = prognostic(advection_form(gamma, theta, u_advect), 'theta')

        # Transport term needs special linearisation
        # TODO #651: we should remove this hand-coded linearisation
        # currently REXI can't handle generated transport linearisations
        if self.linearisation_map(theta_adv.terms[0]):
            print('linearising ...')
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
        
        if PML_options is not None:
            # Modify the normal term
            pressure_gradient_form = pressure_gradient(subject(prognostic(
                cp*(- ((theta_v*w[0]).dx(0) + (1/self.gamma_z)*(theta_v*w[1]).dx(1))*exner*dx_qp
                    + jump(theta_v*w, n)*avg(exner)*dS_v_qp), 'u'), self.X))
            # Add a pressure gradient for the PML variable
            pressure_gradient_form += pressure_gradient(subject(prognostic(
                cp*(- (1/self.gamma_z)*(theta_v*q_u_test[1]).dx(1)*exner*dx_qp
                    + jump(theta_v*q_u_test, n)*avg(exner)*dS_v_qp), 'q_u'), self.X))

        else:
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
        # PML: Add extra terms
        # -------------------------------------------------------------------- #
        if PML_options is not None:

            # Add vertical advection PML terms for the PML variables
            # These are all 1d transport in the vertical, and 
            # have a minus sign in front of them.

            # Advective forms for q_u, q_theta
            q_u_adv = subject(prognostic(advection_form(q_u_test, q_u, u_w), 'q_u'), self.X)
            q_theta_adv = subject(prognostic(advection_form(q_theta_test, q_theta, u_w), 'q_theta'), self.X)

            # Conservative form for q_rho
            L = inner(q_rho_test, (1/self.gamma_z)*((u[1]*q_rho).dx(1)))*dx
            form = transporting_velocity(L, u_w)
            q_rho_adv = subject(prognostic(transport(form, TransportEquationType.conservative), 'q_rho'), self.X)

            # Make these terms negative in the residual
            residual -= subject(q_u_adv + q_theta_adv + q_rho_adv, self.X)

            # Probably should add linearisations, but see if we can get away without for now.
            
            # Six sigma PML terms, one for each equation
            # Negative sign for standard variables, positive for PML
            # These terms won't need linearisations.
            residual -= subject(prognostic(self.sigma*inner(w, q_u)*dx, 'u'), self.X)
            residual -= subject(prognostic(phi*self.sigma*q_rho*dx, 'rho'), self.X)
            residual -= subject(prognostic(gamma*self.sigma*q_theta*dx, 'theta'), self.X) 
            residual += subject(prognostic((self.sigma+alpha)*inner(q_u_test, q_u)*dx, 'q_u'), self.X) 
            residual += subject(prognostic(q_rho_test*(self.sigma+alpha)*q_rho*dx, 'q_rho'), self.X)
            residual += subject(prognostic(q_theta_test*(self.sigma+alpha)*q_theta*dx, 'q_theta'), self.X)

        # -------------------------------------------------------------------- #
        # Linearise equations
        # -------------------------------------------------------------------- #
        # Add linearisations to equations
        self.residual = self.generate_linear_terms(residual, self.linearisation_map)


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
