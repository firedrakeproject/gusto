"""Defines variants of the compressible Euler equations."""

from firedrake import (
    sin, pi, inner, dx, div, cross, FunctionSpace, FacetNormal, jump, avg, dS_v,
    conditional, SpatialCoordinate, split, Constant, as_vector
)
from firedrake.fml import subject, replace_subject
from gusto.core.labels import (
    time_derivative, transport, prognostic, hydrostatic, linearisation,
    pressure_gradient, coriolis, gravity, sponge
)
from gusto.equations.thermodynamics import exner_pressure
from gusto.equations.common_forms import (
    advection_form, continuity_form, vector_invariant_form,
    kinetic_energy_form, advection_equation_circulation_form,
    diffusion_form, linear_continuity_form, linear_advection_form
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

    def __init__(self, domain, parameters, sponge_options=None,
                 extra_terms=None, space_names=None,
                 linearisation_map='default',
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
                then no terms are linearised. Defaults to the string 'default',
                in which case the linearisation includes time derivatives and
                scalar transport terms.
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

        if linearisation_map == 'default':
            # Default linearisation is time derivatives and scalar transport terms
            # Don't include active tracers
            linearisation_map = lambda t: \
                t.get(prognostic) in ['u', 'rho', 'theta'] \
                and (t.has_label(time_derivative)
                     or (t.get(prognostic) != 'u' and t.has_label(transport)))
        super().__init__(field_names, domain, space_names,
                         linearisation_map=linearisation_map,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        self.parameters = parameters
        g = parameters.g
        cp = parameters.cp

        w, phi, gamma = self.tests[0:3]
        u, rho, theta = split(self.X)[0:3]
        u_trial = split(self.trials)[0]
        _, rho_bar, theta_bar = split(self.X_ref)[0:3]
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
            u_adv = prognostic(vector_invariant_form(domain, w, u, u), 'u')
        elif u_transport_option == "vector_advection_form":
            u_adv = prognostic(advection_form(w, u, u), 'u')
        elif u_transport_option == "circulation_form":
            ke_form = prognostic(kinetic_energy_form(w, u, u), 'u')
            u_adv = prognostic(advection_equation_circulation_form(domain, w, u, u), 'u') + ke_form
        else:
            raise ValueError("Invalid u_transport_option: %s" % u_transport_option)

        # Density transport (conservative form)
        rho_adv = prognostic(continuity_form(phi, rho, u), 'rho')
        # Transport term needs special linearisation
        if self.linearisation_map(rho_adv.terms[0]):
            linear_rho_adv = linear_continuity_form(phi, rho_bar, u_trial)
            rho_adv = linearisation(rho_adv, linear_rho_adv)

        # Potential temperature transport (advective form)
        theta_adv = prognostic(advection_form(gamma, theta, u), 'theta')
        # Transport term needs special linearisation
        if self.linearisation_map(theta_adv.terms[0]):
            linear_theta_adv = linear_advection_form(gamma, theta_bar, u_trial)
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
            # TODO: add linearisation
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
                 extra_terms=None, space_names=None, linearisation_map='default',
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
                then no terms are linearised. Defaults to the string 'default',
                in which case the linearisation includes time derivatives and
                scalar transport terms.
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
