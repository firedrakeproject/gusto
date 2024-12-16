"""Classes for defining variants of the shallow-water equations."""

from firedrake import (inner, dx, div, FunctionSpace, FacetNormal, jump, avg,
                       dS, split)
from firedrake.fml import subject
from gusto.core.labels import (time_derivative, transport, prognostic,
                               linearisation, pressure_gradient, coriolis)
from gusto.equations.common_forms import (
    advection_form, advection_form_1d, continuity_form,
    continuity_form_1d, vector_invariant_form,
    kinetic_energy_form, advection_equation_circulation_form, diffusion_form_1d,
    linear_continuity_form
)
from gusto.equations.prognostic_equations import PrognosticEquationSet

__all__ = ["ShallowWaterEquations", "LinearShallowWaterEquations",
           "ShallowWaterEquations_1d", "LinearShallowWaterEquations_1d"]


class ShallowWaterEquations(PrognosticEquationSet):
    u"""
    Class for the (rotating) shallow-water equations, which evolve the velocity
    'u' and the depth field 'D', via some variant of:                         \n
    ∂u/∂t + (u.∇)u + f×u + g*∇(D+b) = 0,                                      \n
    ∂D/∂t + ∇.(D*u) = 0,                                                      \n
    for Coriolis parameter 'f' and bottom surface 'b'.
    """

    def __init__(self, domain, parameters, fexpr=None, bexpr=None,
                 space_names=None, linearisation_map='default',
                 u_transport_option='vector_invariant_form',
                 no_normal_flow_bc_ids=None, active_tracers=None,
                 thermal=False, max_quad_deg=5):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            parameters (:class:`Configuration`, optional): an object containing
                the model's physical parameters.
            fexpr (:class:`ufl.Expr`, optional): an expression for the Coroilis
                parameter. Defaults to None.
            bexpr (:class:`ufl.Expr`, optional): an expression for the bottom
                surface of the fluid. Defaults to None.
            space_names (dict, optional): a dictionary of strings for names of
                the function spaces to use for the spatial discretisation. The
                keys are the names of the prognostic variables. Defaults to None
                in which case the spaces are taken from the de Rham complex. Any
                buoyancy variable is taken by default to lie in the L2 space.
            linearisation_map (func, optional): a function specifying which
                terms in the equation set to linearise. If None is specified
                then no terms are linearised. Defaults to the string 'default',
                in which case the linearisation includes both time derivatives,
                the 'D' transport term and the pressure gradient term.
            u_transport_option (str, optional): specifies the transport term
                used for the velocity equation. Supported options are:
                'vector_invariant_form', 'vector_advection_form', and
                'circulation_form'.
                Defaults to 'vector_invariant_form'.
            no_normal_flow_bc_ids (list, optional): a list of IDs of domain
                boundaries at which no normal flow will be enforced. Defaults to
                None.
            active_tracers (list, optional): a list of `ActiveTracer` objects
                that encode the metadata for any active tracers to be included
                in the equations. Defaults to None.
            thermal (flag, optional): specifies whether the equations have a
                thermal or buoyancy variable that feeds back on the momentum.
                Defaults to False.
            max_quad_deg (int, optional): maximum quadrature degree for any
                form. Defaults to 5.

        Raises:
            NotImplementedError: active tracers are not yet implemented.
        """

        self.thermal = thermal
        field_names = ['u', 'D']

        if space_names is None:
            space_names = {'u': 'HDiv', 'D': 'L2'}

        if active_tracers is None:
            active_tracers = []

        if self.thermal:
            field_names.append('b')
            if 'b' not in space_names.keys():
                space_names['b'] = 'L2'

        if linearisation_map == 'default':
            # Default linearisation is time derivatives, pressure gradient and
            # transport term from depth equation. Don't include active tracers
            linearisation_map = lambda t: \
                t.get(prognostic) in ['u', 'D', 'b'] \
                and (any(t.has_label(time_derivative, pressure_gradient))
                     or (t.get(prognostic) in ['D', 'b'] and t.has_label(transport)))
        super().__init__(field_names, domain, space_names,
                         linearisation_map=linearisation_map,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers,
                         max_quad_deg=max_quad_deg)

        self.parameters = parameters
        g = parameters.g
        H = parameters.H

        w, phi = self.tests[0:2]
        u, D = split(self.X)[0:2]
        u_trial = split(self.trials)[0]

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

        # Depth transport term
        D_adv = prognostic(continuity_form(phi, D, u), 'D')

        # Transport term needs special linearisation
        if self.linearisation_map(D_adv.terms[0]):
            linear_D_adv = linear_continuity_form(phi, H, u_trial)
            # Add linearisation to D_adv
            D_adv = linearisation(D_adv, linear_D_adv)

        adv_form = subject(u_adv + D_adv, self.X)

        # Add transport of tracers
        if len(active_tracers) > 0:
            adv_form += self.generate_tracer_transport_terms(active_tracers)
        # Add transport of buoyancy, if thermal shallow water equations
        if self.thermal:
            gamma = self.tests[2]
            b = split(self.X)[2]
            b_adv = prognostic(advection_form(gamma, b, u), 'b')
            adv_form += subject(b_adv, self.X)

        # -------------------------------------------------------------------- #
        # Pressure Gradient Term
        # -------------------------------------------------------------------- #
        # Add pressure gradient only if not doing thermal
        if self.thermal:
            residual = (mass_form + adv_form)
        else:
            pressure_gradient_form = pressure_gradient(
                subject(prognostic(-g*div(w)*D*dx, 'u'), self.X))

            residual = (mass_form + adv_form + pressure_gradient_form)

        # -------------------------------------------------------------------- #
        # Extra Terms (Coriolis, Topography and Thermal)
        # -------------------------------------------------------------------- #
        # TODO: Is there a better way to store the Coriolis / topography fields?
        # The current approach is that these are prescribed fields, stored in
        # the equation, and initialised when the equation is

        if fexpr is not None:
            V = FunctionSpace(domain.mesh, 'CG', 1)
            f = self.prescribed_fields('coriolis', V).interpolate(fexpr)
            coriolis_form = coriolis(subject(
                prognostic(f*inner(domain.perp(u), w)*dx, "u"), self.X))
            # Add linearisation
            if self.linearisation_map(coriolis_form.terms[0]):
                linear_coriolis = coriolis(
                    subject(prognostic(f*inner(domain.perp(u_trial), w)*dx, 'u'), self.X))
                coriolis_form = linearisation(coriolis_form, linear_coriolis)
            residual += coriolis_form

        if bexpr is not None:
            topography = self.prescribed_fields('topography', domain.spaces('DG')).interpolate(bexpr)
            if self.thermal:
                n = FacetNormal(domain.mesh)
                topography_form = subject(prognostic
                                          (-topography*div(b*w)*dx
                                           + jump(b*w, n)*avg(topography)*dS,
                                           'u'), self.X)
            else:
                topography_form = subject(prognostic
                                          (-g*div(w)*topography*dx, 'u'),
                                          self.X)
            residual += topography_form

        # thermal source terms not involving topography.
        # label these as the equivalent pressure gradient term
        if self.thermal:
            n = FacetNormal(domain.mesh)
            source_form = pressure_gradient(subject(prognostic(-D*div(b*w)*dx
                                                    - 0.5*b*div(D*w)*dx
                                                    + jump(b*w, n)*avg(D)*dS
                                                    + 0.5*jump(D*w, n)*avg(b)*dS,
                                                    'u'), self.X))
            residual += source_form

        # -------------------------------------------------------------------- #
        # Linearise equations
        # -------------------------------------------------------------------- #
        # Add linearisations to equations
        self.residual = self.generate_linear_terms(residual, self.linearisation_map)


class LinearShallowWaterEquations(ShallowWaterEquations):
    u"""
    Class for the linear (rotating) shallow-water equations, which describe the
    velocity 'u' and the depth field 'D', solving some variant of:            \n
    ∂u/∂t + f×u + g*∇(D+b) = 0,                                               \n
    ∂D/∂t + H*∇.(u) = 0,                                                      \n
    for mean depth 'H', Coriolis parameter 'f' and bottom surface 'b'.

    This is set up the from the underlying :class:`ShallowWaterEquations`,
    which is then linearised.
    """

    def __init__(self, domain, parameters, fexpr=None, bexpr=None,
                 space_names=None, linearisation_map='default',
                 u_transport_option="vector_invariant_form",
                 no_normal_flow_bc_ids=None, active_tracers=None,
                 thermal=False, max_quad_deg=5):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            parameters (:class:`Configuration`, optional): an object containing
                the model's physical parameters.
            fexpr (:class:`ufl.Expr`, optional): an expression for the Coroilis
                parameter. Defaults to None.
            bexpr (:class:`ufl.Expr`, optional): an expression for the bottom
                surface of the fluid. Defaults to None.
            space_names (dict, optional): a dictionary of strings for names of
                the function spaces to use for the spatial discretisation. The
                keys are the names of the prognostic variables. Defaults to None
                in which case the spaces are taken from the de Rham complex. Any
                buoyancy variable is taken by default to lie in the L2 space.
            linearisation_map (func, optional): a function specifying which
                terms in the equation set to linearise. If None is specified
                then no terms are linearised. Defaults to the string 'default',
                in which case the linearisation includes both time derivatives,
                the 'D' transport term, pressure gradient and Coriolis terms.
            u_transport_option (str, optional): specifies the transport term
                used for the velocity equation. Supported options are:
                'vector_invariant_form', 'vector_advection_form' and
                'circulation_form'.
                Defaults to 'vector_invariant_form'.
            no_normal_flow_bc_ids (list, optional): a list of IDs of domain
                boundaries at which no normal flow will be enforced. Defaults to
                None.
            active_tracers (list, optional): a list of `ActiveTracer` objects
                that encode the metadata for any active tracers to be included
                in the equations. Defaults to None.
            thermal (flag, optional): specifies whether the equations have a
                thermal or buoyancy variable that feeds back on the momentum.
                Defaults to False.
            max_quad_deg (int, optional): maximum quadrature degree for any
                form. Defaults to 5.
        """

        if linearisation_map == 'default':
            # Default linearisation is time derivatives, pressure gradient,
            # Coriolis and transport term from depth equation
            linearisation_map = lambda t: \
                (any(t.has_label(time_derivative, pressure_gradient, coriolis))
                 or (t.get(prognostic) in ['D', 'b'] and t.has_label(transport)))

        super().__init__(domain, parameters,
                         fexpr=fexpr, bexpr=bexpr, space_names=space_names,
                         linearisation_map=linearisation_map,
                         u_transport_option=u_transport_option,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers, thermal=thermal,
                         max_quad_deg=max_quad_deg)

        # Use the underlying routine to do a first linearisation of the equations
        self.linearise_equation_set()


class ShallowWaterEquations_1d(PrognosticEquationSet):

    u"""
    Class for the (rotating) 1D shallow-water equations, which describe
    the velocity 'u', 'v' and the depth field 'D', solving some variant of:   \n
    ∂u/∂t + u∂u/∂x - fv + g*∂D/∂x = 0,                                        \n
    ∂v/∂t + fu = 0,                                                           \n
    ∂D/∂t + ∂(uD)/∂x = 0,                                                     \n
    for mean depth 'H', Coriolis parameter 'f' and gravity 'g'.

    Args:
        domain (:class:`Domain`): the model's domain object, containing the
            mesh and the compatible function spaces.
        parameters (:class:`Configuration`, optional): an object containing
            the model's physical parameters.
        fexpr (:class:`ufl.Expr`, optional): an expression for the Coroilis
            parameter. Defaults to None.
        space_names (dict, optional): a dictionary of strings for names of
            the function spaces to use for the spatial discretisation. The
            keys are the names of the prognostic variables. Defaults to None
            in which case the spaces are taken from the de Rham complex.
        linearisation_map (func, optional): a function specifying which
            terms in the equation set to linearise. If None is specified
            then no terms are linearised. Defaults to the string 'default',
            in which case the linearisation includes both time derivatives,
            the 'D' transport term, pressure gradient and Coriolis terms.
        no_normal_flow_bc_ids (list, optional): a list of IDs of domain
            boundaries at which no normal flow will be enforced. Defaults to
            None.
        active_tracers (list, optional): a list of `ActiveTracer` objects
            that encode the metadata for any active tracers to be included
            in the equations. Defaults to None.
        max_quad_deg (int, optional): maximum quadrature degree for any
            form. Defaults to 5.
    """

    def __init__(
            self, domain, parameters, fexpr=None, space_names=None,
            linearisation_map='default', diffusion_options=None,
            no_normal_flow_bc_ids=None, active_tracers=None, max_quad_deg=5
    ):

        field_names = ['u', 'v', 'D']
        space_names = {'u': 'HDiv', 'v': 'L2', 'D': 'L2'}

        if active_tracers is not None:
            raise NotImplementedError('Tracers not implemented for 1D shallow water equations')

        if linearisation_map == 'default':
            # Default linearisation is time derivatives, pressure gradient,
            # Coriolis and transport term from depth equation
            linearisation_map = lambda t: \
                (any(t.has_label(time_derivative, pressure_gradient, coriolis))
                 or (t.get(prognostic) == 'D' and t.has_label(transport)))

        super().__init__(
            field_names, domain, space_names,
            linearisation_map=linearisation_map,
            no_normal_flow_bc_ids=no_normal_flow_bc_ids,
            active_tracers=active_tracers, max_quad_deg=max_quad_deg
        )

        self.parameters = parameters
        g = parameters.g
        H = parameters.H

        w1, w2, phi = self.tests
        u, v, D = split(self.X)
        u_trial = split(self.trials)[0]

        # -------------------------------------------------------------------- #
        # Time Derivative Terms
        # -------------------------------------------------------------------- #
        mass_form = self.generate_mass_terms()

        # -------------------------------------------------------------------- #
        # Transport Terms
        # -------------------------------------------------------------------- #
        # Velocity transport term
        u_adv = prognostic(advection_form_1d(w1, u, u), 'u')
        v_adv = prognostic(advection_form_1d(w2, v, u), 'v')

        # Depth transport term
        D_adv = prognostic(continuity_form_1d(phi, D, u), 'D')

        # Transport term needs special linearisation
        if self.linearisation_map(D_adv.terms[0]):
            linear_D_adv = linear_continuity_form(phi, H, u_trial)
            # Add linearisation to D_adv
            D_adv = linearisation(D_adv, linear_D_adv)

        adv_form = subject(u_adv + v_adv + D_adv, self.X)

        pressure_gradient_form = pressure_gradient(subject(
            prognostic(-g * D * w1.dx(0) * dx, "u"), self.X))

        self.residual = (mass_form + adv_form
                         + pressure_gradient_form)

        if fexpr is not None:
            V = FunctionSpace(domain.mesh, 'CG', 1)
            f = self.prescribed_fields('coriolis', V).interpolate(fexpr)
            coriolis_form = coriolis(subject(
                prognostic(-f * v * w1 * dx, "u")
                + prognostic(f * u * w2 * dx, "v"), self.X))
            self.residual += coriolis_form

        if diffusion_options is not None:
            for field, diffusion in diffusion_options:
                idx = self.field_names.index(field)
                test = self.tests[idx]
                fn = split(self.X)[idx]
                self.residual += subject(
                    prognostic(diffusion_form_1d(test, fn, diffusion.kappa),
                               field),
                    self.X)

        # -------------------------------------------------------------------- #
        # Linearise equations
        # -------------------------------------------------------------------- #
        # Add linearisations to equations
        self.residual = self.generate_linear_terms(self.residual,
                                                   self.linearisation_map)


class LinearShallowWaterEquations_1d(ShallowWaterEquations_1d):
    u"""
    Class for the linear (rotating) 1D shallow-water equations, which describe
    the velocity 'u', 'v' and the depth field 'D', solving some variant of:  \n
    ∂u/∂t - fv + g*∂D/∂x = 0,                                                \n
    ∂v/∂t + fu = 0,                                                          \n
    ∂D/∂t + H*∂u/∂x = 0,                                                     \n
    for mean depth 'H', Coriolis parameter 'f' and gravity 'g'.

    This is set up the from the underlying :class:`ShallowWaterEquations_1d`,
    which is then linearised.
    """

    def __init__(self, domain, parameters, fexpr=None,
                 space_names=None, linearisation_map='default',
                 no_normal_flow_bc_ids=None, active_tracers=None,
                 max_quad_deg=5):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            parameters (:class:`Configuration`, optional): an object containing
                the model's physical parameters.
            fexpr (:class:`ufl.Expr`, optional): an expression for the Coroilis
                parameter. Defaults to None.
            space_names (dict, optional): a dictionary of strings for names of
                the function spaces to use for the spatial discretisation. The
                keys are the names of the prognostic variables. Defaults to None
                in which case the spaces are taken from the de Rham complex. Any
                buoyancy variable is taken by default to lie in the L2 space.
            linearisation_map (func, optional): a function specifying which
                terms in the equation set to linearise. If None is specified
                then no terms are linearised. Defaults to the string 'default',
                in which case the linearisation includes both time derivatives,
                the 'D' transport term, pressure gradient and Coriolis terms.
            no_normal_flow_bc_ids (list, optional): a list of IDs of domain
                boundaries at which no normal flow will be enforced. Defaults to
                None.
            active_tracers (list, optional): a list of `ActiveTracer` objects
                that encode the metadata for any active tracers to be included
                in the equations. Defaults to None.
            max_quad_deg (int, optional): maximum quadrature degree for any
                form. Defaults to 5.
            """

        if linearisation_map == 'default':
            # Default linearisation is time derivatives, pressure gradient,
            # Coriolis and transport term from depth equation
            linearisation_map = lambda t: \
                (any(t.has_label(time_derivative, pressure_gradient, coriolis))
                 or (t.get(prognostic) == 'D' and t.has_label(transport)))

        super().__init__(domain, parameters,
                         fexpr=fexpr, space_names=space_names,
                         linearisation_map=linearisation_map,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers,
                         max_quad_deg=max_quad_deg)

        # Use the underlying routine to do a first linearisation of the equations
        self.linearise_equation_set()
