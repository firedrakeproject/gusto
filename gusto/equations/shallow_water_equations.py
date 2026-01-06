"""Classes for defining variants of the shallow-water equations."""

from firedrake import (inner, dx, div, FunctionSpace, FacetNormal, jump, avg,
                       dS, split, conditional, exp, sqrt, Function,
                       SpatialCoordinate, Constant)
from firedrake.fml import subject, drop, all_terms
from gusto.core.labels import (
    linearisation, pressure_gradient, coriolis, prognostic
)
from gusto.core.equation_configuration import CoriolisOptions
from gusto.core.kernels import MinKernel
from gusto.equations.common_forms import (
    advection_form, advection_form_1d, continuity_form,
    continuity_form_1d, vector_invariant_form,
    kinetic_energy_form, advection_equation_circulation_form, diffusion_form_1d,
    linear_continuity_form, linear_advection_form,
    linear_vector_invariant_form, linear_circulation_form
)
from gusto.equations.prognostic_equations import PrognosticEquationSet


__all__ = ["ShallowWaterEquations", "LinearShallowWaterEquations",
           "ThermalShallowWaterEquations",
           "LinearThermalShallowWaterEquations",
           "ShallowWaterEquations_1d", "LinearShallowWaterEquations_1d"]


class ShallowWaterEquations(PrognosticEquationSet):
    u"""
    Class for the (rotating) shallow-water equations, which evolve the velocity
    'u' and the depth field 'D', via some variant of:                         \n
    ∂u/∂t + (u.∇)u + f×u + g*∇(D+B) = 0,                                      \n
    ∂D/∂t + ∇.(D*u) = 0,                                                      \n
    for Coriolis parameter 'f' and bottom surface 'B'.
    """

    def __init__(self, domain, parameters,
                 space_names=None, linearisation_map=all_terms,
                 u_transport_option='vector_invariant_form',
                 no_normal_flow_bc_ids=None, active_tracers=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            parameters (:class:`Configuration`, optional): an object containing
                the model's physical parameters.
            space_names (dict, optional): a dictionary of strings for names of
                the function spaces to use for the spatial discretisation. The
                keys are the names of the prognostic variables. Defaults to None
                in which case the spaces are taken from the de Rham complex. Any
                buoyancy variable is taken by default to lie in the L2 space.
            linearisation_map (func, optional): a function specifying which
                terms in the equation set to linearise. If None is specified
                then no terms are linearised. Defaults to the FML `all_terms`
                function.
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
        """

        if active_tracers is None:
            active_tracers = []

        field_names = ['u', 'D']
        space_names = {'u': 'HDiv', 'D': 'L2'}

        super().__init__(field_names, domain, space_names,
                         linearisation_map=linearisation_map,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        self.parameters = parameters
        self.domain = domain
        self.active_tracers = active_tracers

        self._setup_residual(u_transport_option)

        # -------------------------------------------------------------------- #
        # Linearise equations
        # -------------------------------------------------------------------- #
        # Add linearisations to equations
        self.residual = self.generate_linear_terms(
            self.residual, self.linearisation_map)

    def _setup_residual(self, u_transport_option):
        """
        Sets up the residual for the shallow water equations. This
        is separate from the __init__ method because the thermal
        shallow water equation class expands on this equation set by
        adding additional fields that depend on the formulation. This
        increases the size of the mixed function space and the
        residual must be setup after this has happened.

        Args:
            u_transport_option (str): specifies the transport term
                used for the velocity equation. Supported options are:
                'vector_invariant_form', 'vector_advection_form', and
                'circulation_form'.
        """

        g = self.parameters.g

        w, phi = self.tests[0:2]
        u, D = split(self.X)[0:2]
        u_trial, D_trial = split(self.trials)[0:2]
        ubar, Dbar = split(self.X_ref)[0:2]

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
                linear_u_adv = linear_vector_invariant_form(self.domain, w, u_trial, ubar)
                u_adv = linearisation(u_adv, linear_u_adv)

        elif u_transport_option == "circulation_form":
            # This is different to vector invariant form as the K.E. form
            # doesn't have a variable marked as "transporting velocity"
            ke_form = prognostic(kinetic_energy_form(w, u, u), 'u')
            circ_form = prognostic(advection_equation_circulation_form(self.domain, w, u, u), 'u')
            # Manually add linearisation, as linearisation cannot handle the
            # perp function on the plane / vertical slice
            if self.linearisation_map(circ_form.terms[0]):
                linear_circ_form = linear_circulation_form(self.domain, w, u_trial, ubar)
                circ_form = linearisation(circ_form, linear_circ_form)
            u_adv = circ_form + ke_form

        elif u_transport_option == "vector_advection_form":
            u_adv = prognostic(advection_form(w, u, u), 'u')

        else:
            raise ValueError("Invalid u_transport_option: %s" % self.u_transport_option)

        # Depth transport term
        D_adv = prognostic(continuity_form(phi, D, u), 'D')

        # Transport term needs special linearisation
        # TODO #651: we should remove this hand-coded linearisation
        # currently removing it breaks the REXI tests
        if self.linearisation_map(D_adv.terms[0]):
            linear_D_adv = linear_continuity_form(phi, D_trial, u_trial, Dbar, ubar)
            # Add linearisation to D_adv
            D_adv = linearisation(D_adv, linear_D_adv)

        adv_form = subject(u_adv + D_adv, self.X)

        # Add transport of tracers
        if len(self.active_tracers) > 0:
            adv_form += self.generate_tracer_transport_terms(
                self.active_tracers)

        # -------------------------------------------------------------------- #
        # Pressure Gradient Term
        # -------------------------------------------------------------------- #
        # On assuming ``g``, it is right to keep it out of the integral.
        pressure_gradient_form = pressure_gradient(
            subject(prognostic(-g*(div(w)*D*dx), 'u'), self.X))

        residual = (mass_form + adv_form + pressure_gradient_form)

        # -------------------------------------------------------------------- #
        # Extra Terms (Coriolis, Topography)
        # -------------------------------------------------------------------- #
        # TODO: Is there a better way to store the Coriolis / topography fields?
        # The current approach is that these are prescribed fields, stored in
        # the equation, and initialised when the equation is

        quad = self.domain.max_quad_degree
        rotation = self.parameters.rotation
        if rotation is not CoriolisOptions.nonrotating:
            CG1 = FunctionSpace(self.domain.mesh, "CG", 1)
            if rotation is CoriolisOptions.sphere:
                assert self.domain.on_sphere
                xyz = SpatialCoordinate(self.domain.mesh)
                r = sqrt(inner(xyz, xyz))
                radius_field = Function(CG1)
                radius_field.interpolate(r)
                min_kernel = MinKernel()
                radius = Constant(min_kernel.apply(radius_field))
                fexpr = 2*self.parameters.Omega*xyz[2]/radius
            elif rotation is CoriolisOptions.fplane:
                fexpr = self.parameters.f0
            elif rotation is CoriolisOptions.betaplane:
                xyz = SpatialCoordinate(self.domain.mesh)
                fexpr = self.parameters.f0 + self.parameters.beta * xyz[1]
            else:
                raise NotImplementedError('Coriolis option is not implemented')
            self.prescribed_fields('coriolis', CG1).interpolate(fexpr)
            coriolis_form = coriolis(subject(prognostic(
                fexpr*inner(self.domain.perp(u), w)*dx(degree=quad), "u"), self.X))
            # Add linearisation manually, as linearisation cannot handle the
            # perp function on the plane / vertical slice
            if self.linearisation_map(coriolis_form.terms[0]):
                linear_coriolis = coriolis(subject(prognostic(
                    fexpr*inner(self.domain.perp(u_trial), w)*dx(degree=quad), 'u'), self.X))
                coriolis_form = linearisation(coriolis_form, linear_coriolis)
            residual += coriolis_form

        topog_expr = self.parameters.topog_expr
        if topog_expr is not None:
            self.prescribed_fields('topography', self.domain.spaces('DG')).interpolate(topog_expr)
            topography_form = subject(prognostic
                                      (-g*div(w)*topog_expr*dx(degree=quad), 'u'),
                                      self.X)
            residual += topography_form

        self.residual = residual


class LinearShallowWaterEquations(ShallowWaterEquations):
    u"""
    Class for the linear (rotating) shallow-water equations, which describe the
    velocity 'u' and the depth field 'D', solving some variant of:            \n
    ∂u/∂t + f×u + g*∇(D+B) = 0,                                               \n
    ∂D/∂t + H*∇.(u) = 0,                                                      \n
    for mean depth 'H', Coriolis parameter 'f' and bottom surface 'B'.

    This is set up from the underlying :class:`ShallowWaterEquations`,
    which is then linearised.
    """

    def __init__(self, domain, parameters,
                 space_names=None, linearisation_map=all_terms,
                 u_transport_option="vector_invariant_form",
                 no_normal_flow_bc_ids=None, active_tracers=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            parameters (:class:`Configuration`, optional): an object containing
                the model's physical parameters.
            space_names (dict, optional): a dictionary of strings for names of
                the function spaces to use for the spatial discretisation. The
                keys are the names of the prognostic variables. Defaults to None
                in which case the spaces are taken from the de Rham complex. Any
                buoyancy variable is taken by default to lie in the L2 space.
            linearisation_map (func, optional): a function specifying which
                terms in the equation set to linearise. If None is specified
                then no terms are linearised. Defaults to the FML `all_terms`
                function.
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
        """

        super().__init__(domain, parameters,
                         space_names=space_names,
                         linearisation_map=linearisation_map,
                         u_transport_option=u_transport_option,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        # Use the underlying routine to do a first linearisation of the equations
        self.linearise_equation_set()


class ThermalShallowWaterEquations(ShallowWaterEquations):
    u"""
    Class for the (rotating) shallow-water equations, which evolve the velocity
    'u' and the depth field 'D' via some variant of either:                  \n
    ∂u/∂t + (u.∇)u + f×u + b*∇(D+B) + 0.5*D*∇b= 0,                            \n
    ∂D/∂t + ∇.(D*u) = 0,                                                      \n
    ∂b/∂t + u.∇(b) = 0,                                                       \n
    for Coriolis parameter 'f', bottom surface 'B' and buoyancy field b,
    or, if equivalent_buoyancy=True:
    ∂u/∂t + (u.∇)u + f×u + b_e*∇(D+B) + 0.5*D*∇(b_e + beta_2 q_v)= 0,         \n
    ∂D/∂t + ∇.(D*u) = 0,                                                      \n
    ∂b_e/∂t + u.∇(b_e) = 0,                                                   \n
    ∂q_t/∂t + u.∇(q_t) = 0,                                                   \n
    for Coriolis parameter 'f', bottom surface 'B', equivalent buoyancy field \n
    `b_e`=b-beta_2 q_v, and total moisture `q_t`=q_v+q_c, i.e. the sum of     \n
    water vapour and cloud water.
    """

    def __init__(self, domain, parameters, equivalent_buoyancy=False,
                 space_names=None, linearisation_map=all_terms,
                 u_transport_option='vector_invariant_form',
                 no_normal_flow_bc_ids=None, active_tracers=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            parameters (:class:`Configuration`, optional): an object containing
                the model's physical parameters.
            equivalent_buoyancy (bool, optional): switch to specify formulation
                (see comments above). Defaults to False to give standard
                thermal shallow water.
            space_names (dict, optional): a dictionary of strings for names of
                the function spaces to use for the spatial discretisation. The
                keys are the names of the prognostic variables. Defaults to None
                in which case the spaces are taken from the de Rham complex. Any
                buoyancy variable is taken by default to lie in the L2 space.
            linearisation_map (func, optional): a function specifying which
                terms in the equation set to linearise. If None is specified
                then no terms are linearised. Defaults to the FML `all_terms`
                function.
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
        """

        self.equivalent_buoyancy = equivalent_buoyancy
        field_names = ['u', 'D']
        space_names = {'u': 'HDiv', 'D': 'L2'}
        self.b_name = 'b_e' if equivalent_buoyancy else 'b'

        if equivalent_buoyancy:
            for new_field in [self.b_name, 'q_t']:
                field_names.append(new_field)
                space_names[new_field] = 'L2'
        else:
            field_names.append(self.b_name)
            space_names[self.b_name] = 'L2'

        if active_tracers is None:
            active_tracers = []

        # Bypass ShallowWaterEquations.__init__ to avoid having to
        # define the field_names separately
        PrognosticEquationSet.__init__(
            self, field_names, domain, space_names,
            linearisation_map=linearisation_map,
            no_normal_flow_bc_ids=no_normal_flow_bc_ids,
            active_tracers=active_tracers)

        self.parameters = parameters
        self.domain = domain
        self.active_tracers = active_tracers

        self._setup_residual(u_transport_option)

        # -------------------------------------------------------------------- #
        # Linearise equations
        # -------------------------------------------------------------------- #
        # Add linearisations to equations
        self.residual = self.generate_linear_terms(
            self.residual, self.linearisation_map)

    def _setup_residual(self, u_transport_option):
        """
        Sets up the residual for the thermal shallow water
        equations, first calling the shallow water equation
        _setup_residual method to get the standard shallow water forms.

        Args:
            u_transport_option (str): specifies the transport term
                used for the velocity equation. Supported options are:
                'vector_invariant_form', 'vector_advection_form', and
                'circulation_form'.

        """
        # don't pass topography to super class as we deal with those
        # terms here later
        super()._setup_residual(u_transport_option=u_transport_option)

        w = self.tests[0]
        gamma = self.tests[2]
        u, D, b = split(self.X)[0:3]
        ubar, Dbar, bbar = split(self.X_ref)[0:3]
        u_trial, D_trial, b_trial = split(self.trials)[0:3]
        n = FacetNormal(self.domain.mesh)

        if self.equivalent_buoyancy:
            gamma_qt = self.tests[3]
            qt = split(self.X)[3]
            qtbar = split(self.X_ref)[3]
            qt_trial = split(self.trials)[3]

        # -------------------------------------------------------------------- #
        # Add pressure gradient-like terms to residual
        # -------------------------------------------------------------------- #
        # drop usual pressure gradient term
        residual = self.residual.label_map(
            lambda t: t.has_label(pressure_gradient),
            drop)

        # add (moist) thermal source terms not involving topography -
        # label these as the equivalent pressure gradient term and
        # provide linearisation
        if self.equivalent_buoyancy:
            beta2 = self.parameters.beta2

            qsat_expr = self.compute_saturation(self.X)
            qv = conditional(qt < qsat_expr, qt, qsat_expr)
            qvbar = conditional(qtbar < qsat_expr, qtbar, qsat_expr)
            source_form = pressure_gradient(subject(prognostic(
                -D * div(b*w) * dx - 0.5 * b * div(D*w) * dx
                + jump(b*w, n) * avg(D) * dS + 0.5 * jump(D*w, n) * avg(b) * dS
                - beta2 * D * div(qv*w)*dx - 0.5 * beta2 * qv * div(D*w) * dx
                + beta2 * jump(qv*w, n) * avg(D) * dS
                + 0.5 * beta2 * jump(D*w, n) * avg(qv) * dS,
                'u'), self.X))

            # TODO #651: we should remove this hand-coded linearisation
            # currently removing it throws errors when using the linear
            # thermal shallow water equations
            # Care will be needed with the handling of the qv terms
            linear_source_form = pressure_gradient(subject(prognostic(
                -D_trial * div(bbar*w) * dx
                - 0.5 * b_trial * div(Dbar*w) * dx
                + jump(bbar*w, n) * avg(D_trial) * dS
                + 0.5 * jump(Dbar*w, n) * avg(b_trial) * dS
                - beta2 * D_trial * div(qvbar*w)*dx
                - 0.5 * beta2 * qvbar * div(Dbar*w) * dx
                + beta2 * jump(qvbar*w, n) * avg(D_trial) * dS
                + 0.5 * beta2 * jump(Dbar*w, n) * avg(qvbar) * dS
                - 0.5 * bbar * div(Dbar*w) * dx
                + 0.5 * jump(Dbar*w, n) * avg(bbar) * dS
                - 0.5 * bbar * div(D_trial*w) * dx
                + 0.5 * jump(D_trial*w, n) * avg(bbar) * dS
                - beta2 * 0.5 * qvbar * div(D_trial*w) * dx
                + beta2 * 0.5 * jump(D_trial*w, n) * avg(qvbar) * dS
                - beta2 * 0.5 * qt_trial * div(Dbar*w) * dx
                + beta2 * 0.5 * jump(Dbar*w, n) * avg(qt_trial) * dS,
                'u'), self.X))
            source_form = linearisation(source_form, linear_source_form)

        else:
            source_form = pressure_gradient(
                subject(prognostic(-D * div(b*w) * dx
                                   + jump(b*w, n) * avg(D) * dS
                                   - 0.5 * b * div(D*w) * dx
                                   + 0.5 * jump(D*w, n) * avg(b) * dS,
                                   'u'), self.X))
            # TODO #651: we should remove this hand-coded linearisation
            # currently removing it throws errors when using the linear
            # thermal shallow water equations
            linear_source_form = pressure_gradient(
                subject(prognostic(-D_trial * div(bbar*w) * dx
                                   + jump(bbar*w, n) * avg(D_trial) * dS
                                   - 0.5 * b_trial * div(Dbar*w) * dx
                                   + 0.5 * jump(Dbar*w, n) * avg(b_trial) * dS
                                   - 0.5 * bbar * div(Dbar*w) * dx
                                   + 0.5 * jump(Dbar*w, n) * avg(bbar) * dS
                                   - 0.5 * bbar * div(D_trial*w) * dx
                                   + 0.5 * jump(D_trial*w, n) * avg(bbar) * dS,
                                   'u'), self.X))
            source_form = linearisation(source_form, linear_source_form)

        residual += source_form

        # -------------------------------------------------------------------- #
        # add transport terms and their linearisations:
        # -------------------------------------------------------------------- #
        b_adv = prognostic(advection_form(gamma, b, u), self.b_name)

        # TODO #651: we should remove this hand-coded linearisation
        # currently REXI can't handle generated transport linearisations
        if self.linearisation_map(b_adv.terms[0]):
            linear_b_adv = linear_advection_form(gamma, b_trial, u_trial, bbar, ubar)
            b_adv = linearisation(b_adv, linear_b_adv)
        residual += subject(b_adv, self.X)

        if self.equivalent_buoyancy:
            qt_adv = prognostic(advection_form(gamma_qt, qt, u), "q_t")

            # TODO #651: we should remove this hand-coded linearisation
            # currently REXI can't handle generated transport linearisations
            if self.linearisation_map(qt_adv.terms[0]):
                linear_qt_adv = linear_advection_form(gamma_qt, qt_trial, u_trial, qtbar, ubar)
                qt_adv = linearisation(qt_adv, linear_qt_adv)
            residual += subject(qt_adv, self.X)

        # -------------------------------------------------------------------- #
        # add topography terms:
        # -------------------------------------------------------------------- #
        topog_expr = self.parameters.topog_expr
        if topog_expr is not None:
            topog = self.prescribed_fields(
                'topography', self.domain.spaces('DG')).interpolate(topog_expr)
            if self.equivalent_buoyancy:
                topography_form = subject(prognostic(
                    - topog * div(b*w) * dx
                    + jump(b*w, n) * avg(topog) * dS
                    - beta2 * topog * div(qv*w) * dx
                    + beta2 * jump(qv*w, n) * avg(topog) * dS,
                    'u'), self.X)
            else:
                topography_form = subject(prognostic(
                    - topog * div(b*w) * dx
                    + jump(b*w, n) * avg(topog) * dS,
                    'u'), self.X)
            residual += topography_form

        self.residual = residual

    def compute_saturation(self, X):
        # Returns the saturation expression as a function of the
        # parameters specified in self.parameters and the input
        # functions X. The latter are left as inputs to the
        # function so that it can also be used for initialisation
        q0 = self.parameters.q0
        nu = self.parameters.nu
        g = self.parameters.g
        H = self.parameters.H
        D, b = split(X)[1:3]

        if self.parameters.topog_expr is None:
            sat_expr = q0*H/(D) * exp(nu*(1-b/g))
        else:
            topog = self.prescribed_fields('topography')
            sat_expr = q0*H/(D+topog) * exp(nu*(1-b/g))
        return sat_expr


class LinearThermalShallowWaterEquations(ThermalShallowWaterEquations):
    u"""
    Class for the linear (rotating) thermal shallow-water equations, which
    describe the velocity 'u' and depth field 'D', solving some variant of:   \n
    ∂u/∂t + f×u + bbar*∇D + 0.5*H*∇b = 0,                                     \n
    ∂D/∂t + H*∇.(u) = 0,                                                      \n
    ∂b/∂t + u.∇bbar = 0,                                                      \n
    ∂q_t/∂t + u.∇(q_tbar) = 0,                                                \n
    for mean depth 'H', mean buoyancy `bbar`, Coriolis parameter 'f'

    This is set up from the underlying :class:`ThermalShallowWaterEquations`,
    which is then linearised.
    """

    def __init__(self, domain, parameters, equivalent_buoyancy=False,
                 space_names=None, linearisation_map=all_terms,
                 u_transport_option="vector_invariant_form",
                 no_normal_flow_bc_ids=None, active_tracers=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            parameters (:class:`Configuration`, optional): an object containing
                the model's physical parameters.
            equivalent_buoyancy (bool, optional): switch to specify formulation
                (see comments above). Defaults to False to give standard
                thermal shallow water.
            space_names (dict, optional): a dictionary of strings for names of
                the function spaces to use for the spatial discretisation. The
                keys are the names of the prognostic variables. Defaults to None
                in which case the spaces are taken from the de Rham complex. Any
                buoyancy variable is taken by default to lie in the L2 space.
            linearisation_map (func, optional): a function specifying which
                terms in the equation set to linearise. If None is specified
                then no terms are linearised. Defaults to the FML `all_terms`
                function.
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
        """

        super().__init__(domain, parameters,
                         equivalent_buoyancy=equivalent_buoyancy,
                         space_names=space_names,
                         linearisation_map=linearisation_map,
                         u_transport_option=u_transport_option,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

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
        space_names (dict, optional): a dictionary of strings for names of
            the function spaces to use for the spatial discretisation. The
            keys are the names of the prognostic variables. Defaults to None
            in which case the spaces are taken from the de Rham complex.
            linearisation_map (func, optional): a function specifying which
                terms in the equation set to linearise. If None is specified
                then no terms are linearised. Defaults to the FML `all_terms`
                function.
        no_normal_flow_bc_ids (list, optional): a list of IDs of domain
            boundaries at which no normal flow will be enforced. Defaults to
            None.
        active_tracers (list, optional): a list of `ActiveTracer` objects
            that encode the metadata for any active tracers to be included
            in the equations. Defaults to None.
    """

    def __init__(self, domain, parameters,
                 space_names=None, linearisation_map=all_terms,
                 diffusion_options=None,
                 no_normal_flow_bc_ids=None, active_tracers=None):

        field_names = ['u', 'v', 'D']
        space_names = {'u': 'HDiv', 'v': 'L2', 'D': 'L2'}

        if active_tracers is not None:
            raise NotImplementedError('Tracers not implemented for 1D shallow water equations')

        super().__init__(field_names, domain, space_names,
                         linearisation_map=linearisation_map,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        self.parameters = parameters
        g = parameters.g

        w1, w2, phi = self.tests
        u, v, D = split(self.X)
        ubar, _, Dbar = split(self.X_ref)
        u_trial, _, D_trial = split(self.trials)

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
        # TODO #651: we should remove this hand-coded linearisation
        if self.linearisation_map(D_adv.terms[0]):
            linear_D_adv = linear_continuity_form(phi, D_trial, u_trial, Dbar, ubar)
            # Add linearisation to D_adv
            D_adv = linearisation(D_adv, linear_D_adv)

        adv_form = subject(u_adv + v_adv + D_adv, self.X)

        pressure_gradient_form = pressure_gradient(subject(
            prognostic(-g * D * w1.dx(0) * dx, "u"), self.X))

        self.residual = (mass_form + adv_form
                         + pressure_gradient_form)

        rotation = self.parameters.rotation
        if rotation is not CoriolisOptions.nonrotating:
            quad = domain.max_quad_degree
            V = FunctionSpace(domain.mesh, 'CG', 1)
            if rotation is CoriolisOptions.fplane:
                fexpr = self.parameters.f0
            else:
                raise NotImplementedError('Coriolis option is not implemented')
            self.prescribed_fields('coriolis', V).interpolate(fexpr)
            coriolis_form = coriolis(subject(
                prognostic(-fexpr * v * w1 * dx(degree=quad), "u")
                + prognostic(fexpr * u * w2 * dx(degree=quad), "v"), self.X))
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

    def __init__(self, domain, parameters,
                 space_names=None, linearisation_map=all_terms,
                 no_normal_flow_bc_ids=None, active_tracers=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            parameters (:class:`Configuration`, optional): an object containing
                the model's physical parameters.
            space_names (dict, optional): a dictionary of strings for names of
                the function spaces to use for the spatial discretisation. The
                keys are the names of the prognostic variables. Defaults to None
                in which case the spaces are taken from the de Rham complex. Any
                buoyancy variable is taken by default to lie in the L2 space.
            linearisation_map (func, optional): a function specifying which
                terms in the equation set to linearise. If None is specified
                then no terms are linearised. Defaults to the FML `all_terms`
                function.
            no_normal_flow_bc_ids (list, optional): a list of IDs of domain
                boundaries at which no normal flow will be enforced. Defaults to
                None.
            active_tracers (list, optional): a list of `ActiveTracer` objects
                that encode the metadata for any active tracers to be included
                in the equations. Defaults to None.
        """

        super().__init__(domain, parameters,
                         space_names=space_names,
                         linearisation_map=linearisation_map,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        # Use the underlying routine to do a first linearisation of the equations
        self.linearise_equation_set()
