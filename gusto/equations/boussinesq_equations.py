"""Defines the Boussinesq equations."""

from firedrake import (
    inner, dx, div, cross, split, as_vector, TestFunctions, TrialFunctions, dot, grad, DirichletBC,
    Constant, SpatialCoordinate, conditional, FunctionSpace, ln, sin, pi, grad
)
from firedrake.fml import subject, all_terms
from gusto.core.labels import (
    prognostic, linearisation,
    pressure_gradient, coriolis, divergence, gravity, incompressible, sponge
)
from gusto.equations.common_forms import (
    advection_form, vector_invariant_form,
    kinetic_energy_form, advection_equation_circulation_form,
    linear_advection_form, linear_circulation_form,
    linear_vector_invariant_form
)
from gusto.equations.prognostic_equations import PrognosticEquationSet

__all__ = ["BoussinesqEquations", "LinearBoussinesqEquations", "LinearAcousticBuoyancyEquations"]


class BoussinesqEquations(PrognosticEquationSet):
    """
    Class for the Boussinesq equations, which evolve the velocity
    'u', the pressure 'p' and the buoyancy 'b'. Can be compressible or
    incompressible, depending on the value of the input flag, which defaults
    to compressible.

    The compressible form of the equations is
    ∂u/∂t + (u.∇)u + 2Ω×u + ∇p - b*k = 0,                                     \n
    ∂p/∂t + (u.∇)p + cs**2 ∇.u = 0,                                           \n
    ∂b/∂t + (u.∇)b = 0,                                                       \n
    where k is the vertical unit vector, Ω is the planet's rotation vector
    and cs is the sound speed.

    For the incompressible form of the equations, the pressure features as
    a Lagrange multiplier to enforce incompressibility. The equations are     \n
    ∂u/∂t + (u.∇)u + 2Ω×u + ∇p - b*k = 0,                                     \n
    ∇.u = p,                                                                  \n
    ∂b/∂t + (u.∇)b = 0,                                                       \n
    where k is the vertical unit vector and Ω is the planet's rotation vector.
    """

    def __init__(self, domain, parameters,
                 compressible=True,
                 sponge_options=None,
                 PML_options=None,
                 space_names=None,
                 linearisation_map=all_terms,
                 u_transport_option="vector_invariant_form",
                 no_normal_flow_bc_ids=None,
                 active_tracers=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            parameters (:class:`Configuration`, optional): an object containing
                the model's physical parameters.
            compressible (bool, optional): flag to indicate whether the
                equations are compressible. Defaults to True
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
            no_normal_flow_bc_ids (list, optional): a list of IDs of domain
                boundaries at which no normal flow will be enforced. Defaults to
                None.
            active_tracers (list, optional): a list of `ActiveTracer` objects
                that encode the metadata for any active tracers to be included
                in the equations.. Defaults to None.

        Raises:
            NotImplementedError: active tracers are not implemented.
        """

        field_names = ['u', 'p', 'b']

        if space_names is None:
            space_names = {'u': 'HDiv', 'p': 'L2', 'b': 'theta'}

        if PML_options is not None:
            field_names.extend(['q_u', 'q_p'])
            space_names.update({'q_u': 'HDiv', 'q_p': 'L2'})

            # Include buoyancy terms:
            #field_names.extend(['q_u', 'q_p', 'q_b'])
            #space_names.update({'q_u': 'HDiv', 'q_p': 'L2', 'q_b': 'theta'})

        print(space_names)

        if active_tracers is not None:
            raise NotImplementedError('Tracers not implemented for Boussinesq equations')

        if active_tracers is None:
            active_tracers = []

        super().__init__(field_names, domain, space_names,
                         linearisation_map=linearisation_map,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        self.parameters = parameters
        self.compressible = compressible

        if PML_options is not None:
            w, phi, gamma, q_u_test, q_p_test = self.tests[0:5]
            u, p, b, q_u, q_p= split(self.X)[0:5]
            u_trial, p_trial, b_trial, q_u_trial, q_p_trial = split(self.trials)[0:5]
            u_bar, p_bar, b_bar, q_u_bar, q_p_bar = split(self.X_ref)[0:5]

            # What about including a buoyancy term??
            # Currently don't do this.
            #w, phi, gamma, q_u_test, q_p_test, q_b_test = self.tests[0:6]
            #u, p, b, q_u, q_p, q_b = split(self.X)[0:6]
            #u_trial, p_trial, b_trial, q_u_trial, q_p_trial, q_b_trial = split(self.trials)[0:6]
            #u_bar, p_bar, b_bar, q_u_bar, q_p_bar, q_b_bar = split(self.X_ref)[0:6]

            # Define perturbations:
            u_pert = u - u_bar
            p_pert = p - p_bar
            b_pert = b - b_bar
        else:
            w, phi, gamma = self.tests[0:3]
            u, p, b = split(self.X)
            u_trial, p_trial, b_trial = split(self.trials)[0:3]
            u_bar, p_bar, b_bar = split(self.X_ref)[0:3]

        # -------------------------------------------------------------------- #
        # PML options, if using.
        # -------------------------------------------------------------------- #
        if PML_options is not None:
            # Extract the key PML parameters
            c_max = PML_options.c_max
            delta_frac = PML_options.delta_frac
            tol = PML_options.tol
            gamma0 = PML_options.gamma0
            H = PML_options.H
            alpha_fact = PML_options.alpha_fact

            delta = delta_frac*H
            z_PML = H - delta
            sigma0 = (4*c_max/(2*delta))*ln(1/tol)
            alpha = alpha_fact*sigma0

            x = SpatialCoordinate(domain.mesh)
            z = x[len(x)-1]

            sigma_expr = conditional(z <= z_PML,
                                     0.0,
                                     sigma0*sin((pi/2.)*(z-z_PML)/(delta))**2)

            W_DG = FunctionSpace(domain.mesh, "DG", 2)
            self.sigma = self.prescribed_fields("PML", W_DG).interpolate(sigma_expr)
            self.gamma_z = self.prescribed_fields("gamma_z", W_DG).interpolate(Constant(1.0) + gamma0*self.sigma)

            # Define useful decompositions:
            u_vert = domain.k*inner(u, domain.k)
            u_horiz = u - u_vert

            # This is the velocity test function
            w_vert = domain.k*inner(w, domain.k)
            w_horiz = w - w_vert

            # PML q_u test function
            q_u_test_vert = domain.k*inner(q_u_test, domain.k)

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

        # Buoyancy transport
        b_adv = prognostic(advection_form(gamma, b, u), 'b')

        # TODO #651: we should remove this hand-coded linearisation
        # currently REXI can't handle generated transport linearisations
        if self.linearisation_map(b_adv.terms[0]):
            linear_b_adv = linear_advection_form(gamma, b_trial, u_trial, b_bar, u_bar)
            b_adv = linearisation(b_adv, linear_b_adv)

        if compressible:
            # Pressure transport
            p_adv = prognostic(advection_form(phi, p, u), 'p')

            # TODO #651: we should remove this hand-coded linearisation
            # currently REXI can't handle generated transport linearisations
            if self.linearisation_map(p_adv.terms[0]):
                linear_p_adv = linear_advection_form(phi, p_trial, u_trial, p_bar, u_bar)
                p_adv = linearisation(p_adv, linear_p_adv)
            adv_form = subject(u_adv + p_adv + b_adv, self.X)

        else:
            adv_form = subject(u_adv + b_adv, self.X)

        # Add transport of tracers
        if len(active_tracers) > 0:
            adv_form += self.generate_tracer_transport_terms(active_tracers)

        # -------------------------------------------------------------------- #
        # Pressure Gradient Term
        # -------------------------------------------------------------------- #
        if PML_options is not None:
            pressure_gradient_form = pressure_gradient(subject(prognostic(
                -(div(w_horiz) + div(w_vert/self.gamma_z))*p_pert*dx, 'u'), self.X))
        else:
            pressure_gradient_form = pressure_gradient(subject(prognostic(
                -div(w)*p*dx, 'u'), self.X))
        # -------------------------------------------------------------------- #
        # Gravitational Term
        # -------------------------------------------------------------------- #
        if PML_options is not None:
            gravity_form = gravity(subject(prognostic(
                -b_pert*inner(w, domain.k)*dx, 'u'), self.X))
        else:
            gravity_form = gravity(subject(prognostic(
                -b*inner(w, domain.k)*dx, 'u'), self.X))

        # -------------------------------------------------------------------- #
        # Divergence Term
        # -------------------------------------------------------------------- #
        if compressible:
            cs = parameters.cs
            # On assuming ``cs`` as a constant, it is right keep it out of the
            # integration.
            if PML_options is not None:
                divergence_form = divergence(subject(prognostic(cs**2 * (phi * (div(u_horiz) + (1/self.gamma_z)*div(u_vert)) * dx), 'p'), self.X))
            else:
                linear_div_form = divergence(subject(
                    prognostic(cs**2 * (phi * div(u_trial) * dx), 'p'), self.X))
                divergence_form = divergence(linearisation(
                    subject(prognostic(cs**2 * (phi * div(u) * dx), 'p'), self.X),
                    linear_div_form))
        else:
            # This enforces that div(u) = 0
            # The p features here so that the div(u) evaluated in the
            # "forcing" step replaces the whole pressure field, rather than
            # merely providing an increment to it.
            linear_div_form = incompressible(
                subject(prognostic(phi*(p_trial-div(u_trial))*dx, 'p'), self.X))
            divergence_form = incompressible(linearisation(
                subject(prognostic(phi*(p-div(u))*dx, 'p'), self.X),
                linear_div_form))

        residual = (mass_form + adv_form + divergence_form
                    + pressure_gradient_form + gravity_form)

        # -------------------------------------------------------------------- #
        # Extra Terms (Coriolis)
        # -------------------------------------------------------------------- #
        if self.parameters.Omega is not None:
            Omega = as_vector((0, 0, self.parameters.Omega))
            coriolis_form = coriolis(subject(prognostic(
                inner(w, cross(2*Omega, u))*dx, 'u'), self.X))
            residual += coriolis_form

        # -------------------------------------------------------------------- #
        # Option to add a sponge layer
        # -------------------------------------------------------------------- #
        if sponge_options is not None:
            dx_qp = dx(degree=(domain.max_quad_degree))
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

        # -------------------------------------------------------------------- #
        # PML terms. These act on perturbations from the initial condition.
        # -------------------------------------------------------------------- #

        if PML_options is not None:
            # Vertical pressure gradient term
            residual += subject(prognostic(p_pert*div(q_u_test_vert/self.gamma_z)*dx, 'q_u'), self.X)

            # Divergence term for q_p
            residual -= subject(prognostic(cs**2 * q_p_test * (1/self.gamma_z) * div(u_vert) * dx, 'q_p'), self.X)

            # NEW! Mean shift transport for the PML variables.
            # Apply DG upwind for stability
            #q_u_adv = prognostic(advection_form(q_u_test, q_u, u_bar), 'q_u')
            #q_p_adv = prognostic(advection_form(q_p_test, q_p, u_bar), 'q_p')

            #q_adv_form = subject(q_u_adv + q_p_adv, self.X)
            #residual += q_adv_form

            # The PML damping terms
            residual -= subject(prognostic(self.sigma*inner(w, q_u)*dx, 'u'), self.X)
            residual -= subject(prognostic(phi*self.sigma*q_p*dx, 'p'), self.X)
            residual += subject(prognostic((self.sigma+alpha)*inner(q_u_test, q_u)*dx, 'q_u'), self.X) 
            residual += subject(prognostic(q_p_test*(self.sigma+alpha)*q_p*dx, 'q_p'), self.X)


            # Additional transport terms ... see if they improve the gravity wave
            #u_vert_pert = inner(u_pert, domain.k)

            #dpbar_dz = inner(grad(p_bar), domain.k)
            #residual -= subject(prognostic(q_p_test * (u_vert_pert/self.gamma_z) * dpbar_dz * dx, 'q_p'), self.X)

            #dbbar_dz = inner(grad(b_bar), domain.k)
            #residual -= subject(prognostic(q_b_test * (u_vert_pert/self.gamma_z) * dbbar_dz * dx, 'q_b'), self.X)

            #residual -= subject(prognostic(gamma*self.sigma*q_b*dx, 'b'), self.X) 
            #residual += subject(prognostic(q_b_test*(self.sigma+alpha)*q_b*dx, 'q_b'), self.X)

        # -------------------------------------------------------------------- #
        # Linearise equations
        # -------------------------------------------------------------------- #
        # Add linearisations to equations
        self.residual = self.generate_linear_terms(residual, self.linearisation_map)

    def schur_complement_form(self, alpha=0.5, tau_values=None):
        domain = self.domain
        dt = domain.dt
        tau_values = tau_values or {}
        # Set relaxation parameters. If an alternative has not been given, set
        # to semi-implicit off-centering factor
        beta_u = dt*tau_values.get("u", alpha)
        beta_p = dt*tau_values.get("p", alpha)
        beta_b = dt*tau_values.get("b", alpha)
        Vu = domain.spaces("HDiv")
        Vp = domain.spaces("DG")

        # Build the reduced function space for u,p
        M = Vu*Vp
        w, phi = TestFunctions(M)
        u, p = TrialFunctions(M)

        # Get background fields
        bbar = split(self.X_ref)[2]

        # Analytical (approximate) elimination of theta
        k = self.domain.k             # Upward pointing unit vector
        b = -dot(k, u)*dot(k, grad(bbar))*beta_b

        # vertical projection
        def V(u):
            return k*inner(u, k)

        seqn = (
            inner(w, u)*dx
            - beta_u*div(w)*p*dx
            - beta_u*inner(w, k)*b*dx
        )

        if self.compressible:
            cs = self.parameters.cs
            seqn += phi * p * dx + beta_p * phi * cs**2 * div(u) * dx
        else:
            seqn += phi * div(u) * dx

        if hasattr(self, "mu"):
            seqn += dt*self.mu*inner(w, k)*inner(u, k)*dx

        if self.parameters.Omega is not None:
            Omega = as_vector((0, 0, self.parameter.Omega))
            seqn += inner(w, cross(2*Omega, u))*dx

        # Boundary conditions (assumes extruded mesh)
        # BCs are declared for the plain velocity space. As we need them in
        # a mixed problem, we replicate the BCs but for subspace of M
        sbcs = [DirichletBC(M.sub(0), bc.function_arg, bc.sub_domain) for bc in self.bcs['u']]

        return seqn, sbcs


class LinearBoussinesqEquations(BoussinesqEquations):
    """
    Class for the Boussinesq equations, which evolve the velocity
    'u', the pressure 'p' and the buoyancy 'b'. Can be compressible or
    incompressible, depending on the value of the input flag, which defaults
    to compressible.

    The compressible form of the equations is
    ∂u/∂t + (u.∇)u + 2Ω×u + ∇p - b*k = 0,                                     \n
    ∂p/∂t + (u.∇)p + cs**2 ∇.u = 0,                                           \n
    ∂b/∂t + (u.∇)b = 0,                                                       \n
    where k is the vertical unit vector, Ω is the planet's rotation vector
    and cs is the sound speed.

    For the incompressible form of the equations, the pressure features as
    a Lagrange multiplier to enforce incompressibility. The equations are     \n
    ∂u/∂t + (u.∇)u + 2Ω×u + ∇p - b*k = 0,                                     \n
    ∇.u = p,                                                                  \n
    ∂b/∂t + (u.∇)b = 0,                                                       \n
    where k is the vertical unit vector and Ω is the planet's rotation vector.
    """

    def __init__(self, domain, parameters,
                 compressible=True,
                 space_names=None,
                 linearisation_map=all_terms,
                 u_transport_option="vector_invariant_form",
                 no_normal_flow_bc_ids=None,
                 active_tracers=None):
        """
        Args:
            domain (:class:`Domain`): the model's domain object, containing the
                mesh and the compatible function spaces.
            parameters (:class:`Configuration`, optional): an object containing
                the model's physical parameters.
            compressible (bool, optional): flag to indicate whether the
                equations are compressible. Defaults to True
            space_names (dict, optional): a dictionary of strings for names of
                the function spaces to use for the spatial discretisation. The
                keys are the names of the prognostic variables. Defaults to None
                in which case the spaces are taken from the de Rham complex.
            linearisation_map (func, optional): a function specifying which
                terms in the equation set to linearise. If None is specified
                then no terms are linearised. Defaults to the string 'default',
                in which case the linearisation drops terms for any active
                tracers.
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
                in the equations.. Defaults to None.

        Raises:
            NotImplementedError: active tracers are not implemented.
        """

        super().__init__(domain=domain,
                         parameters=parameters,
                         compressible=compressible,
                         space_names=space_names,
                         linearisation_map=linearisation_map,
                         u_transport_option=u_transport_option,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        # Use the underlying routine to do a first linearisation of
        # the equations
        self.linearise_equation_set()

class LinearAcousticBuoyancyEquations(PrognosticEquationSet):
    """
    Class for the Boussinesq equations, which evolve the velocity
    'u', the pressure 'p' and the buoyancy 'b'. Can be compressible or
    incompressible, depending on the value of the input flag, which defaults
    to compressible.

    The compressible form of the equations is
    ∂u/∂t + ∇p - b*k = 0,                                                     \n
    ∂p/∂t + cs**2(∇.u) = 0,                                                    \n
    ∂b/∂t + N^2(k.u) = 0,                                                       \n
    where k is the vertical unit vector, cs is the sound speed,
    and N is the Brunt-Vaisala frequency.
    """

    def __init__(self, domain, parameters,
                 sponge_options=None,
                 PML_options=None,
                 space_names=None,
                 linearisation_map=all_terms,
                 no_normal_flow_bc_ids=None,
                 active_tracers=None):
        """
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
                Defaults to 'vector_invariant_form'.
            no_normal_flow_bc_ids (list, optional): a list of IDs of domain
                boundaries at which no normal flow will be enforced. Defaults to
                None.
            active_tracers (list, optional): a list of `ActiveTracer` objects
                that encode the metadata for any active tracers to be included
                in the equations.. Defaults to None.

        Raises:
            NotImplementedError: active tracers are not implemented.
        """

        field_names = ['u', 'p', 'b']

        if space_names is None:
            space_names = {'u': 'HDiv', 'p': 'L2', 'b': 'theta'}

        if PML_options is not None:
            field_names.extend(['q_u', 'q_p'])
            p_space = space_names['p']
            space_names.update({'q_u': 'HDiv', 'q_p': p_space})

        print(space_names)

        if active_tracers is not None:
            raise NotImplementedError('Tracers not implemented for Boussinesq equations')

        if active_tracers is None:
            active_tracers = []

        super().__init__(field_names, domain, space_names,
                         linearisation_map=linearisation_map,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        self.parameters = parameters
        self.PML_options = PML_options
        self.compressible = True

        if PML_options is not None:
            w, phi, gamma, q_u_test, q_p_test = self.tests[0:5]
            u, p, b, q_u, q_p = split(self.X)[0:5]
            u_trial, p_trial, b_trial, q_u_trial, q_p_trial = split(self.trials)[0:5]
            u_bar, p_bar, b_bar, q_u_bar, q_p_bar = split(self.X_ref)[0:5]
        else:
            w, phi, gamma = self.tests[0:3]
            u, p, b = split(self.X)
            u_trial, p_trial, b_trial = split(self.trials)[0:3]
            u_bar, p_bar, b_bar = split(self.X_ref)[0:3]

        # -------------------------------------------------------------------- #
        # PML options, if using.
        # -------------------------------------------------------------------- #
        if PML_options is not None:
            # Extract the key PML parameters
            c_max = PML_options.c_max
            delta_frac = PML_options.delta_frac
            tol = PML_options.tol
            gamma0 = PML_options.gamma0
            H = PML_options.H
            alpha_fact = PML_options.alpha_fact
            A = PML_options.A

            delta = delta_frac*H
            z_PML = H - delta
            sigma0 = (A*c_max/delta)*ln(1/tol)
            alpha = alpha_fact*sigma0

            x = SpatialCoordinate(domain.mesh)
            z = x[len(x)-1]

            sigma_expr = conditional(z <= z_PML,
                                     0.0,
                                     sigma0*sin((pi/2.)*(z-z_PML)/(delta))**2)

            W_DG = FunctionSpace(domain.mesh, "DG", 2)
            self.sigma = self.prescribed_fields("PML", W_DG).interpolate(sigma_expr)
            self.gamma_z = self.prescribed_fields("gamma_z", W_DG).interpolate(Constant(1.0) + gamma0*self.sigma)

        # -------------------------------------------------------------------- #
        # Time Derivative Terms
        # -------------------------------------------------------------------- #
        mass_form = self.generate_mass_terms()

        # -------------------------------------------------------------------- #
        # PML modified advection
        # -------------------------------------------------------------------- #
        if PML_options is not None:
            # Decompose the velocity
            u_vert = domain.k*inner(u, domain.k)
            u_horiz = u - u_vert

            # This is the velocity test function
            w_vert = domain.k*inner(w, domain.k)
            w_horiz = w - w_vert

            # PML u test function
            q_u_test_vert = domain.k*inner(q_u_test, domain.k)

        # -------------------------------------------------------------------- #
        # Pressure Gradient Term
        # -------------------------------------------------------------------- #
        if PML_options is not None:
            pressure_gradient_form = pressure_gradient(subject(prognostic(
                -(div(w_horiz) + div(w_vert/self.gamma_z))*p*dx, 'u'), self.X))
        else:
            pressure_gradient_form = pressure_gradient(subject(prognostic(
                -div(w)*p*dx, 'u'), self.X))

        # -------------------------------------------------------------------- #
        # Gravitational Term
        # -------------------------------------------------------------------- #
        gravity_form = gravity(subject(prognostic(
                -b*inner(w, domain.k)*dx, 'u'), self.X))

        # -------------------------------------------------------------------- #
        # Divergence Term
        # -------------------------------------------------------------------- #
        cs = parameters.cs
        # On assuming ``cs`` as a constant, it is right keep it out of the
        # integration.
        if PML_options is not None:
            divergence_form = divergence(subject(prognostic(cs**2 * (phi * (div(u_horiz) + (1/self.gamma_z)*div(u_vert)) * dx), 'p'), self.X))
        else:
            divergence_form = divergence(subject(prognostic(cs**2 * (phi * div(u) * dx), 'p'), self.X))

        # -------------------------------------------------------------------- #
        # Buoyancy term
        # -------------------------------------------------------------------- #

        N = parameters.N
        buoyancy_form = gravity(subject(prognostic(N**2 * gamma * inner(u, domain.k) * dx, 'b'), self.X))


        residual = (mass_form + divergence_form + gravity_form
                    + pressure_gradient_form + buoyancy_form)
        
        # -------------------------------------------------------------------- #
        # Option to add a sponge layer
        # -------------------------------------------------------------------- #
        if sponge_options is not None:
            dx_qp = dx(degree=(domain.max_quad_degree))
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

        # -------------------------------------------------------------------- #
        # PML terms
        # -------------------------------------------------------------------- #

        if PML_options is not None:

            # Vertical pressure gradient term
            residual += subject(prognostic(
            p*div(q_u_test_vert/self.gamma_z)*dx, 'q_u'), self.X)

            # Divergence term
            residual -= divergence(
                subject(prognostic(cs**2 * (q_p_test * (1/self.gamma_z)*div(u_vert) * dx), 'q_p'), self.X))
            
            # The PML damping terms
            residual -= subject(prognostic(self.sigma*inner(w, q_u)*dx, 'u'), self.X)
            #residual -= subject(prognostic(inner(w, self.sigma*q_u)*dx, 'u'), self.X)
            residual -= subject(prognostic(self.sigma*phi*q_p*dx, 'p'), self.X)
            residual += subject(prognostic((self.sigma+alpha)*inner(q_u_test, q_u)*dx, 'q_u'), self.X) 
            #residual += subject(prognostic(inner(q_u_test, (self.sigma+alpha)*q_u)*dx, 'q_u'), self.X) 
            residual += subject(prognostic((self.sigma+alpha)*q_p_test*q_p*dx, 'q_p'), self.X)

        self.residual = residual
