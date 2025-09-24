"""Defines the Boussinesq equations."""

from firedrake import (
    inner, dx, div, cross, split, as_vector, Constant, SpatialCoordinate,
    conditional, FunctionSpace, ln
)

from firedrake.fml import subject, all_terms
from gusto.core.labels import (
    prognostic, linearisation,
    pressure_gradient, coriolis, divergence, gravity, incompressible
)
from gusto.equations.common_forms import (
    advection_form, vector_invariant_form,
    kinetic_energy_form, advection_equation_circulation_form,
    linear_advection_form, linear_circulation_form,
    linear_vector_invariant_form
)
from gusto.equations.prognostic_equations import PrognosticEquationSet

__all__ = ["BoussinesqEquations", "LinearBoussinesqEquations"]


class BoussinesqEquations(PrognosticEquationSet):
    """
    Class for the Boussinesq equations, which evolve the velocity
    'u', the pressure 'p' and the buoyancy 'b'. Can be compressible or
    incompressible, depending on the value of the input flag, which defaults
    to compressible.

    The compressible form of the equations is
    ∂u/∂t + (u.∇)u + 2Ω×u + ∇p + b*k = 0,                                     \n
    ∂p/∂t + cs**2 ∇.u = p,                                                    \n
    ∂b/∂t + (u.∇)b = 0,                                                       \n
    where k is the vertical unit vector, Ω is the planet's rotation vector
    and cs is the sound speed.

    For the incompressible form of the equations, the pressure features as
    a Lagrange multiplier to enforce incompressibility. The equations are     \n
    ∂u/∂t + (u.∇)u + 2Ω×u + ∇p + b*k = 0,                                     \n
    ∇.u = p,                                                                  \n
    ∂b/∂t + (u.∇)b = 0,                                                       \n
    where k is the vertical unit vector and Ω is the planet's rotation vector.
    """

    def __init__(self, domain, parameters,
                 compressible=True,
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
            # Define PML variables in the same space as the original ones
            field_names.extend(['q_u', 'q_p', 'q_b'])
            space_names.update({'q_u': 'HDiv', 'q_p': 'L2', 'q_b': 'theta'})

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
            w, phi, gamma, q_u_test, q_p_test, q_b_test = self.tests[0:6]
            u, p, b, q_u, q_p, q_b = split(self.X)[0:6]
            u_trial, p_trial, b_trial, q_u_trial, q_p_trial, q_b_trial = split(self.trials)[0:6]
            u_bar, p_bar, b_bar, q_u_bar, q_p_bar, q_b_bar = split(self.X_ref)[0:6]
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
            H_PML = H - delta
            sigma0 = (4*c_max/(2*delta))*ln(1/tol)
            alpha = Constant(alpha_fact*sigma0)

            x = SpatialCoordinate(domain.mesh)
            z = x[len(x)-1]

            print(len(x))
            print(H_PML)
            print(Constant(sigma0))

            #import sys; sys.exit()

            sigma_expr = conditional(z <= H_PML,
                                     0.0,
                                     sigma0*((z-H_PML)/delta)**3)

            W_DG = FunctionSpace(domain.mesh, "DG", 2)
            self.sigma = self.prescribed_fields("PML", W_DG).interpolate(sigma_expr)

            self.gamma_z = self.prescribed_fields("gamma_z", W_DG).interpolate(Constant(1.0))# + gamma0*self.sigma)

            # Also need to define equivalents for the RT space.

        # -------------------------------------------------------------------- #
        # Time Derivative Terms
        # -------------------------------------------------------------------- #
        mass_form = self.generate_mass_terms()

        # -------------------------------------------------------------------- #
        # Transport Terms
        # -------------------------------------------------------------------- #
        if PML_options is not None:
            # Decompose the velocity
            u_vert = domain.k*inner(u, domain.k)
            u_horiz = u - u_vert
            u_vert_scaled = (Constant(1.0)/self.gamma_z)*u_vert

            # This is the test function
            w_vert = domain.k*inner(w, domain.k)
            w_horiz = w - w_vert

            # PML u test function
            #q_u_test_vert = domain.k*inner(q_u_test, domain.k)
            #q_u_test_horiz = q_u_test - q_u_test_vert
            q_u_test_vert = as_vector([Constant(0.0), q_u_test[1]])

            q_u_trial_vert = as_vector([Constant(0.0), q_u_trial[1]])

            # For the PML, scale the vertical advecting velocity:
            #scale_vect = as_vector([Constant(1.0), self.gamma_z])
            #u_advect = as_vector([u_horiz, u_vert_scaled])
            u_advect = u
            u_w = as_vector([Constant(0.0), u[1]])
            #u_w = u_vert
        else:
            u_advect = u


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
            u_adv = prognostic(advection_form(w, u, u_advect), 'u')

        else:
            raise ValueError("Invalid u_transport_option: %s" % self.u_transport_option)

        # Buoyancy transport
        b_adv = prognostic(advection_form(gamma, b, u_advect), 'b')

        # TODO #651: we should remove this hand-coded linearisation
        # currently REXI can't handle generated transport linearisations
        if self.linearisation_map(b_adv.terms[0]):
            linear_b_adv = linear_advection_form(gamma, b_trial, u_trial, b_bar, u_bar)
            b_adv = linearisation(b_adv, linear_b_adv)

        if compressible:
            # Pressure transport
            p_adv = prognostic(advection_form(phi, p, u_advect), 'p')

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
        if compressible:
            cs = parameters.cs
            # On assuming ``cs`` as a constant, it is right keep it out of the
            # integration.
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
        # PML terms
        # -------------------------------------------------------------------- #

        if PML_options is not None:
            # Add vertical advection PML terms for the PML variables
            # These are all 1d transport in the vertical, and 
            # have a minus sign in front of them.

            # Advective forms for q_u, q_p, q_b
            q_u_adv = subject(prognostic(advection_form(q_u_test, u, u_w), 'q_u'), self.X)

            q_p_adv = subject(prognostic(advection_form(q_p_test, p, u_w), 'q_p'), self.X)
            if self.linearisation_map(q_p_adv.terms[0]):
                print('yup yup, q_p linearisation')
                linear_q_p_adv = linear_advection_form(q_p_test, p_trial, u_trial, p_bar, u_bar)
                q_p_adv = linearisation(q_p_adv, linear_q_p_adv)
  
            q_b_adv = subject(prognostic(advection_form(q_b_test, b, u_w), 'q_b'), self.X)
            if self.linearisation_map(q_b_adv.terms[0]):
                print('yup yup, q_b linearisation')
                linear_q_b_adv = linear_advection_form(q_b_test, b_trial, u_trial, b_bar, u_bar)
                q_b_adv = linearisation(q_b_adv, linear_q_b_adv)

            #residual -= subject(q_u_adv + q_p_adv + q_b_adv, self.X)

            # Vertical pressure gradient term
            residual -= pressure_gradient(subject(prognostic(
            -div(q_u_test_vert)*p*dx, 'q_u'), self.X))

            

            # Divergence term
            residual -= divergence(
                subject(prognostic(cs**2 * (q_p_test * div(u_w) * dx), 'p'), self.X))
            
            linear_q_div_form = divergence(subject(
                prognostic(cs**2 * (q_p_test * div(q_u_trial_vert) * dx), 'p'), self.X))
            q_div_form = divergence(linearisation(
                subject(prognostic(cs**2 * (q_p_test * div(u_w) * dx), 'p'), self.X),
                linear_div_form))
            #residual -= q_div_form
            
            # The PML damping terms
            residual -= subject(prognostic(self.sigma*inner(w, q_u)*dx, 'u'), self.X)
            residual -= subject(prognostic(phi*self.sigma*q_p*dx, 'p'), self.X)
            residual -= subject(prognostic(gamma*self.sigma*q_b*dx, 'b'), self.X) 
            residual += subject(prognostic((self.sigma+alpha)*inner(q_u_test, q_u)*dx, 'q_u'), self.X) 
            residual += subject(prognostic(q_p_test*(self.sigma+alpha)*q_p*dx, 'q_p'), self.X)
            residual += subject(prognostic(q_b_test*(self.sigma+alpha)*q_b*dx, 'q_b'), self.X)

            # Provide linearisations of these:



        # -------------------------------------------------------------------- #
        # Linearise equations
        # -------------------------------------------------------------------- #
        # Add linearisations to equations
        self.residual = self.generate_linear_terms(residual, self.linearisation_map)


class LinearBoussinesqEquations(BoussinesqEquations):
    """
    Class for the Boussinesq equations, which evolve the velocity
    'u', the pressure 'p' and the buoyancy 'b'. Can be compressible or
    incompressible, depending on the value of the input flag, which defaults
    to compressible.

    The compressible form of the equations is
    ∂u/∂t + (u.∇)u + 2Ω×u + ∇p + b*k = 0,                                     \n
    ∂p/∂t + cs**2 ∇.u = p,                                                    \n
    ∂b/∂t + (u.∇)b = 0,                                                       \n
    where k is the vertical unit vector, Ω is the planet's rotation vector
    and cs is the sound speed.

    For the incompressible form of the equations, the pressure features as
    a Lagrange multiplier to enforce incompressibility. The equations are     \n
    ∂u/∂t + (u.∇)u + 2Ω×u + ∇p + b*k = 0,                                     \n
    ∇.u = p,                                                                  \n
    ∂b/∂t + (u.∇)b = 0,                                                       \n
    where k is the vertical unit vector and Ω is the planet's rotation vector.
    """

    def __init__(self, domain, parameters,
                 compressible=True,
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
                         PML_options=PML_options,
                         space_names=space_names,
                         linearisation_map=linearisation_map,
                         u_transport_option=u_transport_option,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        # Use the underlying routine to do a first linearisation of
        # the equations
        self.linearise_equation_set()
