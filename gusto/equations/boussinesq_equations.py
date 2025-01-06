"""Defines the Boussinesq equations."""

from firedrake import inner, dx, div, cross, split, as_vector
from firedrake.fml import subject
from gusto.core.labels import (
    time_derivative, transport, prognostic, linearisation,
    pressure_gradient, coriolis, divergence, gravity, incompressible
)
from gusto.core.configuration import convert_parameters_to_real_space
from gusto.equations.common_forms import (
    advection_form, vector_invariant_form,
    kinetic_energy_form, advection_equation_circulation_form,
    linear_advection_form
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
                 space_names=None,
                 linearisation_map='default',
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
                in which case the linearisation includes time derivatives and
                scalar transport terms.
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

        if active_tracers is not None:
            raise NotImplementedError('Tracers not implemented for Boussinesq equations')

        if active_tracers is None:
            active_tracers = []

        if linearisation_map == 'default':
            # Default linearisation is time derivatives, scalar transport,
            # pressure gradient, gravity and divergence terms
            # Don't include active tracers
            linearisation_map = lambda t: \
                t.get(prognostic) in ['u', 'p', 'b'] \
                and (any(t.has_label(time_derivative, pressure_gradient,
                                     divergence, gravity))
                     or (t.get(prognostic) not in ['u', 'p'] and t.has_label(transport)))

        super().__init__(field_names, domain, space_names,
                         linearisation_map=linearisation_map,
                         no_normal_flow_bc_ids=no_normal_flow_bc_ids,
                         active_tracers=active_tracers)

        self.parameters = parameters
        # This function converts the parameters to real space.
        # This is a preventive way to avoid adjoint issues when the parameters
        # attribute are the control in the sensitivity computations.
        convert_parameters_to_real_space(parameters, domain.mesh)
        self.compressible = compressible

        w, phi, gamma = self.tests[0:3]
        u, p, b = split(self.X)
        u_trial, p_trial, _ = split(self.trials)
        _, p_bar, b_bar = split(self.X_ref)

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

        # Buoyancy transport
        b_adv = prognostic(advection_form(gamma, b, u), 'b')
        if self.linearisation_map(b_adv.terms[0]):
            linear_b_adv = linear_advection_form(gamma, b_bar, u_trial)
            b_adv = linearisation(b_adv, linear_b_adv)

        if compressible:
            # Pressure transport
            p_adv = prognostic(advection_form(phi, p, u), 'p')
            if self.linearisation_map(p_adv.terms[0]):
                linear_p_adv = linear_advection_form(phi, p_bar, u_trial)
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
            # TODO: add linearisation
            Omega = as_vector((0, 0, self.parameters.Omega))
            coriolis_form = coriolis(subject(prognostic(
                inner(w, cross(2*Omega, u))*dx, 'u'), self.X))
            residual += coriolis_form
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
                 space_names=None,
                 linearisation_map='default',
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
                in which case the linearisation includes time derivatives and
                scalar transport terms.
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

        if linearisation_map == 'default':
            # Default linearisation is time derivatives, pressure gradient,
            # Coriolis and transport term from depth equation
            linearisation_map = lambda t: \
                (any(t.has_label(time_derivative, pressure_gradient, coriolis,
                                 gravity, divergence, incompressible))
                 or (t.get(prognostic) in ['p', 'b'] and t.has_label(transport)))
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
