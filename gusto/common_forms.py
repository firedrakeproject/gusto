"""
Provides some basic forms for discretising various common terms in equations for
geophysical fluid dynamics."""

from firedrake import (dx, dot, grad, div, inner, outer, cross, curl, split,
                       TestFunction, TestFunctions, TrialFunctions)
from firedrake.fml import subject, drop
from gusto.configuration import TransportEquationType
from gusto.labels import (transport, transporting_velocity, diffusion,
                          prognostic, linearisation)

__all__ = ["advection_form", "advection_form_1d", "continuity_form",
           "continuity_form_1d", "vector_invariant_form",
           "kinetic_energy_form", "advection_equation_circulation_form",
           "diffusion_form", "diffusion_form_1d",
           "linear_advection_form", "linear_continuity_form",
           "split_continuity_form", "tracer_conservative_form"]


def advection_form(test, q, ubar):
    u"""
    The form corresponding to the advective transport operator.

    This describes (u.∇)q, for transporting velocity u and transported q.

    Args:
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the variable to be transported.
        ubar (:class:`ufl.Expr`): the transporting velocity.

    Returns:
        class:`LabelledForm`: a labelled transport form.
    """

    L = inner(test, dot(ubar, grad(q)))*dx
    form = transporting_velocity(L, ubar)

    return transport(form, TransportEquationType.advective)


def advection_form_1d(test, q, ubar):
    u"""
    The form corresponding to the advective transport operator.

    This describes (u.∇)q, for transporting velocity u and transported q.

    Args:
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the variable to be transported.
        ubar (:class:`ufl.Expr`): the transporting velocity.

    Returns:
        class:`LabelledForm`: a labelled transport form.
    """

    L = test * ubar * q.dx(0)*dx
    form = transporting_velocity(L, ubar)

    return transport(form, TransportEquationType.advective)


def linear_advection_form(test, qbar, ubar):
    """
    The form corresponding to the linearised advective transport operator.

    Args:
        test (:class:`TestFunction`): the test function.
        qbar (:class:`ufl.Expr`): the variable to be transported.
        ubar (:class:`ufl.Expr`): the transporting velocity.

    Returns:
        :class:`LabelledForm`: a labelled transport form.
    """

    L = test*dot(ubar, grad(qbar))*dx
    form = transporting_velocity(L, ubar)

    return transport(form, TransportEquationType.advective)


def continuity_form(test, q, ubar):
    u"""
    The form corresponding to the continuity transport operator.

    This describes ∇.(u*q), for transporting velocity u and transported q.

    Args:
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the variable to be transported.
        ubar (:class:`ufl.Expr`): the transporting velocity.

    Returns:
        class:`LabelledForm`: a labelled transport form.
    """

    L = inner(test, div(outer(q, ubar)))*dx
    form = transporting_velocity(L, ubar)

    return transport(form, TransportEquationType.conservative)


def continuity_form_1d(test, q, ubar):
    u"""
    The form corresponding to the continuity transport operator.

    This describes ∇.(u*q), for transporting velocity u and transported q.

    Args:
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the variable to be transported.
        ubar (:class:`ufl.Expr`): the transporting velocity.

    Returns:
        class:`LabelledForm`: a labelled transport form.
    """

    L = test * (q * ubar).dx(0)*dx
    form = transporting_velocity(L, ubar)

    return transport(form, TransportEquationType.conservative)


def linear_continuity_form(test, qbar, ubar):
    """
    The form corresponding to the linearised continuity transport operator.

    Args:
        test (:class:`TestFunction`): the test function.
        qbar (:class:`ufl.Expr`): the variable to be transported.
        ubar (:class:`ufl.Expr`): the transporting velocity.

    Returns:
        :class:`LabelledForm`: a labelled transport form.
    """

    L = qbar*test*div(ubar)*dx
    form = transporting_velocity(L, ubar)

    return transport(form, TransportEquationType.conservative)


def vector_invariant_form(domain, test, q, ubar):
    u"""
    The form corresponding to the vector invariant transport operator.

    The self-transporting transport operator for a vector-valued field u can be
    written as circulation and kinetic energy terms:
    (u.∇)u = (∇×u)×u + (1/2)∇u^2

    When the transporting field u and transported field q are similar, we write
    this as:
    (u.∇)q = (∇×q)×u + (1/2)∇(u.q)

    Args:
        domain (:class:`Domain`): the model's domain object, containing the
            mesh and the compatible function spaces.
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the variable to be transported.
        ubar (:class:`ufl.Expr`): the transporting velocity.

    Returns:
        class:`LabelledForm`: a labelled transport form.
    """

    L = advection_equation_circulation_form(domain, test, q, ubar).terms[0].form

    # Add K.E. term
    L -= 0.5*div(test)*inner(q, ubar)*dx
    form = transporting_velocity(L, ubar)

    return transport(form, TransportEquationType.vector_invariant)


def kinetic_energy_form(test, q, ubar):
    u"""
    The form corresponding to the kinetic energy term.

    Writing the kinetic energy term as (1/2)∇u^2, if the transported variable
    q is similar to the transporting variable u then this can be written as:
    (1/2)∇(u.q).

    Args:
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the variable to be transported.
        ubar (:class:`ufl.Expr`): the transporting velocity.

    Returns:
        class:`ufl.Form`: the kinetic energy form.
    """

    L = -0.5*div(test)*inner(q, ubar)*dx

    return L


def advection_equation_circulation_form(domain, test, q, ubar):
    u"""
    The circulation term in the transport of a vector-valued field.

    The self-transporting transport operator for a vector-valued field u can be
    written as circulation and kinetic energy terms:
    (u.∇)u = (∇×u)×u + (1/2)∇u^2

    When the transporting field u and transported field q are similar, we write
    this as:
    (u.∇)q = (∇×q)×u + (1/2)∇(u.q)

    The form returned by this function corresponds to the (∇×q)×u circulation
    term.

    Args:
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the variable to be transported.
        ubar (:class:`ufl.Expr`): the transporting velocity.

    Returns:
        class:`LabelledForm`: a labelled transport form.
    """

    if domain.mesh.topological_dimension() == 3:
        L = inner(test, cross(curl(q), ubar))*dx

    else:
        perp = domain.perp
        L = inner(test, div(perp(q))*perp(ubar))*dx

    form = transporting_velocity(L, ubar)

    return transport(form, TransportEquationType.circulation)


def diffusion_form(test, q, kappa):
    u"""
    The diffusion form, -∇.(κ∇q) for diffusivity κ and variable q.

    Args:
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the variable to be diffused.
        kappa: (:class:`ufl.Expr`): the diffusivity value.
    """

    form = -inner(test, div(kappa*grad(q)))*dx

    return diffusion(form)


def diffusion_form_1d(test, q, kappa):
    u"""
    The diffusion form, -∇.(κ∇q) for diffusivity κ and variable q.

    Args:
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the variable to be diffused.
        kappa: (:class:`ufl.Expr`): the diffusivity value.
    """

    form = -test * (kappa*q.dx(0)).dx(0)*dx

    return diffusion(form)


def split_continuity_form(equation):
    u"""
    Loops through terms in a given equation, and splits all continuity terms
    into advective and divergence terms.

    This describes splitting ∇.(u*q) terms into u.∇q and q(∇.u),
    for transporting velocity u and transported q.

    Args:
        equation (:class:`PrognosticEquation`): the model's equation.

    Returns:
        :class:`PrognosticEquation`: the model's equation.
    """

    for t in equation.residual:
        if (t.get(transport) == TransportEquationType.conservative):
            # Get fields and test functions
            subj = t.get(subject)
            prognostic_field_name = t.get(prognostic)
            if hasattr(equation, "field_names"):
                idx = equation.field_names.index(prognostic_field_name)
                W = equation.function_space
                test = TestFunctions(W)[idx]
                q = split(subj)[idx]
            else:
                W = equation.function_space
                test = TestFunction(W)
                q = subj
            # u is either a prognostic or prescribed field
            if (hasattr(equation, "field_names")
               and 'u' in equation.field_names):
                u_idx = equation.field_names.index('u')
                uadv = split(equation.X)[u_idx]
            elif 'u' in equation.prescribed_fields._field_names:
                uadv = equation.prescribed_fields('u')
            else:
                raise ValueError('Cannot get velocity field')

            # Create new advective and divergence terms
            adv_term = prognostic(advection_form(test, q, uadv), prognostic_field_name)
            div_term = prognostic(test*q*div(uadv)*dx, prognostic_field_name)

            # Add linearisations of new terms if required
            if (t.has_label(linearisation)):
                u_trial = TrialFunctions(W)[u_idx]
                qbar = split(equation.X_ref)[idx]
                # Add linearisation to adv_term
                linear_adv_term = linear_advection_form(test, qbar, u_trial)
                adv_term = linearisation(adv_term, linear_adv_term)
                # Add linearisation to div_term
                linear_div_term = transporting_velocity(qbar*test*div(u_trial)*dx, u_trial)
                div_term = linearisation(div_term, linear_div_term)

            # Add new terms onto residual
            equation.residual += subject(adv_term + div_term, subj)
            # Drop old term
            equation.residual = equation.residual.label_map(
                lambda t: t.get(transport) == TransportEquationType.conservative,
                map_if_true=drop)

    return equation


def tracer_conservative_form(test, q, rho, ubar):
    u"""
    The form corresponding to the continuity transport operator.

    This describes ∇.(u*q*rho) for transporting velocity u and a
    transported tracer (mixing ratio), q, with an associated density, rho.

    Args:
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the tracer to be transported.
        rho (:class:`ufl.Expr`): the reference density that will
        mulitply with q before taking the divergence.
        ubar (:class:`ufl.Expr`): the transporting velocity.

    Returns:
        class:`LabelledForm`: a labelled transport form.
    """

    q_rho = q*rho
    L = inner(test, div(outer(q_rho, ubar)))*dx
    form = transporting_velocity(L, ubar)

    return transport(form, TransportEquationType.tracer_conservative)
