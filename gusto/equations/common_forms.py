"""
Provides some basic forms for discretising various common terms in equations for
geophysical fluid dynamics."""

from firedrake import (dx, dot, grad, div, inner, outer, cross, curl, split,
                       TestFunction, TestFunctions, TrialFunctions)
from firedrake.fml import subject, drop
from gusto.core.configuration import TransportEquationType
from gusto.core.labels import (transport, transporting_velocity, diffusion,
                               prognostic, linearisation, horizontal_transport,
                               vertical_transport)

__all__ = ["advection_form", "advection_form_1d", "continuity_form",
           "continuity_form_1d", "vector_invariant_form",
           "linear_vector_invariant_form",
           "kinetic_energy_form", "advection_equation_circulation_form",
           "linear_circulation_form",
           "diffusion_form", "diffusion_form_1d",
           "linear_advection_form", "linear_continuity_form",
           "split_continuity_form", "tracer_conservative_form", "split_hv_advective_form",
           "replace_linear_vertical_adv_form"]


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


def linear_advection_form(test, q, u, qbar, ubar):
    """
    The form corresponding to the linearised advective transport operator.

    Args:
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the perturbation variable to be transported.
        u (:class:`ufl.Expr`): the perturbation transporting velocity.
        qbar (:class:`ufl.Expr`): the mean variable to be transported.
        ubar (:class:`ufl.Expr`): the mean transporting velocity.

    Returns:
        :class:`LabelledForm`: a labelled transport form.
    """

    form = (
        transporting_velocity(test*dot(ubar, grad(q))*dx, ubar)
        + transporting_velocity(test*dot(u, grad(qbar))*dx, u)
    )

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


def linear_continuity_form(test, q, u, qbar, ubar):
    """
    The form corresponding to the linearised continuity transport operator.

    Args:
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the perturbation variable to be transported.
        u (:class:`ufl.Expr`): the perturbation transporting velocity.
        qbar (:class:`ufl.Expr`): the mean variable to be transported.
        ubar (:class:`ufl.Expr`): the mean transporting velocity.

    Returns:
        :class:`LabelledForm`: a labelled transport form.
    """

    form = (
        transporting_velocity(test*div(q*ubar)*dx, ubar)
        + transporting_velocity(test*div(qbar*u)*dx, u)
    )

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


def linear_vector_invariant_form(domain, test, q, ubar):
    u"""
    The linear form corresponding to the vector invariant transport operator.

    The vector invariant transport operator is: (∇×q)×u + (1/2)∇(u.q)
    and its linearised form is:
    (∇×q')×u_bar + (∇×q_bar)×u' + (1/2)∇(u_bar.q') + (1/2)∇(u'.q_bar)

    Args:
        domain (:class:`Domain`): the model's domain object, containing the
            mesh and the compatible function spaces.
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the variable to be transported.
        ubar (:class:`ufl.Expr`): the transporting velocity.

    Returns:
        class:`LabelledForm`: a labelled transport form.
    """

    L = linear_circulation_form(domain, test, q, ubar).form

    # Add K.E. term
    L -= div(test)*inner(q, ubar)*dx
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
        domain (:class:`Domain`): the model's domain object.
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


def linear_circulation_form(domain, test, q, ubar):
    """
    The linear circulation term in the transport of a vector-valued field.

    Args:
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the variable to be transported.
        ubar (:class:`ufl.Expr`): the transporting velocity.

    Returns:
        class:`LabelledForm`: a labelled transport form.
    """

    form = (
        advection_equation_circulation_form(domain, test, q, ubar)
        + advection_equation_circulation_form(domain, test, ubar, q)
    )
    return form


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


def split_advection_form(test, q, ubar, ubar_full):
    u"""
    The form corresponding to the advective transport operator in either horzontal
    or vertical directions (dependent on ubar).

    This describes either u_h.(∇)q or w dq/dz, for transporting velocity u and transported q.

    Args:
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the variable to be transported.
        ubar (:class:`ufl.Expr`): the transporting velocity in a subset of dimensions.
        ubar_full (:class:`ufl.Expr`): the transporting velocity in all dimensions.

    Returns:
        class:`LabelledForm`: a labelled transport form.
    """

    L = inner(test, dot(ubar, grad(q)))*dx
    form = transporting_velocity(L, ubar_full)

    return transport(form, TransportEquationType.advective)


def split_linear_advection_form(test, qbar, ubar, ubar_full):
    """
    The form corresponding to the linearised advective transport operator in
    either horzontal or vertical directions (dependent on ubar).

    Args:
        test (:class:`TestFunction`): the test function.
        qbar (:class:`ufl.Expr`): the variable to be transported.
        ubar (:class:`ufl.Expr`): the transporting velocity in a subset of dimensions.
        ubar_full (:class:`ufl.Expr`): the transporting velocity in all dimensions.

    Returns:
        :class:`LabelledForm`: a labelled transport form.
    """

    L = test*dot(ubar, grad(qbar))*dx
    form = transporting_velocity(L, ubar_full)

    return transport(form, TransportEquationType.advective)


def split_hv_advective_form(equation, field_name):
    u"""
    Splits advective term into horizontal and vertical terms.
    This describes splitting u.∇(q) terms into u_h.(∇)q and w dq/dz,
    for transporting velocity u and transported q.
    Args:
        equation (:class:`PrognosticEquation`): the model's equation.
    Returns:
        :class:`PrognosticEquation`: the model's equation.
    """
    k = equation.domain.k   # vertical unit vector
    for t in equation.residual:
        if (t.get(transport) == TransportEquationType.advective and t.get(prognostic) == field_name):
            # Get fields and test functions
            subj = t.get(subject)

            # u is either a prognostic or prescribed field
            if (hasattr(equation, "field_names")
               and 'u' in equation.field_names):
                idx = equation.field_names.index(field_name)
                W = equation.function_space
                test = TestFunctions(W)[idx]
                q = split(subj)[idx]
                u_idx = equation.field_names.index('u')
                uadv = split(equation.X)[u_idx]
            elif 'u' in equation.prescribed_fields._field_names:
                uadv = equation.prescribed_fields('u')
                q = subj
                W = equation.function_space
                test = TestFunction(W)
            else:
                raise ValueError('Cannot get velocity field')

            # Create new advective and divergence terms
            u_vertical = k*inner(uadv, k)
            u_horizontal = uadv - u_vertical
            vertical_adv_term = prognostic(
                vertical_transport(
                    split_advection_form(test, q, u_vertical, uadv)
                ),
                field_name
            )
            horizontal_adv_term = prognostic(
                horizontal_transport(
                    split_advection_form(test, q, u_horizontal, uadv)
                ),
                field_name
            )

            # Add linearisations of new terms if required
            if (t.has_label(linearisation)):
                u_trial = TrialFunctions(W)[u_idx]
                u_trial_vert = k*inner(u_trial, k)
                u_trial_horiz = u_trial - u_trial_vert
                qbar = split(equation.X_ref)[idx]
                # Add linearisations
                linear_hori_term = horizontal_transport(
                    split_linear_advection_form(test, qbar, u_trial_horiz, u_trial)
                )
                adv_horiz_term = linearisation(horizontal_adv_term, linear_hori_term)

                linear_vert_term = vertical_transport(
                    split_linear_advection_form(test, qbar, u_trial_vert, u_trial)
                )
                adv_vert_term = linearisation(vertical_adv_term, linear_vert_term)
            else:
                adv_vert_term = vertical_adv_term
                adv_horiz_term = horizontal_adv_term
            # Drop old term
            equation.residual = equation.residual.label_map(
                lambda t: t.get(transport) == TransportEquationType.advective and t.get(prognostic) == field_name,
                map_if_true=drop)

            # Add new terms onto residual
            equation.residual += subject(adv_horiz_term, subj) + subject(adv_vert_term, subj)

    return equation

def replace_linear_vertical_adv_form(equation, field_name):
    u"""
    Replaces 3D linear advective term with vertical term.
    This describes replacing u.∇(qbar) terms with  dqbar/dz,
    for transporting velocity u and qbar.
    Args:
        equation (:class:`PrognosticEquation`): the model's equation.
    Returns:
        :class:`PrognosticEquation`: the model's equation.
    """
    k = equation.domain.k   # vertical unit vector
    for t in equation.residual:
        if (t.get(transport) == TransportEquationType.advective and t.get(prognostic) == field_name and t.has_label(linearisation)):
            # Get fields and test functions
            subj = t.get(subject)

            # u is either a prognostic or prescribed field
            if (hasattr(equation, "field_names")
               and 'u' in equation.field_names):
                idx = equation.field_names.index(field_name)
                W = equation.function_space
                test = TestFunctions(W)[idx]
                q = split(subj)[idx]
                u_idx = equation.field_names.index('u')
                uadv = split(equation.X)[u_idx]
            elif 'u' in equation.prescribed_fields._field_names:
                uadv = equation.prescribed_fields('u')
                q = subj
                W = equation.function_space
                test = TestFunction(W)
            else:
                raise ValueError('Cannot get velocity field')

            adv_term = equation.residual.label_map(
                lambda t: t.get(transport) == TransportEquationType.advective and t.get(prognostic) == field_name,
                map_if_false=drop).terms[0].form
            # Add linearisations of new terms if required
            u_trial = TrialFunctions(W)[u_idx]
            u_trial_vert = k*inner(u_trial, k)
            qbar = split(equation.X_ref)[idx]
            # Add linearisations
            linear_vert_term = vertical_transport(
                split_linear_advection_form(test, qbar, u_trial_vert, u_trial)
            )
            new_adv_term = linearisation(adv_term, linear_vert_term)
            # Drop old term
            equation.residual = equation.residual.label_map(
                lambda t: t.get(transport) == TransportEquationType.advective and t.get(prognostic) == field_name,
                map_if_true=drop)

            # Add new terms onto residual
            equation.residual += subject(new_adv_term, subj)

    return equation
