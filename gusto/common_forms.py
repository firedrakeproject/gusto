"""
Provides some basic forms for discretising various common terms in equations for
geophysical fluid dynamics."""

from firedrake import dx, dot, grad, div, inner, outer, cross, curl
from gusto.configuration import TransportEquationType
from gusto.labels import transport, transporting_velocity, diffusion

__all__ = ["advection_form", "continuity_form", "vector_invariant_form",
           "kinetic_energy_form", "advection_equation_circulation_form",
           "diffusion_form"]


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

    Raises:
        NotImplementedError: the specified integration by parts is not 'once'.

    Returns:
        class:`LabelledForm`: a labelled transport form.
    """

    if domain.mesh.topological_dimension() == 3:
        L = inner(test, cross(curl(q), ubar))*dx

    else:
        perp = domain.perp
        L = inner(test, div(perp(q))*perp(ubar))*dx

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
        class:`LabelledForm`: a labelled transport form.
    """

    L = 0.5*div(test)*inner(q, ubar)*dx

    form = transporting_velocity(L, ubar)

    return transport(form, TransportEquationType.vector_invariant)


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
    term. An an upwind discretisation is used when integrating by parts.

    Args:
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the variable to be transported.
        ubar (:class:`ufl.Expr`): the transporting velocity.

    Raises:
        NotImplementedError: the specified integration by parts is not 'once'.

    Returns:
        class:`LabelledForm`: a labelled transport form.
    """

    form = (
        vector_invariant_form(domain, test, q, ubar)
        - kinetic_energy_form(test, q, ubar)
    )

    return form


def diffusion_form(test, q, kappa):
    u"""
    The diffusion form, ∇.(κ∇q) for diffusivity κ and variable q.

    Args:
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the variable to be diffused.
        kappa: (:class:`ufl.Expr`): the diffusivity value.
    """

    form = inner(test, div(kappa*grad(q)))*dx

    return diffusion(form)
