"""Provides forms for different transport operators."""

from firedrake import (Function, FacetNormal,
                       dx, dot, grad, div, jump, avg, dS, dS_v, dS_h, inner,
                       ds_v, ds_t, ds_b,
                       outer, sign, cross, CellNormal,
                       curl)
from gusto.configuration import IntegrateByParts, TransportEquationType
from gusto.labels import transport, transporting_velocity, ibp_label


__all__ = ["advection_form", "continuity_form", "vector_invariant_form",
           "vector_manifold_advection_form", "kinetic_energy_form",
           "advection_equation_circulation_form", "linear_continuity_form"]


def linear_advection_form(domain, test, qbar):
    """
    The form corresponding to the linearised advective transport operator.

    Args:
        domain (:class:`Domain`): the model's domain object, containing the
            mesh and the compatible function spaces.
        test (:class:`TestFunction`): the test function.
        qbar (:class:`ufl.Expr`): the variable to be transported.

    Returns:
        :class:`LabelledForm`: a labelled transport form.
    """

    ubar = Function(domain.spaces("HDiv"))

    # TODO: why is there a k here?
    L = test*dot(ubar, domain.k)*dot(domain.k, grad(qbar))*dx

    form = transporting_velocity(L, ubar)

    return transport(form, TransportEquationType.advective)


def linear_continuity_form(domain, test, qbar, facet_term=False):
    """
    The form corresponding to the linearised continuity transport operator.

    Args:
        domain (:class:`Domain`): the model's domain object, containing the
            mesh and the compatible function spaces.
        test (:class:`TestFunction`): the test function.
        qbar (:class:`ufl.Expr`): the variable to be transported.
        facet_term (bool, optional): whether to include interior facet terms.
            Defaults to False.

    Returns:
        :class:`LabelledForm`: a labelled transport form.
    """

    Vu = domain.spaces("HDiv")
    ubar = Function(Vu)

    L = qbar*test*div(ubar)*dx

    if facet_term:
        n = FacetNormal(domain.mesh)
        Vu = domain.spaces("HDiv")
        dS_ = (dS_v + dS_h) if Vu.extruded else dS
        L += jump(ubar*test, n)*avg(qbar)*dS_

    form = transporting_velocity(L, ubar)

    return transport(form, TransportEquationType.conservative)


def advection_form(domain, test, q, ibp=IntegrateByParts.ONCE, outflow=False):
    u"""
    The form corresponding to the advective transport operator.

    This discretises (u.∇)q, for transporting velocity u and transported
    variable q. An upwind discretisation is used for the facet terms when the
    form is integrated by parts.

    Args:
        domain (:class:`Domain`): the model's domain object, containing the
            mesh and the compatible function spaces.
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the variable to be transported.
        ibp (:class:`IntegrateByParts`, optional): an enumerator representing
            the number of times to integrate by parts. Defaults to
            `IntegrateByParts.ONCE`.
        outflow (bool, optional): whether to include outflow at the domain
            boundaries, through exterior facet terms. Defaults to False.

    Raises:
        ValueError: Can only use outflow option when the integration by parts
            option is not "never".

    Returns:
        class:`LabelledForm`: a labelled transport form.
    """

    if outflow and ibp == IntegrateByParts.NEVER:
        raise ValueError("outflow is True and ibp is None are incompatible options")
    Vu = domain.spaces("HDiv")
    dS_ = (dS_v + dS_h) if Vu.extruded else dS
    ubar = Function(Vu)

    if ibp == IntegrateByParts.ONCE:
        L = -inner(div(outer(test, ubar)), q)*dx
    else:
        L = inner(outer(test, ubar), grad(q))*dx

    if ibp != IntegrateByParts.NEVER:
        n = FacetNormal(domain.mesh)
        un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))

        L += dot(jump(test), (un('+')*q('+') - un('-')*q('-')))*dS_

        if ibp == IntegrateByParts.TWICE:
            L -= (inner(test('+'), dot(ubar('+'), n('+'))*q('+'))
                  + inner(test('-'), dot(ubar('-'), n('-'))*q('-')))*dS_

    if outflow:
        n = FacetNormal(domain.mesh)
        un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
        L += test*un*q*(ds_v + ds_t + ds_b)

    form = transporting_velocity(L, ubar)

    return ibp_label(transport(form, TransportEquationType.advective), ibp)


def continuity_form(domain, test, q, ibp=IntegrateByParts.ONCE, outflow=False):
    u"""
    The form corresponding to the continuity transport operator.

    This discretises ∇.(u*q), for transporting velocity u and transported
    variable q. An upwind discretisation is used for the facet terms when the
    form is integrated by parts.

    Args:
        domain (:class:`Domain`): the model's domain object, containing the
            mesh and the compatible function spaces.
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the variable to be transported.
        ibp (:class:`IntegrateByParts`, optional): an enumerator representing
            the number of times to integrate by parts. Defaults to
            `IntegrateByParts.ONCE`.
        outflow (bool, optional): whether to include outflow at the domain
            boundaries, through exterior facet terms. Defaults to False.

    Raises:
        ValueError: Can only use outflow option when the integration by parts
            option is not "never".

    Returns:
        class:`LabelledForm`: a labelled transport form.
    """

    if outflow and ibp == IntegrateByParts.NEVER:
        raise ValueError("outflow is True and ibp is None are incompatible options")
    Vu = domain.spaces("HDiv")
    dS_ = (dS_v + dS_h) if Vu.extruded else dS
    ubar = Function(Vu)

    if ibp == IntegrateByParts.ONCE:
        L = -inner(grad(test), outer(q, ubar))*dx
    else:
        L = inner(test, div(outer(q, ubar)))*dx

    if ibp != IntegrateByParts.NEVER:
        n = FacetNormal(domain.mesh)
        un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))

        L += dot(jump(test), (un('+')*q('+') - un('-')*q('-')))*dS_

        if ibp == IntegrateByParts.TWICE:
            L -= (inner(test('+'), dot(ubar('+'), n('+'))*q('+'))
                  + inner(test('-'), dot(ubar('-'), n('-'))*q('-')))*dS_

    if outflow:
        n = FacetNormal(domain.mesh)
        un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
        L += test*un*q*(ds_v + ds_t + ds_b)

    form = transporting_velocity(L, ubar)

    return ibp_label(transport(form, TransportEquationType.conservative), ibp)


def vector_manifold_advection_form(domain, test, q, ibp=IntegrateByParts.ONCE, outflow=False):
    """
    Form for advective transport operator including vector manifold correction.

    This creates the form corresponding to the advective transport operator, but
    also includes a correction for the treatment of facet terms when the
    transported field is vector-valued and the mesh is curved. This correction
    is based on that of Bernard, Remacle et al (2009).

    Args:
        domain (:class:`Domain`): the model's domain object, containing the
            mesh and the compatible function spaces.
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the variable to be transported.
        ibp (:class:`IntegrateByParts`, optional): an enumerator representing
            the number of times to integrate by parts. Defaults to
            `IntegrateByParts.ONCE`.
        outflow (bool, optional): whether to include outflow at the domain
            boundaries, through exterior facet terms. Defaults to False.

    Returns:
        class:`LabelledForm`: a labelled transport form.
    """

    L = advection_form(domain, test, q, ibp, outflow)

    # TODO: there should maybe be a restriction on IBP here
    Vu = domain.spaces("HDiv")
    dS_ = (dS_v + dS_h) if Vu.extruded else dS
    ubar = Function(Vu)
    n = FacetNormal(domain.mesh)
    un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
    L += un('+')*inner(test('-'), n('+')+n('-'))*inner(q('+'), n('+'))*dS_
    L += un('-')*inner(test('+'), n('+')+n('-'))*inner(q('-'), n('-'))*dS_

    return L


def vector_manifold_continuity_form(domain, test, q, ibp=IntegrateByParts.ONCE, outflow=False):
    """
    Form for continuity transport operator including vector manifold correction.

    This creates the form corresponding to the continuity transport operator,
    but also includes a correction for the treatment of facet terms when the
    transported field is vector-valued and the mesh is curved. This correction
    is based on that of Bernard, Remacle et al (2009).

    Args:
        domain (:class:`Domain`): the model's domain object, containing the
            mesh and the compatible function spaces.
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the variable to be transported.
        ibp (:class:`IntegrateByParts`, optional): an enumerator representing
            the number of times to integrate by parts. Defaults to
            `IntegrateByParts.ONCE`.
        outflow (bool, optional): whether to include outflow at the domain
            boundaries, through exterior facet terms. Defaults to False.

    Returns:
        class:`LabelledForm`: a labelled transport form.
    """

    L = continuity_form(domain, test, q, ibp, outflow)

    Vu = domain.spaces("HDiv")
    dS_ = (dS_v + dS_h) if Vu.extruded else dS
    ubar = Function(Vu)
    n = FacetNormal(domain.mesh)
    un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
    L += un('+')*inner(test('-'), n('+')+n('-'))*inner(q('+'), n('+'))*dS_
    L += un('-')*inner(test('+'), n('+')+n('-'))*inner(q('-'), n('-'))*dS_

    form = transporting_velocity(L, ubar)

    return transport(form)


def vector_invariant_form(domain, test, q, ibp=IntegrateByParts.ONCE):
    u"""
    The form corresponding to the vector invariant transport operator.

    The self-transporting transport operator for a vector-valued field u can be
    written as circulation and kinetic energy terms:
    (u.∇)u = (∇×u)×u + (1/2)∇u^2

    When the transporting field u and transported field q are similar, we write
    this as:
    (u.∇)q = (∇×q)×u + (1/2)∇(u.q)

    This form discretises this final equation, using an upwind discretisation
    when integrating by parts.

    Args:
        domain (:class:`Domain`): the model's domain object, containing the
            mesh and the compatible function spaces.
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the variable to be transported.
        ibp (:class:`IntegrateByParts`, optional): an enumerator representing
            the number of times to integrate by parts. Defaults to
            `IntegrateByParts.ONCE`.

    Raises:
        NotImplementedError: the specified integration by parts is not 'once'.

    Returns:
        class:`LabelledForm`: a labelled transport form.
    """

    Vu = domain.spaces("HDiv")
    dS_ = (dS_v + dS_h) if Vu.extruded else dS
    ubar = Function(Vu)
    n = FacetNormal(domain.mesh)
    Upwind = 0.5*(sign(dot(ubar, n))+1)

    if domain.mesh.topological_dimension() == 3:

        if ibp != IntegrateByParts.ONCE:
            raise NotImplementedError

        # <w,curl(u) cross ubar + grad( u.ubar)>
        # =<curl(u),ubar cross w> - <div(w), u.ubar>
        # =<u,curl(ubar cross w)> -
        #      <<u_upwind, [[n cross(ubar cross w)cross]]>>

        both = lambda u: 2*avg(u)

        L = (
            inner(q, curl(cross(ubar, test)))*dx
            - inner(both(Upwind*q),
                    both(cross(n, cross(ubar, test))))*dS_
        )

    else:

        perp = domain.perp
        if domain.on_sphere:
            outward_normals = CellNormal(domain.mesh)
            perp_u_upwind = lambda q: Upwind('+')*cross(outward_normals('+'), q('+')) + Upwind('-')*cross(outward_normals('-'), q('-'))
        else:
            perp_u_upwind = lambda q: Upwind('+')*perp(q('+')) + Upwind('-')*perp(q('-'))

        if ibp == IntegrateByParts.ONCE:
            L = (
                -inner(perp(grad(inner(test, perp(ubar)))), q)*dx
                - inner(jump(inner(test, perp(ubar)), n),
                        perp_u_upwind(q))*dS_
            )
        else:
            L = (
                (-inner(test, div(perp(q))*perp(ubar)))*dx
                - inner(jump(inner(test, perp(ubar)), n),
                        perp_u_upwind(q))*dS_
                + jump(inner(test,
                             perp(ubar))*perp(q), n)*dS_
            )

    L -= 0.5*div(test)*inner(q, ubar)*dx

    form = transporting_velocity(L, ubar)

    return transport(form, TransportEquationType.vector_invariant)


def kinetic_energy_form(domain, test, q):
    u"""
    The form corresponding to the kinetic energy term.

    Writing the kinetic energy term as (1/2)∇u^2, if the transported variable
    q is similar to the transporting variable u then this can be written as:
    (1/2)∇(u.q).

    Args:
        domain (:class:`Domain`): the model's domain object, containing the
            mesh and the compatible function spaces.
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the variable to be transported.

    Returns:
        class:`LabelledForm`: a labelled transport form.
    """

    ubar = Function(domain.spaces("HDiv"))
    L = div(test)*inner(q, ubar)*dx

    form = transporting_velocity(L, ubar)

    return transport(form, TransportEquationType.vector_invariant)


def advection_equation_circulation_form(domain, test, q,
                                        ibp=IntegrateByParts.ONCE):
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
        domain (:class:`Domain`): the model's domain object, containing the
            mesh and the compatible function spaces.
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the variable to be transported.
        ibp (:class:`IntegrateByParts`, optional): an enumerator representing
            the number of times to integrate by parts. Defaults to
            `IntegrateByParts.ONCE`.

    Raises:
        NotImplementedError: the specified integration by parts is not 'once'.

    Returns:
        class:`LabelledForm`: a labelled transport form.
    """

    form = (
        vector_invariant_form(domain, test, q, ibp=ibp)
        - kinetic_energy_form(domain, test, q)
    )

    return form
