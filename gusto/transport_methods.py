"""
Defines TransportMethod objects, which are used to solve a transport problem.
"""

from firedrake import (dx, dS, dS_v, dS_h, ds_t, ds_b, dot, inner, outer, jump,
                       grad, div, FacetNormal, Function)
from gusto.configuration import IntegrateByParts, TransportEquationType
from gusto.fml import Term, keep, drop
from gusto.labels import prognostic, transport, transporting_velocity, ibp_label
from gusto.spatial_methods import SpatialMethod

__all__ = ["DGUpwind"]

class TransportMethod(SpatialMethod):
    """
    The base object for describing a transport scheme.
    """

    def __init__(self, equation, variable):
        """
        Args:
            equation (:class:`PrognosticEquation`): the equation, which includes
                a transport term.
            variable (str): name of the variable to set the transport scheme for
        """

        # Inherited init method extracts original term to be replaced
        super().__init__(equation, variable, transport)

        self.transport_equation_type = self.original_form.terms[0].get(transport)

    def replace_form(self, equation):
        """
        Replaces the form for the transport term in the equation with the
        form for the transport discretisation.

        Args:
            equation (:class:`PrognosticEquation`): the equation or scheme whose
                transport term should be replaced with the transport term of
                this discretisation.
        """

        # We need to take care to replace the term with all the same labels,
        # except the label for the transporting velocity
        # This is easiest to do by extracting the transport term itself
        original_form = equation.residual.label_map(
            lambda t: t.has_label(transport) and t.get(prognostic) == self.variable,
            map_if_true=keep, map_if_false=drop
        )
        original_term = original_form.terms[0]

        # Update transporting velocity
        new_transporting_velocity = self.form.terms[0].get(transporting_velocity)
        original_term = transporting_velocity.update_value(original_term, new_transporting_velocity)

        # Create new term
        new_term = Term(self.form.form, original_term.labels)

        # Replace original term with new term
        equation.residual = equation.residual.label_map(
            lambda t: t.has_label(transport) and t.get(prognostic) == self.variable,
            map_if_true=lambda t: new_term)


def upwind_advection_form(domain, test, q, ibp=IntegrateByParts.ONCE, outflow=False):
    u"""
    The form corresponding to the DG upwind advective transport operator.

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


def upwind_continuity_form(domain, test, q, ibp=IntegrateByParts.ONCE, outflow=False):
    u"""
    The form corresponding to the DG upwind continuity transport operator.

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

    L = upwind_advection_form(domain, test, q, ibp, outflow)

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

    L = upwind_continuity_form(domain, test, q, ibp, outflow)

    Vu = domain.spaces("HDiv")
    dS_ = (dS_v + dS_h) if Vu.extruded else dS
    ubar = Function(Vu)
    n = FacetNormal(domain.mesh)
    un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
    L += un('+')*inner(test('-'), n('+')+n('-'))*inner(q('+'), n('+'))*dS_
    L += un('-')*inner(test('+'), n('+')+n('-'))*inner(q('-'), n('-'))*dS_

    form = transporting_velocity(L, ubar)

    return transport(form)


class DGUpwind(TransportMethod):
    """
    The Discontinuous Galerkin Upwind transport scheme.

    Discretises the gradient of a field weakly, taking the upwind value of the
    transported variable at facets.
    """
    def __init__(self, equation, variable, ibp=IntegrateByParts.ONCE,
                 vector_manifold_correction=False, outflow=False):
        """
        Args:
            equation (:class:`PrognosticEquation`): the equation, which includes
                a transport term.
            variable (str): name of the variable to set the transport scheme for
            ibp (:class:`IntegrateByParts`, optional): an enumerator for how
                many times to integrate by parts. Defaults to `ONCE`.
            vector_manifold_correction (bool, optional): whether to include a
                vector manifold correction term. Defaults to False.
            outflow (bool, optional): whether to include outflow at the domain
                boundaries, through exterior facet terms. Defaults to False.
        """

        super().__init__(equation, variable)
        self.ibp = ibp
        self.vector_manifold_correction = vector_manifold_correction
        self.outflow = outflow

        # -------------------------------------------------------------------- #
        # Determine appropriate form to use
        # -------------------------------------------------------------------- #

        if self.transport_equation_type == TransportEquationType.advective:
            if vector_manifold_correction:
                form = vector_manifold_advection_form(self.domain, self.test,
                                                      self.field, ibp=ibp,
                                                      outflow=outflow)
            else:
                form = upwind_advection_form(self.domain, self.test, self.field,
                                             ibp=ibp, outflow=outflow)

        elif self.transport_equation_type == TransportEquationType.conservative:
            if vector_manifold_correction:
                form = vector_manifold_continuity_form(self.domain, self.test,
                                                       self.field, ibp=ibp,
                                                       outflow=outflow)
            else:
                form = upwind_continuity_form(self.domain, self.test, self.field,
                                              ibp=ibp, outflow=outflow)

        elif self.transport_equation_type == TransportEquationType.vector_invariant:
            if outflow:
                raise NotImplementedError('Outflow not implemented for upwind vector invariant')
            form = upwind_vector_invariant_form(self.domain, self.test, self.field, ibp=ibp)

        else:
            raise NotImplementedError('Upwind transport scheme has not been '
                                      + 'implemented for this transport equation type')

        self.form = form
