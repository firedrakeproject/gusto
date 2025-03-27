"""
Defines TransportMethod objects, which are used to solve a transport problem.
"""

from firedrake import (
    dx, dS, dS_v, dS_h, ds_t, ds_b, ds_v, dot, inner, outer, jump,
    grad, div, FacetNormal, Function, sign, avg, cross, curl, split
)
from firedrake.fml import Term, keep, drop
from gusto.core.configuration import IntegrateByParts, TransportEquationType
from gusto.core.labels import (
    prognostic, transport, transporting_velocity, ibp_label, mass_weighted,
    all_but_last, horizontal_transport, vertical_transport
)
from gusto.core.logging import logger
from gusto.spatial_methods.spatial_methods import SpatialMethod

__all__ = ["DefaultTransport", "DGUpwind", "SplitDGUpwind"]


# ---------------------------------------------------------------------------- #
# Base TransportMethod class
# ---------------------------------------------------------------------------- #
class TransportMethod(SpatialMethod):
    """
    The base object for describing a transport scheme.
    """

    def __init__(self, equation, variable, term_labels=[transport]):
        """
        Args:
            equation (:class:`PrognosticEquation`): the equation, which includes
                a transport term.
            variable (str): name of the variable to set the transport scheme for
            term_labels (list of :class:`Label`, optional): the label specifying
                which type of term to be discretised. Defaults to [transport].
        """

        # Inherited init method extracts original term to be replaced
        super().__init__(equation, variable, term_labels)

        # If this is term has a mass_weighted label, then we need to
        # use the tracer_conservative version of the transport method.
        if self.original_form.terms[0].has_label(mass_weighted):
            self.transport_equation_type = TransportEquationType.tracer_conservative
        else:
            self.transport_equation_type = self.original_form.terms[0].get(transport)

        if self.transport_equation_type == TransportEquationType.tracer_conservative:
            # Extract associated density of the variable
            tracer = next(x for x in self.equation.active_tracers if x.name == variable)
            density_idx = self.equation.field_names.index(tracer.density_name)
            self.conservative_density = split(self.equation.X)[density_idx]

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

        if len(original_form.terms) == 0:
            # This is likely not the appropriate equation so skip
            logger.warning(f'No transport term found for {self.variable} in '
                           + 'this equation. Skipping.')

        elif len(original_form.terms) == 1:
            # Replace form
            original_term = original_form.terms[0]

            # Update transporting velocity
            new_transporting_velocity = self.form.terms[0].get(transporting_velocity)
            original_term = transporting_velocity.update_value(original_term, new_transporting_velocity)

            # Create new term
            new_term = Term(self.form.form, original_term.labels)

            # Add all_but_last form
            if hasattr(self, "all_but_last_form"):
                new_term = all_but_last(new_term, self.all_but_last_form)

            # Check if this is a conservative transport
            if original_term.has_label(mass_weighted):
                # Extract the original and discretised mass_weighted terms
                original_mass_weighted_term = original_term.get(mass_weighted).terms[0]
                new_mass_weighted = self.form.terms[0].get(mass_weighted)

                # Ensure the correct labels for the new mass weighted term
                new_mass_weighted_term = Term(new_mass_weighted.form, original_mass_weighted_term.labels)
                # Update the mass weighted transporting velocity
                new_mass_weighted_transporting_velocity = new_mass_weighted.terms[0].get(transporting_velocity)
                new_mass_weighted_term = transporting_velocity.update_value(new_mass_weighted_term, new_mass_weighted_transporting_velocity)

                # Add the discretised mass weighted transport term as the
                # new mass weighted label.
                new_term = mass_weighted.update_value(new_term, new_mass_weighted_term)

            # Replace original term with new term
            equation.residual = equation.residual.label_map(
                lambda t: t.has_label(transport) and t.get(prognostic) == self.variable,
                map_if_true=lambda t: new_term)

        else:
            horizontal_form = equation.residual.label_map(
                lambda t: t.has_label(transport) and t.has_label(horizontal_transport) and t.get(prognostic) == self.variable,
                map_if_true=keep, map_if_false=drop
            )
            vertical_form = equation.residual.label_map(
                lambda t: t.has_label(transport) and t.has_label(vertical_transport) and t.get(prognostic) == self.variable,
                map_if_true=keep, map_if_false=drop
            )
            if len(horizontal_form.terms) == 1 and len(vertical_form.terms) == 1:

                # Replace forms
                horizontal_term = horizontal_form.terms[0]
                vertical_term = vertical_form.terms[0]

                # Update transporting velocity
                new_horizontal_transporting_velocity = self.form_h.terms[0].get(transporting_velocity)
                new_vertical_transporting_velocity = self.form_v.terms[0].get(transporting_velocity)
                horizontal_term = transporting_velocity.update_value(horizontal_term, new_horizontal_transporting_velocity)
                vertical_term = transporting_velocity.update_value(vertical_term, new_vertical_transporting_velocity)

                # Create new terms
                new_horizontal_term = Term(self.form_h.form, horizontal_term.labels)
                new_vertical_term = Term(self.form_v.form, vertical_term.labels)

                # Check if this is a conservative transport
                if horizontal_term.has_label(mass_weighted) or vertical_term.has_label(mass_weighted):
                    raise NotImplementedError('Mass weighted transport terms not yet supported for multiple terms')

                # Replace original terms with new terms
                equation.residual = equation.residual.label_map(
                    lambda t: t.has_label(transport) and t.has_label(horizontal_transport) and t.get(prognostic) == self.variable,
                    map_if_true=lambda _: new_horizontal_term)

                equation.residual = equation.residual.label_map(
                    lambda t: t.has_label(transport) and t.has_label(vertical_transport) and t.get(prognostic) == self.variable,
                    map_if_true=lambda _: new_vertical_term)
            else:
                raise RuntimeError('Found multiple transport terms for the same '
                                   'variable in the equation where there should only be one')


# ---------------------------------------------------------------------------- #
# TransportMethod for using underlying default transport form
# ---------------------------------------------------------------------------- #
class DefaultTransport(TransportMethod):
    """
    Performs no manipulation of the transport form, so the scheme is simply
    based on the transport terms that are declared when the equation is set up.
    """
    def __init__(self, equation, variable):
        """
        Args:
            equation (:class:`PrognosticEquation`): the equation, which includes
                a transport term.
            variable (str): name of the variable to set the transport scheme for
        """

        super().__init__(equation, variable)

    def replace_form(self, equation):
        """
        In theory replaces the transport form in the equation, but in this case
        does nothing.

        Args:
            equation (:class:`PrognosticEquation`): the equation or scheme whose
                transport term should (not!) be replaced with the transport term
                of this discretisation.
        """
        pass


# ---------------------------------------------------------------------------- #
# Class for DG Upwind transport methods
# ---------------------------------------------------------------------------- #
class DGUpwind(TransportMethod):
    """
    The Discontinuous Galerkin Upwind transport scheme.

    Discretises the gradient of a field weakly, taking the upwind value of the
    transported variable at facets.
    """
    def __init__(self, equation, variable, ibp=IntegrateByParts.ONCE,
                 vector_manifold_correction=False, outflow=False,
                 advective_then_flux=False):
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
            advective_then_flux (bool, optional): whether to use the advective-
                then-flux formulation. This uses the advective form of the
                transport equation for all but the last steps of some
                (potentially subcycled) Runge-Kutta scheme, before using the
                conservative form for the final step to deliver a mass-
                conserving increment. This option only makes sense to use with
                Runge-Kutta, and should be used with the "linear" Runge-Kutta
                formulation. Defaults to False, in which case the conservative
                form is used for every step.
        """

        super().__init__(equation, variable)
        self.ibp = ibp
        self.vector_manifold_correction = vector_manifold_correction
        self.outflow = outflow

        if (advective_then_flux
                and self.transport_equation_type != TransportEquationType.conservative):
            raise ValueError(
                'DG Upwind: advective_then_flux form can only be used with '
                + 'the conservative form of the transport equation'
            )

        # -------------------------------------------------------------------- #
        # Determine appropriate form to use
        # -------------------------------------------------------------------- #
        # first check for 1d mesh and scalar velocity space
        if equation.domain.mesh.topological_dimension() == 1 and len(equation.domain.spaces("HDiv").shape) == 0:
            assert not vector_manifold_correction
            if self.transport_equation_type == TransportEquationType.advective:
                form = upwind_advection_form_1d(
                    self.domain, self.test, self.field, ibp=ibp,
                    outflow=outflow
                )
            elif self.transport_equation_type == TransportEquationType.conservative:
                form = upwind_continuity_form_1d(
                    self.domain, self.test, self.field, ibp=ibp,
                    outflow=outflow
                )

        else:
            if self.transport_equation_type == TransportEquationType.advective:
                if vector_manifold_correction:
                    form = vector_manifold_advection_form(
                        self.domain, self.test, self.field, ibp=ibp,
                        outflow=outflow
                    )
                else:
                    form = upwind_advection_form(
                        self.domain, self.test, self.field, ibp=ibp,
                        outflow=outflow
                    )

            elif self.transport_equation_type == TransportEquationType.conservative:
                if vector_manifold_correction:
                    form = vector_manifold_continuity_form(
                        self.domain, self.test, self.field, ibp=ibp,
                        outflow=outflow
                    )
                else:
                    form = upwind_continuity_form(
                        self.domain, self.test, self.field, ibp=ibp,
                        outflow=outflow
                    )

                if advective_then_flux and vector_manifold_correction:
                    self.all_but_last_form = vector_manifold_advection_form(
                        self.domain, self.test, self.field, ibp=ibp,
                        outflow=outflow
                    )

                elif advective_then_flux:
                    self.all_but_last_form = upwind_advection_form(
                        self.domain, self.test, self.field, ibp=ibp,
                        outflow=outflow
                    )

            elif self.transport_equation_type == TransportEquationType.circulation:
                if outflow:
                    raise NotImplementedError('Outflow not implemented for upwind circulation')
                form = upwind_circulation_form(self.domain, self.test,
                                               self.field, ibp=ibp)

            elif self.transport_equation_type == TransportEquationType.vector_invariant:
                if outflow:
                    raise NotImplementedError('Outflow not implemented for upwind vector invariant')
                form = upwind_vector_invariant_form(self.domain, self.test,
                                                    self.field, ibp=ibp)

            elif self.transport_equation_type == TransportEquationType.tracer_conservative:
                mass_weighted_form = upwind_tracer_conservative_form(self.domain, self.test,
                                                                     self.field,
                                                                     self.conservative_density,
                                                                     ibp=ibp)
                advective_form = upwind_advection_form(self.domain, self.test,
                                                       self.field,
                                                       ibp=ibp)

                # Store the conservative transport form in the mass_weighted label,
                # but by default use an advective form.
                form = mass_weighted(advective_form, mass_weighted_form)
            else:
                raise NotImplementedError('Upwind transport scheme has not been '
                                          + 'implemented for this transport equation type')
        self.form = form


class SplitDGUpwind(TransportMethod):
    """
    The Discontinuous Galerkin Upwind transport scheme applied separately in the
    horizontal and vertical directions.
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

        super().__init__(equation, variable, [horizontal_transport, vertical_transport])
        self.ibp = ibp
        self.vector_manifold_correction = vector_manifold_correction
        self.outflow = outflow

        # -------------------------------------------------------------------- #
        # Determine appropriate form to use
        # -------------------------------------------------------------------- #
        # first check for 1d mesh and scalar velocity space
        if equation.domain.mesh.topological_dimension() == 1 and len(equation.domain.spaces("HDiv").shape) == 0:
            assert not vector_manifold_correction
            raise ValueError('You cannot do horizontal and vertical splitting in 1D')
        else:
            if self.transport_equation_type == TransportEquationType.advective:

                form_h, form_v = split_upwind_advection_form(self.domain, self.test,
                                                             self.field,
                                                             ibp=ibp, outflow=outflow)

            else:
                raise NotImplementedError('Split hv Upwind transport scheme has not been '
                                          + 'implemented for this transport equation type')
        self.form_v = form_v
        self.form_h = form_h


# ---------------------------------------------------------------------------- #
# Forms for DG Upwind transport
# ---------------------------------------------------------------------------- #
def split_upwind_advection_form(domain, test, q, ibp=IntegrateByParts.ONCE, outflow=False):
    u"""
    The forms corresponding to the DG upwind advective transport operator in
    the horizontal and vertical directions.
    This discretises u_h.(∇_h)q and w dq/dz, for transporting velocity u and transported
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
    k = domain.k
    quad = domain.max_quad_degree
    dS_ = (dS_v(degree=quad) + dS_h(degree=quad)) if Vu.extruded else dS
    ubar = Function(Vu)
    ubar_v = k*inner(ubar, k)
    ubar_h = ubar - ubar_v

    if ibp == IntegrateByParts.ONCE:
        L_h = -inner(div(outer(test, ubar_h)), q)*dx(degree=quad)
        L_v = -inner(div(outer(test, ubar_v)), q)*dx(degree=quad)
    else:
        L_h = inner(outer(test, ubar_h), grad(q))*dx(degree=quad)
        L_v = inner(outer(test, ubar_v), grad(q))*dx(degree=quad)

    if ibp != IntegrateByParts.NEVER:
        n = FacetNormal(domain.mesh)
        un_h = 0.5*(dot(ubar_h, n) + abs(dot(ubar_h, n)))

        L_h += dot(jump(test), (un_h('+')*q('+') - un_h('-')*q('-')))*dS_

        un_v = 0.5*(dot(ubar_v, n) + abs(dot(ubar_v, n)))

        L_v += dot(jump(test), (un_v('+')*q('+') - un_v('-')*q('-')))*dS_

        if ibp == IntegrateByParts.TWICE:
            L_h -= (inner(test('+'), dot(ubar_h('+'), n('+')) * q('+'))
                    + inner(test('-'), dot(ubar_h('-'), n('-')) * q('-'))) * dS_

            L_v -= (inner(test('+'), dot(ubar_v('+'), n('+')) * q('+'))
                    + inner(test('-'), dot(ubar_v('-'), n('-')) * q('-'))) * dS_

    if outflow:
        n = FacetNormal(domain.mesh)
        un_h = 0.5*(dot(ubar_h, n) + abs(dot(ubar_h, n)))
        L_h += test*un_h*q*(ds_v + ds_t + ds_b)

        un_v = 0.5*(dot(ubar_v, n) + abs(dot(ubar_v, n)))
        L_v += test*un_v*q*(ds_v + ds_t + ds_b)

    form_h = transporting_velocity(L_h, ubar)
    form_v = transporting_velocity(L_v, ubar)
    labelled_form_h = ibp_label(transport(form_h, TransportEquationType.advective), ibp)
    labelled_form_v = ibp_label(transport(form_v, TransportEquationType.advective), ibp)
    return labelled_form_h, labelled_form_v


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
    quad = domain.max_quad_degree
    dS_ = (dS_v(degree=quad) + dS_h(degree=quad)) if Vu.extruded else dS(degree=quad)
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


def upwind_advection_form_1d(domain, test, q, ibp=IntegrateByParts.ONCE,
                             outflow=False):
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
    ubar = Function(Vu)
    n = FacetNormal(domain.mesh)
    un = 0.5*(ubar * n[0] + abs(ubar * n[0]))
    quad = domain.max_quad_degree
    dS_ = dS(degree=quad)

    if ibp == IntegrateByParts.ONCE:
        L = -(test * ubar).dx(0) * q * dx + \
            jump(test) * (un('+')*q('+') - un('-')*q('-'))*dS_
    else:
        raise NotImplementedError("1d advection form only implemented with option ibp=IntegrateByParts.ONCE")

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
    quad = domain.max_quad_degree
    dS_ = (dS_v(degree=quad) + dS_h(degree=quad)) if Vu.extruded else dS(degree=quad)
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


def upwind_continuity_form_1d(domain, test, q, ibp=IntegrateByParts.ONCE,
                              outflow=False):
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
    ubar = Function(Vu)
    n = FacetNormal(domain.mesh)
    un = 0.5*(ubar * n[0] + abs(ubar * n[0]))
    quad = domain.max_quad_degree
    dS_ = dS(degree=quad)

    if ibp == IntegrateByParts.ONCE:
        L = -test.dx(0) * q * ubar * dx \
            + jump(test) * (un('+')*q('+') - un('-')*q('-')) * dS_
    else:
        raise NotImplementedError("1d continuity form only implemented with option ibp=IntegrateByParts.ONCE")

    form = transporting_velocity(L, ubar)

    return ibp_label(transport(form, TransportEquationType.conservative), ibp)


def upwind_tracer_conservative_form(domain, test, q, rho,
                                    ibp=IntegrateByParts.ONCE, outflow=False):
    u"""
    The form corresponding to the DG upwind continuity transport operator.

    This discretises ∇.(u*q*rho), for transporting velocity u, transported
    variable q, and its reference density, rho. Although the tracer q obeys an advection
    equation, the transport term is in a conservative form.

    Args:
        domain (:class:`Domain`): the model's domain object, containing the
            mesh and the compatible function spaces.
        test (:class:`TestFunction`): the test function.
        q (:class:`ufl.Expr`): the variable to be transported.
        rho (:class:`ufl.Expr`): the reference density for the tracer.
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
    quad = domain.max_quad_degree
    dS_ = (dS_v(degree=quad) + dS_h(degree=quad)) if Vu.extruded else dS(degree=quad)
    ubar = Function(Vu)

    if ibp == IntegrateByParts.ONCE:
        L = -inner(grad(test), outer(inner(q, rho), ubar))*dx
    else:
        L = inner(test, div(outer(inner(q, rho), ubar)))*dx

    if ibp != IntegrateByParts.NEVER:
        n = FacetNormal(domain.mesh)
        un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))

        L += dot(jump(test), (un('+')*q('+')*rho('+') - un('-')*q('-')*rho('-')))*dS_

        if ibp == IntegrateByParts.TWICE:
            L -= (inner(test('+'), dot(ubar('+'), n('+'))*q('+')*rho('+'))
                  + inner(test('-'), dot(ubar('-'), n('-'))*q('-')*rho('-')))*dS_

    if outflow:
        n = FacetNormal(domain.mesh)
        un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
        L += test*un*q*rho*(ds_v + ds_t + ds_b)

    form = transporting_velocity(L, ubar)

    return ibp_label(transport(form, TransportEquationType.tracer_conservative), ibp)


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
    quad = domain.max_quad_degree
    dS_ = (dS_v(degree=quad) + dS_h(degree=quad)) if Vu.extruded else dS(degree=quad)
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
    quad = domain.max_quad_degree
    dS_ = (dS_v(degree=quad) + dS_h(degree=quad)) if Vu.extruded else dS(degree=quad)
    ubar = Function(Vu)
    n = FacetNormal(domain.mesh)
    un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
    L += un('+')*inner(test('-'), n('+')+n('-'))*inner(q('+'), n('+'))*dS_
    L += un('-')*inner(test('+'), n('+')+n('-'))*inner(q('-'), n('-'))*dS_

    form = transporting_velocity(L, ubar)

    return transport(form)


def upwind_circulation_form(domain, test, q, ibp=IntegrateByParts.ONCE):
    u"""
    The form corresponding to the DG upwind vector circulation operator.

    The self-transporting transport operator for a vector-valued field u can be
    written as circulation and kinetic energy terms:
    (u.∇)u = (∇×u)×u + (1/2)∇u^2

    This form discretises the first term in this equation, (∇×u)×u, using an
    upwind discretisation when integrating by parts.

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
    quad = domain.max_quad_degree
    dS_ = (dS_v(degree=quad) + dS_h(degree=quad)) if Vu.extruded else dS(degree=quad)
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
            outward_normals = domain.outward_normals
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

    form = transporting_velocity(L, ubar)

    return transport(form, TransportEquationType.circulation)


def upwind_vector_invariant_form(domain, test, q, ibp=IntegrateByParts.ONCE):
    u"""
    The form corresponding to the DG upwind vector invariant transport operator.

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

    circulation_form = upwind_circulation_form(domain, test, q, ibp=ibp)
    ubar = circulation_form.terms[0].get(transporting_velocity)

    L = circulation_form.terms[0].form - 0.5*div(test)*inner(q, ubar)*dx
    form = transporting_velocity(L, ubar)

    return transport(form, TransportEquationType.vector_invariant)
