"""
Defines TransportMethod objects, which are used to solve a transport problem.
"""

from firedrake import split
from gusto.configuration import IntegrateByParts, TransportEquationType
from gusto.fml import Term, keep, drop
from gusto.labels import prognostic, transport, transporting_velocity
from gusto.transport_forms import *

__all__ = ["DGUpwind"]

class TransportMethod(object):
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
        self.equation = equation
        self.variable = variable
        self.domain = self.equation.domain

        if hasattr(equation, "field_names"):
            # Equation with multiple prognostic variables
            variable_idx = equation.field_names.index(variable)
            self.test = equation.tests[variable_idx]
            self.field = split(equation.X)[variable_idx]
        else:
            self.field = equation.X
            self.test = equation.test

        # Find the original transport term to be used, which we use to extract
        # information about the transport equation type
        original_form = equation.residual.label_map(
            lambda t: t.has_label(transport) and t.get(prognostic) == variable,
            map_if_true=keep, map_if_false=drop)

        num_terms = len(original_form.terms)
        assert num_terms == 1, \
            f'Unable to find transport term for {variable}. {num_terms} found'

        self.transport_equation_type = original_form.terms[0].get(transport)

    def replace_transport_form(self, equation):
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
